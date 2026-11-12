#!/usr/bin/env python3
"""The ARM embedding table with the exact result cache off (2026-11-12).

Every row is timed on both engines in one process group: one fresh server per
engine per model, both servers up for the whole model, and the timed rounds
interleave the engines in a rotating order.  Peregrine always runs with
`--opt EMBED_CACHE_MB=0`; llama.cpp's embedding path has no such cache.

Cells: one text of 16, 128, 512 and 2 000 tokens (the last skipped for a file
whose own limit is 512), one batch of 32 texts of 128 tokens, and one mixed
batch of 32 texts whose lengths were drawn from 8..512 with seed 20261111.
`texts/s` is texts divided by the whole request's wall time, median of five
rounds.  Standard library only.

  python run.py --model minilm-f16 --out minilm-f16.json
  python run.py --all --out summary.json

The texts are built with llama-server's own `/tokenize` and are byte-identical
for both engines.
"""

import argparse
import json
import math
import os
import random
import re
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HOME = os.path.expanduser("~")
LLAMA = Path(
    os.environ.get(
        "LLAMA_SERVER_BIN",
        f"{HOME}/.cache/peregrine-bench/llama.cpp-v0.9.0/build/bin/llama-server",
    )
)
PEREGRINE = Path(os.environ.get("PEREGRINE_BIN", f"{HOME}/wt-arm-emb/target/release/peregrine"))
CORPUS = Path(__file__).resolve().parents[2] / "prompts" / "corpus.txt"
THREADS = 10
SLOTS = 32
MIX_SEED = 20261111
REQUESTS_PER_CELL = 3

# model key -> (path, llama pooling, ctx tokens, include 2 000)
MODELS = {
    "embeddinggemma-q8_0": (f"{HOME}/models/embed/embeddinggemma/embeddinggemma-Q8_0.gguf", "mean", 2048, True),
    "qwen3-embed-0.6b-q8_0": (f"{HOME}/models/embed/qwen3-embed-0.6b/qwen3-embed-0.6b-q8_0.gguf", "last", 2048, True),
    "qwen3-embed-0.6b-f16": (f"{HOME}/models/embed/qwen3-embed-0.6b/qwen3-embed-0.6b-f16.gguf", "last", 2048, True),
    "nemotron-3.5-embed-1b-q8_0": (f"{HOME}/models/embed/nemotron/Nemotron-3.5-Embed-1B-Q8_0.gguf", "mean", 2048, True),
    "bge-m3-q8_0": (f"{HOME}/models/embed/bge-m3-Q8_0.gguf", "cls", 2048, True),
    "gte-modernbert-base-f16": (f"{HOME}/models/gte-modernbert-base-F16.gguf", "cls", 2048, True),
    "gte-modernbert-base-q8_0": (f"{HOME}/models/gte-modernbert-base-Q8_0.gguf", "cls", 2048, True),
    "minilm-f16": (f"{HOME}/models/embed/minilm/all-MiniLM-L6-v2-ggml-model-f16.gguf", "mean", 512, False),
    "minilm-q8_0": (f"{HOME}/models/embed/minilm/all-MiniLM-L6-v2-Q8_0.gguf", "mean", 512, False),
}


def post(url, payload, timeout=900.0):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def alive(url, timeout=1.0):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def wait_ready(base, proc, timeout=240.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        if alive(base + "/health") or alive(base + "/v1/models"):
            return True
        time.sleep(0.2)
    return False


def n_tokens(base, text):
    return len(post(base + "/tokenize", {"content": text, "add_special": True})["tokens"])


def trim_to(tok, text, target, cap):
    """Trim `text` by characters to the token target (closest, exact preferred)."""
    lo, hi = 1, len(text)
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        chunk = text[:mid].rstrip()
        n = tok(chunk)
        if n <= cap:
            if best is None or abs(n - target) < best[0]:
                best = (abs(n - target), chunk, n)
            lo = mid + 1
        else:
            hi = mid - 1
    if best is None:
        best = (None, text[:1], tok(text[:1]))
    _, chunk, n = best
    if n != target:
        cut = len(chunk)
        for d in range(1, 120):
            for c in (cut - d, cut + d):
                if 1 <= c <= len(text):
                    cand = text[:c].rstrip()
                    cn = tok(cand)
                    if cn == target and cn <= cap:
                        return cand, cn
    return chunk, n


def fit(tok, words, start, target, cap):
    """A window from `start` with a token count close to `target`."""
    # Upper bound on words: ~3 tokens per word for these tokenizers, plus slack.
    lo, hi = 1, min(len(words) - start, target * 3 + 16)
    best = 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if tok(" ".join(words[start:start + mid])) <= cap:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    k = max(1, best - 4)
    text = " ".join(words[start:start + k])
    return trim_to(tok, text, target, cap)


def load_words():
    text = CORPUS.read_text(encoding="utf-8")
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'’\-]*|[,.;:!?]", text)
    return (words * 40)[:40000]


def build_texts(base, max_len):
    words = load_words()
    cap = max_len - 4  # room for special tokens
    tok = lambda s: n_tokens(base, s)

    def window(start, target):
        text, n = fit(tok, words, start % max(1, len(words) - 20), target, cap)
        return {"text": text, "tokens": n}

    lengths = [16, 128, 512] + ([2000] if max_len >= 2000 else [])
    texts = {}
    for L in lengths:
        texts[str(L)] = [window(L * 7 + i * 101, L) for i in range(8)]
    # The 32-text batch of 128 tokens: 32 distinct windows.
    texts["batch"] = [window(5000 + i * 173, 128) for i in range(32)]
    # The mixed batch: 32 lengths drawn from 8..512, one seed, then calibrated.
    rng = random.Random(MIX_SEED)
    texts["mixed"] = [window(20000 + i * 211, rng.randint(8, 512)) for i in range(32)]
    return texts, lengths


def start(engine, path, pooling, ctx, port, tag):
    base = f"http://127.0.0.1:{port}"
    log = open(f"/var/tmp/embcache-{tag}-{engine.replace('.', '_')}-{port}.log", "w")
    if engine == "llama.cpp":
        cmd = [
            str(LLAMA), "-m", str(path), "--embedding", "--pooling", pooling,
            "-t", str(THREADS), "-c", str((ctx + 8) * SLOTS), "-b", "2048",
            "-ub", "2048", "-np", str(SLOTS), "-fa", "auto",
            "--host", "127.0.0.1", "--port", str(port),
        ]
    else:
        cmd = [
            str(PEREGRINE), "serve", "-m", str(path),
            "--host", "127.0.0.1", "--port", str(port), "-t", str(THREADS),
            "-c", str(ctx), "--slots", str(SLOTS), "--no-profile",
            "--opt", "EMBED_CACHE_MB=0",
        ]
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    if not wait_ready(base, proc):
        log.close()
        proc.kill()
        raise SystemExit(f"{engine} failed to start; see /var/tmp/embcache-*.log")
    return proc, base, cmd


def embed(base, texts):
    t0 = time.perf_counter()
    reply = post(base + "/v1/embeddings", {
        "input": texts, "model": "embed", "encoding_format": "float",
    })
    wall = time.perf_counter() - t0
    if len(reply.get("data", [])) != len(texts):
        raise SystemExit("wrong number of vectors")
    return wall


def run_model(key, out, runs=REQUESTS_PER_CELL, only=None, raw_out=None):
    path, pooling, ctx, long = MODELS[key]
    path = Path(path)
    # Calibrate with llama-server first, on its own.
    port = 18400
    proc, base, _ = start("llama.cpp", path, pooling, ctx, port, key)
    try:
        texts, lengths = build_texts(base, ctx if long else 512)
    finally:
        proc.terminate()
        proc.wait()
    cells = [str(L) for L in lengths] + ["batch", "mixed"]
    if only:
        cells = [c for c in cells if c in only]

    def payload(cell, run):
        if cell in ("batch", "mixed"):
            return [t["text"] for t in texts[cell]]
        single = texts[cell]
        return [single[run % len(single)]["text"]]

    tokens = {c: sum(t["tokens"] for t in texts[c]) if c in ("batch", "mixed")
              else texts[c][0]["tokens"] for c in cells}
    servers = {}
    try:
        for i, engine in enumerate(("peregrine", "llama.cpp")):
            proc, base, cmd = start(engine, path, pooling, ctx, port + 1 + i, key)
            servers[engine] = (proc, base, cmd)
        for engine, (_, base, _) in servers.items():
            for c in cells:  # warm-up, excluded
                embed(base, payload(c, 0))
        samples = {e: {c: [] for c in cells} for e in servers}
        order = ["peregrine", "llama.cpp"]
        for run in range(runs):
            for engine in order[run % 2:] + order[: run % 2]:
                base = servers[engine][1]
                for c in cells:
                    texts_in = payload(c, run)
                    wall = embed(base, texts_in)
                    print(f"  {engine} run {run} {c}: {wall:.2f}s", flush=True)
                    samples[engine][c].append(round(len(texts_in) / wall, 3))
                    with open(str(raw_out) + "l", "a") as fh:
                        fh.write(json.dumps({"model": key, "run": run, "engine": engine,
                                             "cell": c, "texts": len(texts_in),
                                             "seconds": round(wall, 3),
                                             "texts_per_s": round(len(texts_in) / wall, 3)}) + "\n")
        out[key] = {
            "path": str(path),
            "pooling": pooling,
            "ctx": ctx,
            "cells": {},
        }
        for c in cells:
            v = samples["peregrine"][c]
            l = samples["llama.cpp"][c]
            out[key]["cells"][c] = {
                "texts": 1 if c not in ("batch", "mixed") else (32 if c == "batch" else 32),
                "tokens_per_text": (tokens[c] if c not in ("batch", "mixed")
                                    else round(tokens[c] / 32, 2)),
                "peregrine": v,
                "llama.cpp": l,
                "peregrine_median": round(statistics.median(v), 3),
                "llama_median": round(statistics.median(l), 3),
                "ratio": round(statistics.median(v) / statistics.median(l), 3),
                "peregrine_spread_pct": round(100 * (max(v) - min(v)) / statistics.median(v), 2),
                "llama_spread_pct": round(100 * (max(l) - min(l)) / statistics.median(l), 2),
            }
        for engine, (_, _, cmd) in servers.items():
            out[key][engine] = {
                "cmd": " ".join(cmd).replace(HOME, "~"),
                "version": subprocess.run([str(Path(cmd[0])) if engine == "llama.cpp"
                                           else str(PEREGRINE), "--version"],
                                          capture_output=True, text=True).stdout.strip().splitlines()[:1],
            }
        print(f"{key}: " + ", ".join(
            f"{c} {out[key]['cells'][c]['ratio']:.2f}x" for c in cells), flush=True)
    finally:
        for proc, _, _ in servers.values():
            proc.terminate()
            proc.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", type=int, default=REQUESTS_PER_CELL)
    ap.add_argument("--cells")
    args = ap.parse_args()
    keys = list(MODELS) if args.all else [args.model]
    out = {}
    for key in keys:
        only = args.cells.split(",") if args.cells else None
        run_model(key, out, args.runs, only, args.out)
        Path(args.out).write_text(
            json.dumps(out, indent=1).replace(HOME, "~"), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
