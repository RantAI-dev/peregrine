#!/usr/bin/env python3
"""The `modern-bert` family on the ARM host: Peregrine before/after S4.9n vs llama.cpp.

Cells: one text of 16, 128, 512 and 2 000 tokens, and one batch of 16 texts of
about 32 tokens. Five runs per cell; the table reports the median and the
min..max spread of `texts/s`. Standard library only.

  python run.py --out results.json
"""

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import urllib.request

HOME = os.path.expanduser("~")
LLAMA = Path(
    os.environ.get(
        "LLAMA_SERVER_BIN",
        f"{HOME}/.cache/peregrine-bench/llama.cpp-v0.9.0/build/bin/llama-server",
    )
)
MODELS = {
    "Q8_0": Path(f"{HOME}/models/gte-modernbert-base-Q8_0.gguf"),
    "F16": Path(f"{HOME}/models/gte-modernbert-base-F16.gguf"),
}
ENGINES = {
    "peregrine-before": Path(f"{HOME}/wt-s49n-base/target/release/peregrine"),
    "peregrine-after": Path(f"{HOME}/wt-s49n-neon/target/release/peregrine"),
    "llama.cpp": LLAMA,
}
THREADS = 10
SLOTS = 32
CTX = 2048
PORT_BASE = 18110


def post(url, payload, timeout=900.0):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def get(url, timeout=5.0):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def wait_ready(base, proc, timeout=180.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        if get(base + "/health") or get(base + "/v1/models"):
            return True
        time.sleep(0.25)
    return False


def n_tokens(base, text):
    r = post(base + "/tokenize", {"content": text, "add_special": True})
    return len(r["tokens"])


def fit_text(base, source_words, target):
    """The longest prefix of `source_words` whose token count is <= target."""
    lo, hi, best = 1, len(source_words), " ".join(source_words[:1])
    while lo <= hi:
        mid = (lo + hi) // 2
        text = " ".join(source_words[:mid])
        if n_tokens(base, text) <= target:
            best = text
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def build_texts(base, fixture):
    lines = [ln.split() for ln in fixture.read_text().splitlines() if ln.strip()]
    sources = []
    for ln in lines:
        sources.extend(ln)
    words = sources
    texts = {}
    for target in (16, 128, 512, 2000):
        texts[str(target)] = fit_text(base, words, target)
    batch = []
    for i in range(16):
        # A different window for each text: rotate the word stream.
        rotated = words[(i * 37) % len(words):] + words[:(i * 37) % len(words)]
        batch.append(fit_text(base, rotated, 32))
    texts["batch32"] = batch
    return texts


def start_engine(engine, model_path, port):
    base = f"http://127.0.0.1:{port}"
    log = open(f"/var/tmp/modern-bert-{engine.replace('.', '_')}-{port}.log", "w")
    if engine == "llama.cpp":
        cmd = [
            str(LLAMA), "-m", str(model_path), "--embedding", "--pooling", "cls",
            "-t", str(THREADS), "-c", str((CTX + 8) * SLOTS), "-b", "2048",
            "-ub", "2048", "-np", str(SLOTS), "-fa", "auto",
            "--host", "127.0.0.1", "--port", str(port),
        ]
    else:
        cmd = [
            str(ENGINES[engine]), "serve", "-m", str(model_path),
            "--host", "127.0.0.1", "--port", str(port), "-t", str(THREADS),
            "-c", str(CTX), "--slots", str(SLOTS), "--no-profile",
            # The exact cache of finished embedding vectors would turn a repeated
            # run into a lookup; every engine here computes every text every time.
            "--opt", "EMBED_CACHE_MB=0",
        ]
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    if not wait_ready(base, proc):
        log.close()
        proc.kill()
        raise SystemExit(f"{engine} failed to start; see /var/tmp/modern-bert-*.log")
    return proc, base, cmd


def embed(base, texts):
    t0 = time.perf_counter()
    reply = post(base + "/v1/embeddings", {
        "input": texts, "model": "embed", "encoding_format": "float",
    })
    wall = time.perf_counter() - t0
    rows = sorted(reply["data"], key=lambda d: d.get("index", 0))
    return wall, [r["embedding"] for r in rows]


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb)


def cell_texts(texts, name):
    return texts[name] if name == "batch32" else [texts[name]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--runs", type=int, default=5)
    args = ap.parse_args()
    fixture = Path(__file__).resolve().parents[3] / (
        "crates/peregrine-models/tests/fixtures/modern-bert/texts.txt"
    )
    cells = ["16", "128", "512", "2000", "batch32"]
    out = {"runs": args.runs, "threads": THREADS, "cells": cells, "models": {}}
    port = PORT_BASE
    for label, model_path in MODELS.items():
        out["models"][label] = {"path": str(model_path), "engines": {}}
        # Calibrate the texts once per model with llama-server's tokenizer.
        texts_file = Path(f"/var/tmp/modern-bert-texts-{label}.json")
        proc, base, cmd = start_engine("llama.cpp", model_path, port)
        port += 1
        try:
            if texts_file.exists():
                texts = json.loads(texts_file.read_text())
            else:
                texts = build_texts(base, fixture)
                texts_file.write_text(json.dumps(texts))
            cell_tokens = {
                name: sum(n_tokens(base, t) for t in cell_texts(texts, name))
                for name in cells
            }
        finally:
            proc.terminate()
            proc.wait()
        out["models"][label]["texts"] = dict(texts)
        out["models"][label]["cell_tokens"] = cell_tokens
        # All three engines stay up for the whole model: the rounds then
        # interleave them, so a frequency or thermal drift is not charged to one.
        servers = {}
        try:
            for engine in ENGINES:
                proc, base, cmd = start_engine(engine, model_path, port)
                port += 1
                servers[engine] = (proc, base, cmd)
            for engine, (_, base, _) in servers.items():
                # Warm-up, excluded from the timing.
                for name in cells:
                    embed(base, cell_texts(texts, name))
            samples = {engine: {name: [] for name in cells} for engine in ENGINES}
            vectors = {engine: {} for engine in ENGINES}
            order = list(ENGINES)
            for run in range(args.runs):
                for engine in order[run % len(order):] + order[: run % len(order)]:
                    base = servers[engine][1]
                    for name in cells:
                        t = cell_texts(texts, name)
                        wall, vecs = embed(base, t)
                        samples[engine][name].append(round(len(t) / wall, 4))
                        if run == 0:
                            vectors[engine][name] = vecs
            for engine, (_, _, cmd) in servers.items():
                cells_out = {}
                for name in cells:
                    v = samples[engine][name]
                    cells_out[name] = {
                        "texts": len(cell_texts(texts, name)),
                        "tokens": cell_tokens[name],
                        "texts_per_s": v,
                        "median": round(statistics.median(v), 4),
                        "min": min(v),
                        "max": max(v),
                        "spread_pct": round(100.0 * (max(v) - min(v)) / statistics.median(v), 2),
                    }
                out["models"][label]["engines"][engine] = {
                    "cmd": " ".join(cmd).replace(HOME, "~"),
                    "binary_version": (
                        lambda v: (v.stdout + v.stderr).strip().splitlines()[0]
                    )(
                        subprocess.run(
                            [str(ENGINES[engine]), "--version"],
                            capture_output=True, text=True,
                        )
                    ),
                    "cells": cells_out,
                    "vectors": vectors[engine],
                }
                print(f"{label} {engine}: " + ", ".join(
                    f"{name} {cells_out[name]['median']:.3f}" for name in cells), flush=True)
        finally:
            for proc, _, _ in servers.values():
                proc.terminate()
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
    # Cosine to llama.cpp, per text.
    for label in out["models"]:
        llama = out["models"][label]["engines"]["llama.cpp"]["vectors"]
        for engine in ENGINES:
            vecs = out["models"][label]["engines"][engine]["vectors"]
            cos = {}
            for name in cells:
                cs = [cosine(a, b) for a, b in zip(vecs[name], llama[name])]
                cos[name] = {"min": min(cs), "mean": sum(cs) / len(cs), "n": len(cs)}
            out["models"][label]["engines"][engine]["cosine_vs_llama"] = cos
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())
