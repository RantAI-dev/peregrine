#!/usr/bin/env python3
"""Q8_0 encoders on the ARM host's kernel tiers: Peregrine per tier vs llama.cpp.

Cells: one text of 16, 128 and 512 tokens, and one batch of 16 texts of 32
tokens. Every arm stays up for a model and the rounds interleave them in a
rotating order, so a frequency or thermal drift is not charged to one engine.
The table reports the median of `--runs` runs and the min..max spread.
Standard library only.

  python3 run.py --out summary.json --runs 7
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
PEREGRINE = Path(
    os.environ.get("PEREGRINE_BIN", f"{HOME}/wt-s49o/target/release/peregrine")
)
PEREGRINE_BEFORE = Path(
    os.environ.get("PEREGRINE_BEFORE_BIN", "/var/tmp/q8probe/peregrine-before")
)
HERE = Path(__file__).resolve().parent

MODELS = {
    "gte-modernbert-base-Q8_0": {
        "path": Path(f"{HOME}/models/gte-modernbert-base-Q8_0.gguf"),
        "fixture": "crates/peregrine-models/tests/fixtures/modern-bert/texts.txt",
        "ctx": 2048,
        "pooling": "cls",
    },
    "gte-modernbert-base-F16": {
        "path": Path(f"{HOME}/models/gte-modernbert-base-F16.gguf"),
        "fixture": "crates/peregrine-models/tests/fixtures/modern-bert/texts.txt",
        "ctx": 2048,
        "pooling": "cls",
    },
    "EmbeddingGemma-300M-Q8_0": {
        "path": Path(f"{HOME}/models/embed/embeddinggemma/embeddinggemma-Q8_0.gguf"),
        "fixture": "crates/peregrine-models/tests/fixtures/modern-bert/texts.txt",
        "ctx": 2048,
        "pooling": "mean",
    },
    "all-MiniLM-L6-v2-Q8_0": {
        "path": Path(f"{HOME}/models/embed/minilm/all-MiniLM-L6-v2-Q8_0.gguf"),
        "fixture": "crates/peregrine-models/tests/fixtures/bert/texts.txt",
        "ctx": 512,
        "pooling": "mean",
    },
    "all-MiniLM-L6-v2-F16": {
        "path": Path(f"{HOME}/models/embed/minilm/all-MiniLM-L6-v2-ggml-model-f16.gguf"),
        "fixture": "crates/peregrine-models/tests/fixtures/bert/texts.txt",
        "ctx": 512,
        "pooling": "mean",
    },
}
ENGINES = {
    "peregrine-before": {"kind": "peregrine", "isa": "native", "binary": "PEREGRINE_BEFORE_BIN"},
    "peregrine-i8mm": {"kind": "peregrine", "isa": "native"},
    "peregrine-dotprod": {"kind": "peregrine", "isa": "neon-dotprod"},
    "peregrine-neon": {"kind": "peregrine", "isa": "neon"},
    "llama.cpp": {"kind": "llama"},
}
THREADS = 10
SLOTS = 32
# `PIN_CPUS` (e.g. `5-9,15-19`) wraps every engine in `taskset -c`: on this host the
# ten 3.9 GHz Cortex-X925 cores are 5-9 and 15-19 and the ten 2.8 GHz A725s are the
# rest, and an unpinned ten-thread process can be scheduled onto the slow cluster
# (measured 2.1x slower on the same binary). Empty = no pinning.
PIN = os.environ.get("PIN_CPUS", "")
PORT_BASE = 18210
CELLS = ["16", "128", "512", "batch32"]


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
    words = [w for ln in lines for w in ln]
    texts = {}
    for target in (16, 128, 512):
        texts[str(target)] = fit_text(base, words, target)
    batch = []
    for i in range(16):
        # A different window for each text: rotate the word stream.
        rotated = words[(i * 37) % len(words):] + words[:(i * 37) % len(words)]
        batch.append(fit_text(base, rotated, 32))
    texts["batch32"] = batch
    return texts


def start_engine(engine, model, port):
    spec = ENGINES[engine]
    base = f"http://127.0.0.1:{port}"
    tag = engine.replace(".", "_")
    log = open(f"/var/tmp/q8-encoders-{tag}-{port}.log", "w")
    if spec["kind"] == "llama":
        cmd = [
            str(LLAMA), "-m", str(model["path"]), "--embedding",
            "--pooling", model["pooling"], "-t", str(THREADS),
            "-c", str((model["ctx"] + 8) * SLOTS), "-b", "2048", "-ub", "2048",
            "-np", str(SLOTS), "-fa", "auto",
            "--host", "127.0.0.1", "--port", str(port),
        ]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    else:
        binary = PEREGRINE_BEFORE if spec.get("binary") else PEREGRINE
        cmd = [
            str(binary), "serve", "-m", str(model["path"]),
            "--host", "127.0.0.1", "--port", str(port), "-t", str(THREADS),
            "-c", str(model["ctx"]), "--slots", str(SLOTS), "--no-profile",
            # The exact cache of finished embedding vectors would turn a repeated
            # run into a lookup; every engine here computes every text every time.
            "--opt", "EMBED_CACHE_MB=0",
        ]
        env = dict(
            os.environ, PEREGRINE_ISA=spec["isa"], PEREGRINE_PACK_CACHE="0"
        )
    if PIN:
        cmd = ["taskset", "-c", PIN] + cmd
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    if not wait_ready(base, proc):
        log.close()
        proc.kill()
        raise SystemExit(f"{engine} failed to start; see /var/tmp/q8-encoders-*.log")
    return proc, base, cmd, env


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
    ap.add_argument("--out", default="summary.json")
    ap.add_argument("--runs", type=int, default=7)
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--engines", default=",".join(ENGINES))
    args = ap.parse_args()
    models = [m for m in args.models.split(",") if m]
    engines = [e for e in args.engines.split(",") if e]
    out = {
        "runs": args.runs,
        "threads": THREADS,
        "pin_cpus": PIN or None,
        "cells": CELLS,
        "models": {},
    }
    port = PORT_BASE
    for label in models:
        model = MODELS[label]
        out["models"][label] = {
            "path": str(model["path"]).replace(HOME, "~"),
            "bytes": model["path"].stat().st_size,
            "engines": {},
        }
        fixture = HERE.parents[1].parent / model["fixture"]
        # Calibrate the texts once per model with llama-server's tokenizer.
        texts_file = Path(f"/var/tmp/q8-encoders-texts-{label}.json")
        proc, base, cmd, _ = start_engine("llama.cpp", model, port)
        port += 1
        try:
            if texts_file.exists():
                texts = json.loads(texts_file.read_text())
            else:
                texts = build_texts(base, fixture)
                texts_file.write_text(json.dumps(texts))
            cell_tokens = {
                name: sum(n_tokens(base, t) for t in cell_texts(texts, name))
                for name in CELLS
            }
        finally:
            proc.terminate()
            proc.wait()
        out["models"][label]["texts"] = dict(texts)
        out["models"][label]["cell_tokens"] = cell_tokens
        # Every arm stays up for the whole model: the rounds then interleave them.
        servers = {}
        try:
            for engine in engines:
                proc, base, cmd, env = start_engine(engine, model, port)
                port += 1
                servers[engine] = (proc, base, cmd, env)
            for engine, (_, base, _, _) in servers.items():
                for name in CELLS:  # warm-up, excluded from the timing
                    embed(base, cell_texts(texts, name))
            samples = {engine: {name: [] for name in CELLS} for engine in engines}
            vectors = {engine: {} for engine in engines}
            for run in range(args.runs):
                order = engines[run % len(engines):] + engines[: run % len(engines)]
                for engine in order:
                    base = servers[engine][1]
                    for name in CELLS:
                        t = cell_texts(texts, name)
                        wall, vecs = embed(base, t)
                        samples[engine][name].append(round(len(t) / wall, 4))
                        if run == 0:
                            vectors[engine][name] = vecs
            for engine, (_, _, cmd, env) in servers.items():
                cells_out = {}
                for name in CELLS:
                    v = samples[engine][name]
                    cells_out[name] = {
                        "texts": len(cell_texts(texts, name)),
                        "tokens": cell_tokens[name],
                        "texts_per_s": v,
                        "median": round(statistics.median(v), 4),
                        "min": min(v),
                        "max": max(v),
                        "spread_pct": round(
                            100.0 * (max(v) - min(v)) / statistics.median(v), 2
                        ),
                    }
                shown = " ".join(
                    f"{k}={v}"
                    for k, v in sorted(env.items())
                    if k.startswith("PEREGRINE_") or k == "CUDA_VISIBLE_DEVICES"
                )
                if ENGINES[engine]["kind"] == "llama":
                    binary = LLAMA
                elif ENGINES[engine].get("binary"):
                    binary = PEREGRINE_BEFORE
                else:
                    binary = PEREGRINE
                out["models"][label]["engines"][engine] = {
                    "cmd": " ".join(cmd).replace(HOME, "~"),
                    "env": shown,
                    "binary_version": (
                        (lambda v: (v.stdout + v.stderr).strip().splitlines()[0])(
                            subprocess.run(
                                [str(binary), "--version"],
                                capture_output=True, text=True, env=env,
                            )
                        )
                    ),
                    "cells": cells_out,
                    "vectors": vectors[engine],
                }
                print(f"{label} {engine}: " + ", ".join(
                    f"{name} {cells_out[name]['median']:.3f}" for name in CELLS),
                    flush=True)
        finally:
            for proc, _, _, _ in servers.values():
                proc.terminate()
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
    # Cosine to llama.cpp, per text.
    for label in out["models"]:
        llama = out["models"][label]["engines"].get("llama.cpp", {}).get("vectors")
        if not llama:
            continue
        for engine in out["models"][label]["engines"]:
            vecs = out["models"][label]["engines"][engine]["vectors"]
            if not vecs:
                continue
            cos = {}
            for name in CELLS:
                cs = [cosine(a, b) for a, b in zip(vecs[name], llama[name])]
                cos[name] = {"min": min(cs), "mean": sum(cs) / len(cs), "n": len(cs)}
            out["models"][label]["engines"][engine]["cosine_vs_llama"] = cos
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())
