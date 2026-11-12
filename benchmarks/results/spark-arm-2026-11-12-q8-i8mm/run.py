#!/usr/bin/env python3
"""Q8_0 embedding files that stay on the int8 tiles: Peregrine before/after S4.9q vs llama.cpp.

Cells: one text of 16, 128, 512 and 2000 tokens, one batch of 32 texts of 128
tokens, and the mixed-length batch of 32 texts (8..512 tokens, seed 20261111).
Every arm stays up for a model and the rounds interleave them in a rotating
order, so a frequency or thermal drift is not charged to one engine. The table
reports the median of `--runs` runs and the min..max spread. Standard library
only.

  PIN_CPUS=5-9,15-19 python3 run.py --out summary.json --runs 5
"""

import argparse
import json
import math
import os
from pathlib import Path
import random
import statistics
import struct
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
PEREGRINE_AFTER = Path(
    os.environ.get("PEREGRINE_BIN", f"{HOME}/wt-s49q/target/release/peregrine")
)
PEREGRINE_BEFORE = Path(
    os.environ.get("PEREGRINE_BEFORE_BIN", f"{HOME}/wt-s49q-before/target/release/peregrine")
)
HERE = Path(__file__).resolve().parent

# `texts` is the calibration file the earlier embedding run (`spark-2026-11-12-embeddings`)
# left in /var/tmp; a missing one is built here with the same recipe.
MODELS = {
    "EmbeddingGemma-300M-Q8_0": {
        "path": Path(f"{HOME}/models/embed/embeddinggemma/embeddinggemma-Q8_0.gguf"),
        "texts": "/var/tmp/embed-texts/embeddinggemma.json",
        "fixture": "crates/peregrine-models/tests/fixtures/bert/texts.txt",
        "ctx": 2048,
        "pooling": "mean",
        "llama": {"parallel": 32, "ctx": 65792, "batch": 2048, "ubatch": 2048},
    },
    "Qwen3-Embedding-0.6B-Q8_0": {
        "path": Path(f"{HOME}/models/embed/qwen3-embed-0.6b/qwen3-embed-0.6b-q8_0.gguf"),
        "texts": "/var/tmp/embed-texts/qwen3-embed-0.6b.json",
        "fixture": "crates/peregrine-models/tests/fixtures/bert/texts.txt",
        "ctx": 2048,
        "pooling": "last",
        "llama": {"parallel": 32, "ctx": 65792, "batch": 2048, "ubatch": 2048},
    },
    "Nemotron-3.5-Embed-1B-Q8_0": {
        "path": Path(f"{HOME}/models/embed/nemotron/Nemotron-3.5-Embed-1B-Q8_0.gguf"),
        "texts": "/var/tmp/embed-texts/nemotron-3.5-embed-1b.json",
        "fixture": "crates/peregrine-models/tests/fixtures/bert/texts.txt",
        "ctx": 2048,
        "pooling": "mean",
        "llama": {"parallel": 32, "ctx": 65792, "batch": 2048, "ubatch": 2048},
    },
    "bge-m3-Q8_0": {
        "path": Path(f"{HOME}/models/embed/bge-m3-Q8_0.gguf"),
        "texts": "/var/tmp/embed-texts/bge-m3.json",
        "fixture": "crates/peregrine-models/tests/fixtures/bert/texts.txt",
        "ctx": 2048,
        "pooling": "cls",
        "llama": {"parallel": 8, "ctx": 16448, "batch": 2048, "ubatch": 2048},
    },
}
ENGINES = {
    "peregrine-before": {"kind": "peregrine", "isa": "native", "binary": "before"},
    "peregrine-after": {"kind": "peregrine", "isa": "native"},
    "llama.cpp": {"kind": "llama"},
}
THREADS = 10
SLOTS = 32
PIN = os.environ.get("PIN_CPUS", "")
PORT_BASE = 18410
CELLS = ["t16", "t128", "t512", "t2000", "b32x128", "mixed"]
MIXED_SEED = 20261111
MIXED_POOL = [8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512]


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


def wait_ready(base, proc, timeout=300.0):
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


def words_of(fixture):
    lines = [ln.split() for ln in fixture.read_text().splitlines() if ln.strip()]
    return [w for ln in lines for w in ln]


def fit(base, words, start, target):
    """A text of at most `target` tokens (including the special tokens) from a window."""
    span = len(words)
    rotated = words[start % span:] + words[: start % span]
    lo, hi, best = 1, len(rotated), rotated[0]
    while lo <= hi:
        mid = (lo + hi) // 2
        text = " ".join(rotated[:mid])
        if n_tokens(base, text) <= target:
            best = text
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def build_texts(base, fixture):
    """The earlier run's recipe: 32 distinct windows per length, and the seeded mixed set."""
    words = words_of(fixture)
    out = {"tokenizer": "llama-server /tokenize", "per_length": 32, "lengths": {}}
    for target in (16, 128, 512, 2000):
        texts = []
        for i in range(32):
            text = fit(base, words, i * 53 + target * 11, target)
            texts.append({"text": text, "tokens": n_tokens(base, text)})
        out["lengths"][str(target)] = texts
    batch = []
    for i in range(32):
        text = fit(base, words, i * 37 + 128 * 3, 128)
        batch.append({"text": text, "tokens": n_tokens(base, text)})
    out["lengths"]["b32x128"] = batch
    rng = random.Random(MIXED_SEED)
    mixed = []
    for i in range(32):
        target = rng.choice(MIXED_POOL)
        text = fit(base, words, i * 53 + target * 11, target)
        mixed.append({"text": text, "tokens": n_tokens(base, text)})
    out["mixed_seed"] = MIXED_SEED
    out["mixed_lengths"] = sorted({t["tokens"] for t in mixed})
    out["lengths"]["mixed"] = mixed
    return out


def load_texts(base, model, fixture, path):
    p = Path(path)
    if p.exists():
        return json.loads(p.read_text())
    data = build_texts(base, fixture)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data))
    return data


def cell_texts(texts, name):
    if name == "b32x128":
        if "b32x128" in texts["lengths"]:
            return [t["text"] for t in texts["lengths"]["b32x128"]]
        out = []
        for i in range(32):  # 32 texts of 128 tokens from the file's own L128 windows
            out.append(texts["lengths"]["128"][i % len(texts["lengths"]["128"])]["text"])
        return out
    if name == "mixed":
        return [t["text"] for t in texts["lengths"]["mixed"]]
    return [texts["lengths"][name[1:]][0]["text"]]


def start_engine(engine, model, port):
    spec = ENGINES[engine]
    base = f"http://127.0.0.1:{port}"
    tag = engine.replace(".", "_")
    log = open(f"/var/tmp/q8-i8mm-{tag}-{port}.log", "w")
    if spec["kind"] == "llama":
        lc = model["llama"]
        cmd = [
            str(LLAMA), "-m", str(model["path"]), "--embedding",
            "--pooling", model["pooling"], "-t", str(THREADS),
            "-c", str(lc["ctx"]), "-b", str(lc["batch"]), "-ub", str(lc["ubatch"]),
            "-np", str(lc["parallel"]), "-fa", "auto",
            "--host", "127.0.0.1", "--port", str(port),
        ]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    else:
        binary = PEREGRINE_BEFORE if spec.get("binary") == "before" else PEREGRINE_AFTER
        cmd = [
            str(binary), "serve", "-m", str(model["path"]),
            "--host", "127.0.0.1", "--port", str(port), "-t", str(THREADS),
            "-c", str(model["ctx"]), "--slots", str(SLOTS), "--no-profile",
            # The exact cache of finished embedding vectors would turn a repeated
            # run into a lookup; every engine here computes every text every time.
            "--opt", "EMBED_CACHE_MB=0",
        ]
        env = dict(os.environ, PEREGRINE_ISA=spec["isa"], PEREGRINE_PACK_CACHE="0")
    if PIN:
        cmd = ["taskset", "-c", PIN] + cmd
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    if not wait_ready(base, proc):
        log.close()
        proc.kill()
        raise SystemExit(f"{engine} failed to start; see /var/tmp/q8-i8mm-*.log")
    return proc, base, cmd, env


def embed(base, texts):
    t0 = time.perf_counter()
    reply = post(base + "/v1/embeddings", {
        "input": texts, "model": "embed", "encoding_format": "float",
    })
    wall = time.perf_counter() - t0
    rows = sorted(reply["data"], key=lambda d: d.get("index", 0))
    return wall, [r["embedding"] for r in rows]


def bits(vec):
    """The raw float bits, so before/after can be compared exactly."""
    return [struct.unpack("<I", struct.pack("<f", x))[0] for x in vec]


def floats(bit_list):
    return [struct.unpack("<f", struct.pack("<I", x))[0] for x in bit_list]


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb)


def meminfo():
    try:
        info = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            k, v = line.split(":", 1)
            info[k] = v.strip()
        return {"MemAvailable_gb": round(int(info["MemAvailable"].split()[0]) / 1048576, 1)}
    except Exception:
        return {}


def host_state():
    return {
        "loadavg": list(os.getloadavg()),
        **meminfo(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="summary.json")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--engines", default=",".join(ENGINES))
    ap.add_argument("--build-texts-only", action="store_true")
    args = ap.parse_args()
    models = [m for m in args.models.split(",") if m]
    engines = [e for e in args.engines.split(",") if e]
    out = {
        "runs": args.runs,
        "threads": THREADS,
        "pin_cpus": PIN or None,
        "cells": CELLS,
        "host_before": host_state(),
        "models": {},
    }
    port = PORT_BASE
    for label in models:
        model = MODELS[label]
        fixture = (HERE.parents[2] / model["fixture"]).resolve()
        texts_file = Path(model["texts"])
        if args.build_texts_only and texts_file.exists():
            print(f"{label}: {texts_file} exists")
            continue
        # Calibrate the texts once per model with llama-server's tokenizer.
        if not texts_file.exists():
            proc, base, cmd, _ = start_engine("llama.cpp", model, port)
            port += 1
            try:
                texts = load_texts(base, model, fixture, texts_file)
            finally:
                proc.terminate()
                proc.wait()
        else:
            texts = json.loads(texts_file.read_text())
        if args.build_texts_only:
            print(f"{label}: {texts_file}: " + ", ".join(
                f"{k} {len(v)} texts {sum(t['tokens'] for t in v)} tok"
                for k, v in texts["lengths"].items()))
            continue
        out["models"][label] = {
            "path": str(model["path"]).replace(HOME, "~"),
            "bytes": model["path"].stat().st_size,
            "sha256": subprocess.run(
                ["sha256sum", str(model["path"])], capture_output=True, text=True
            ).stdout.split()[0],
            "texts": {k: [t["text"] for t in v] for k, v in texts["lengths"].items()},
            "cell_tokens": {
                name: sum(
                    t["tokens"]
                    for t in texts["lengths"][
                        "b32x128"
                        if name == "b32x128" and "b32x128" in texts["lengths"]
                        else ("128" if name == "b32x128" else ("mixed" if name == "mixed" else name[1:]))
                    ][: (32 if name in ("b32x128", "mixed") else 1)]
                )
                for name in CELLS
            },
            "engines": {},
        }
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
                            vectors[engine][name] = [bits(v) for v in vecs]
            for engine, (_, _, cmd, env) in servers.items():
                cells_out = {}
                for name in CELLS:
                    v = samples[engine][name]
                    cells_out[name] = {
                        "texts": len(cell_texts(texts, name)),
                        "tokens": out["models"][label]["cell_tokens"][name],
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
                elif ENGINES[engine].get("binary") == "before":
                    binary = PEREGRINE_BEFORE
                else:
                    binary = PEREGRINE_AFTER
                try:
                    version = subprocess.run(
                        [str(binary), "--version"], capture_output=True, text=True, env=env
                    )
                    version = (version.stdout + version.stderr).strip().splitlines()[0]
                except Exception:
                    version = ""
                out["models"][label]["engines"][engine] = {
                    "cmd": " ".join(cmd).replace(HOME, "~"),
                    "env": shown,
                    "binary_version": version,
                    "cells": cells_out,
                    "vectors_bits": vectors[engine],
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
    # The before/after vectors must be the same bits, and every arm against llama.cpp.
    for label, m in out["models"].items():
        before = m["engines"].get("peregrine-before", {}).get("vectors_bits")
        after = m["engines"].get("peregrine-after", {}).get("vectors_bits")
        if before and after:
            same = {}
            for name in CELLS:
                a, b = before[name], after[name]
                same[name] = {
                    "identical": a == b,
                    "n": len(a),
                    "elements": sum(len(v) for v in a),
                }
            m["vectors_before_after"] = same
        llama = m["engines"].get("llama.cpp", {}).get("vectors_bits")
        for engine, e in m["engines"].items():
            if engine == "llama.cpp" or not llama:
                continue
            cos = {}
            for name in CELLS:
                cs = [
                    cosine(floats(x), floats(y))
                    for x, y in zip(e["vectors_bits"][name], llama[name])
                ]
                cos[name] = {"min": min(cs), "mean": sum(cs) / len(cs), "n": len(cs)}
            e["cosine_vs_llama"] = cos
    out["host_after"] = host_state()
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())
