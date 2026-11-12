#!/usr/bin/env python3
"""Rerank throughput with the result cache off, both engines in one process group.

One query and 8 or 32 documents of about 128 tokens, `POST /v1/rerank`;
`pairs/s` is documents divided by the request's wall time, median of five
interleaved rounds.  Both servers stay up for the whole run, and the rounds
alternate their order.  Standard library only.

  python rerank.py --out rerank.json
"""

import argparse
import json
import os
import re
import statistics
import subprocess
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
MODEL = Path(f"{HOME}/models/bge-reranker-v2-m3-Q8_0.gguf")
THREADS = 10
QUERY = "Which of these passages explains how the new index was scheduled to be built?"
PORT = 18600


def post(url, payload, timeout=1800.0):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def alive(url, timeout=1.0):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def wait_ready(base, proc, timeout=240.0):
    end = time.time() + timeout
    while time.time() < end:
        if proc.poll() is not None:
            return False
        if alive(base + "/health") or alive(base + "/v1/models"):
            return True
        time.sleep(0.2)
    return False


def n_tokens(base, text):
    return len(post(base + "/tokenize", {"content": text, "add_special": True})["tokens"])


def window(base, words, start, target):
    lo, hi, best = 1, min(len(words) - start, target * 3 + 16), 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if n_tokens(base, " ".join(words[start:start + mid])) <= target:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    text = " ".join(words[start:start + best])
    # Character trim to land on the target.
    lo, hi, chosen = 1, len(text), text
    while lo <= hi:
        mid = (lo + hi) // 2
        chunk = text[:mid].rstrip()
        if n_tokens(base, chunk) <= target:
            chosen = chunk
            lo = mid + 1
        else:
            hi = mid - 1
    return chosen


def start(engine, port):
    base = f"http://127.0.0.1:{port}"
    log = open(f"/var/tmp/embcache-rerank-{engine.replace('.', '_')}-{port}.log", "w")
    if engine == "llama.cpp":
        cmd = [str(LLAMA), "-m", str(MODEL), "--reranking", "-t", str(THREADS),
               "-c", "8192", "-b", "2048", "-ub", "2048", "-np", "1", "-fa", "auto",
               "--host", "127.0.0.1", "--port", str(port)]
    else:
        cmd = [str(PEREGRINE), "serve", "-m", str(MODEL), "--host", "127.0.0.1",
               "--port", str(port), "-t", str(THREADS), "-c", "8192", "--slots", "1",
               "--no-profile", "--opt", "EMBED_CACHE_MB=0"]
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    if not wait_ready(base, proc):
        log.close()
        proc.kill()
        raise SystemExit(f"{engine} failed to start; see /var/tmp/embcache-rerank-*.log")
    return proc, base, cmd


def rerank(base, docs):
    t0 = time.perf_counter()
    reply = post(base + "/v1/rerank", {"query": QUERY, "documents": docs,
                                       "top_n": len(docs)})
    wall = time.perf_counter() - t0
    if len(reply.get("results", [])) != len(docs):
        raise SystemExit(f"wrong number of results: {reply}")
    return wall, reply.get("usage", {}).get("prompt_tokens", 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="rerank.json")
    ap.add_argument("--runs", type=int, default=5)
    args = ap.parse_args()
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'’\-]*|[,.;:!?]",
                       CORPUS.read_text(encoding="utf-8"))
    words = (words * 200)[:200000]
    servers = {}
    out = {"model": str(MODEL), "query": QUERY, "cells": {}, "engines": {}}
    try:
        for i, engine in enumerate(("peregrine", "llama.cpp")):
            proc, base, cmd = start(engine, PORT + i)
            servers[engine] = (proc, base, cmd)
        # Calibrate the documents with llama-server's tokenizer.
        base = servers["llama.cpp"][1]
        docs = [window(base, words, 300 + i * 431, 128) for i in range(32)]
        counts = [n_tokens(base, d) for d in docs]
        out["document_tokens"] = {"min": min(counts), "max": max(counts),
                                  "mean": round(sum(counts) / len(counts), 2)}
        for docs_n in (8, 32):
            payload = docs[:docs_n]
            samples = {e: [] for e in servers}
            tokens = {}
            for engine, (_, b, _) in servers.items():
                _, tokens[engine] = rerank(b, payload)  # warm-up
            order = ["peregrine", "llama.cpp"]
            for run in range(args.runs):
                for engine in order[run % 2:] + order[: run % 2]:
                    wall, pt = rerank(servers[engine][1], payload)
                    samples[engine].append(round(docs_n / wall, 3))
                    tokens[engine] = pt
            out["cells"][str(docs_n)] = {}
            for engine in samples:
                v = samples[engine]
                out["cells"][str(docs_n)][engine] = {
                    "pairs_per_s": v,
                    "median": round(statistics.median(v), 3),
                    "min": min(v), "max": max(v),
                    "prompt_tokens": tokens[engine],
                    "pairs_per_s_spread_pct": round(
                        100 * (max(v) - min(v)) / statistics.median(v), 2),
                }
            p = statistics.median(samples["peregrine"])
            l = statistics.median(samples["llama.cpp"])
            out["cells"][str(docs_n)]["ratio"] = round(p / l, 3)
            print(f"docs {docs_n}: peregrine {p:.2f} llama {l:.2f} ratio {p/l:.2f}", flush=True)
        for engine, (_, _, cmd) in servers.items():
            out["engines"][engine] = {
                "cmd": " ".join(cmd).replace(HOME, "~"),
                "version": subprocess.run(
                    [str(Path(cmd[0])) if engine == "llama.cpp" else str(PEREGRINE),
                     "--version"], capture_output=True, text=True
                ).stdout.strip().splitlines()[:1],
            }
    finally:
        for proc, _, _ in servers.values():
            proc.terminate()
            proc.wait()
    Path(args.out).write_text(json.dumps(out, indent=1).replace(HOME, "~"), encoding="utf-8")


if __name__ == "__main__":
    main()
