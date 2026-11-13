#!/usr/bin/env python3
"""The S5.6 embedding cell: 32 different 128-token texts, one request each, in a wave.

Starts one server per (engine, thread count), then for every interleaved round and
disturbance level sends the 32 requests concurrently and times the wave. The exact
result cache is off on Peregrine (`--opt EMBED_CACHE_MB=0`); llama.cpp has no such
cache. The disturbance generator is started for one measurement and stopped by PID
in a `finally`, like the decode/prefill grid.

  python embed_wave.py --engine pg --threads 10 --levels 0,4 --runs 3 --out raw/pg-embed.jsonl
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HOME = os.path.expanduser("~")
MODEL = f"{HOME}/models/gte-modernbert-base-F16.gguf"
TEXTS = "/tmp/opencode/texts-128x32.json"
LLAMA = f"{HOME}/.cache/peregrine-bench/llama.cpp-v0.9.0/build/bin"
IK = f"{HOME}/.cache/peregrine-bench/ik_llama.cpp/build/bin"
CLUSTER = "5-9,15-19"
IDLE_CPUS = (5, 6, 7, 8, 9, 15, 16, 17, 18, 19)


def idle_snapshot():
    out = {}
    for line in open("/proc/stat"):
        parts = line.split()
        if parts and parts[0].startswith("cpu") and parts[0] != "cpu":
            cpu = int(parts[0][3:])
            if cpu in IDLE_CPUS:
                vals = [int(v) for v in parts[1:]]
                out[cpu] = (vals[3] + (vals[4] if len(vals) > 4 else 0), sum(vals[:8]))
    return out


def idle_share(a, b):
    vals = []
    for c in a:
        di, dt = b[c][0] - a[c][0], b[c][1] - a[c][1]
        if dt > 0:
            vals.append(di / dt)
    return sum(vals) / len(vals) if vals else 0.0


def load1():
    return float(open("/proc/loadavg").read().split()[0])


def texts():
    d = json.load(open(TEXTS))
    return [t["text"] for t in d["128"]]


def port_for(engine, threads):
    return {"pg": 18310, "llama": 18320, "ik": 18330}[engine] + threads


def start(engine, threads, peregrine_bin):
    port = port_for(engine, threads)
    env = dict(os.environ)
    env["PEREGRINE_EMBED_CACHE_MB"] = "0"
    if engine == "pg":
        argv = [peregrine_bin, "serve", "-m", MODEL,
                "--threads", str(threads), "--port", str(port), "--slots", "32",
                "--opt", "EMBED_CACHE_MB=0", "--opt", "PACK_CACHE=0", "--no-profile"]
    elif engine == "llama":
        argv = [f"{LLAMA}/llama-server", "-m", MODEL, "--embeddings", "--pooling", "cls",
                "-t", str(threads), "-np", "32", "-c", "8192", "--port", str(port),
                "--host", "127.0.0.1"]
    else:
        argv = [f"{IK}/llama-server", "-m", MODEL, "--embeddings", "--pooling", "cls",
                "-t", str(threads), "-np", "32", "-c", "8192", "--port", str(port),
                "--host", "127.0.0.1"]
    p = subprocess.Popen(["taskset", "-c", CLUSTER] + argv, env=env,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base = f"http://127.0.0.1:{port}"
    deadline = time.time() + 180
    while time.time() < deadline:
        if p.poll() is not None:
            raise RuntimeError(f"{engine} server exited {p.returncode}")
        try:
            with urllib.request.urlopen(base + "/health", timeout=1) as r:
                if r.status == 200:
                    return p, base
        except Exception:
            pass
        time.sleep(0.5)
    raise RuntimeError(f"{engine} server did not become ready")


def wave(base, body_texts, timeout):
    out = [None] * len(body_texts)
    err = []

    def one(i):
        req = urllib.request.Request(
            base + "/v1/embeddings",
            data=json.dumps({"input": body_texts[i], "model": "x"}).encode(),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                d = json.loads(r.read().decode())
                out[i] = len(d["data"][0]["embedding"])
        except Exception as e:  # noqa: BLE001
            err.append(str(e)[:200])

    ts = [threading.Thread(target=one, args=(i,)) for i in range(len(body_texts))]
    t0 = time.time()
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return time.time() - t0, out, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, choices=["pg", "llama", "ik"])
    ap.add_argument("--threads", default="10")
    ap.add_argument("--levels", default="0,1,2,4")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--peregrine-bin", default=f"{ROOT}/target/release/peregrine")
    args = ap.parse_args()
    body = texts()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for threads in [int(t) for t in args.threads.split(",") if t]:
        server, base = start(args.engine, threads, args.peregrine_bin)
        # warm-up wave: pages the weights in, not measured
        wave(base, body, args.timeout)
        try:
            for r in range(args.runs):
                for level in [int(x) for x in args.levels.split(",") if x != ""]:
                    dist = None
                    try:
                        if level > 0:
                            dist = subprocess.Popen(
                                ["taskset", "-c", CLUSTER,
                                 f"{ROOT}/target/release/examples/loadgen", str(level)],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                            time.sleep(1.0)
                        l1 = load1()
                        before = idle_snapshot()
                        wall, dims, err = wave(base, body, args.timeout)
                        idle = idle_share(before, idle_snapshot())
                        rec = {"engine": args.engine, "level": level, "threads": threads,
                               "cell": "embed32x128", "run": r,
                               "value": round(len(body) / wall, 3) if not err else 0.0,
                               "wall_s": round(wall, 3), "idle": round(idle, 3),
                               "load1": l1, "errors": err[:3]}
                    finally:
                        if dist is not None:
                            dist.send_signal(signal.SIGTERM)
                            try:
                                dist.wait(timeout=5)
                            except subprocess.TimeoutExpired:
                                dist.kill()
                                dist.wait(timeout=5)
                    with open(out, "a") as f:
                        f.write(json.dumps(rec) + "\n")
                        f.flush()
                    print(f'r{r} {rec["engine"]:5} t{threads:>2} l{level} embed '
                          f'{rec["value"]:8.2f} texts/s load {rec["load1"]:5.2f} '
                          f'idle {rec["idle"]:.2f} {rec["errors"][:1]}', flush=True)
        finally:
            server.send_signal(signal.SIGTERM)
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=5)


if __name__ == "__main__":
    main()
