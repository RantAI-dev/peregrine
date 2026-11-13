#!/usr/bin/env python3
"""S5.6's contended-host grid: one executable per cell, interleaved disturbance levels.

Every round measures every requested disturbance level, in the same order, so a
co-tenant that arrives or leaves affects each level of that round alike (the
interleaved-rounds rule the earlier rounds used). The disturbance generator (N
busy threads pinned to the Cortex-X925 cluster, N = 0, 1, 2, 4) runs for exactly
one (level, round, thread count, cell) measurement and is stopped by PID in a
`finally` block, whether the measurement finishes or not. Every record carries
the 1-minute load average and the per-core idle share of CPUs 5-9,15-19 over
that run's own window.

  python run_grid.py --engine pg --levels 0,1,2,4 --threads 4,8,10 \
      --cells prompt4096 --runs 5 --out raw/before-pg-4096.jsonl
  python run_grid.py --engine llama --levels 0,4 --threads 10 \
      --cells prompt512 --pin no --runs 5 --out raw/llama-unpinned.jsonl

Cells (exact mode; the pages and embedding cache are off):
  prompt512 / prompt4096   Llama 3.2 1B Q4_K_M prompt tok/s
  decode                   decode tok/s after a 16-token prompt, 128 tokens
  embed                    gte-modernbert-base F16, 32 texts of 128 tokens
                           (server wave; see embed_wave.py, not this driver)

The output is JSONL, appended and flushed per record: a command that overruns its
timeout keeps everything it measured.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HOME = os.path.expanduser("~")
MODEL = f"{HOME}/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
CORPUS = "/tmp/opencode/corpus-long.txt"
LLAMA = f"{HOME}/.cache/peregrine-bench/llama.cpp-v0.9.0/build/bin"
IK = f"{HOME}/.cache/peregrine-bench/ik_llama.cpp/build/bin"
CLUSTER = "5-9,15-19"
IDLE_CPUS = (5, 6, 7, 8, 9, 15, 16, 17, 18, 19)

BASE_ENV = {"PEREGRINE_EMBED_CACHE_MB": "0"}


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


def cpus(threads, pin):
    if pin == "no":
        return None
    if threads == 1:
        return "5"
    if threads == 2:
        return "5,6"
    if threads == 4:
        return "5-8"
    if threads == 6:
        return "5-9,15"
    if threads == 8:
        return "5-9,15-17"
    return CLUSTER


def cmd_for(engine, cell, threads, pin):
    mask = cpus(threads, pin)
    env = dict(os.environ)
    env.update(BASE_ENV)
    if engine == "pg":
        if cell == "prompt512":
            argv = [f"{ROOT}/target/release/examples/promptbench", MODEL, CORPUS, "512", str(threads)]
        elif cell == "prompt4096":
            argv = [f"{ROOT}/target/release/examples/promptbench", MODEL, CORPUS, "4096", str(threads)]
        elif cell == "decode":
            argv = [
                f"{ROOT}/target/release/peregrine", "bench", "-m", MODEL,
                "--prompt-tokens", "16", "--tokens", "128", "--threads", str(threads), "--json",
            ]
        else:
            raise SystemExit(f"cell {cell} is not a single-process cell")
    elif engine in ("llama", "ik"):
        b = LLAMA if engine == "llama" else IK
        p = "512" if cell == "prompt512" else "4096"
        if cell in ("prompt512", "prompt4096"):
            argv = [f"{b}/llama-bench", "-m", MODEL, "-p", p, "-n", "0",
                    "-t", str(threads), "-r", "1", "-o", "json"]
        elif cell == "decode":
            argv = [f"{b}/llama-bench", "-m", MODEL, "-p", "16", "-n", "128",
                    "-t", str(threads), "-r", "1", "-o", "json"]
        else:
            raise SystemExit(f"cell {cell} is not a single-process cell")
    else:
        raise SystemExit(f"unknown engine {engine}")
    if mask:
        argv = ["taskset", "-c", mask] + argv
    return argv, env


def parse_single(engine, cell, text):
    if engine == "pg":
        d = json.loads(text)
        return d["decode_tok_s"] if cell == "decode" else d["prefill_tok_s"]
    d = json.loads(text)
    rec = d[0] if isinstance(d, list) else d
    return rec["avg_ts"]


def one_run(engine, cell, threads, pin, timeout):
    argv, env = cmd_for(engine, cell, threads, pin)
    before = idle_snapshot()
    t0 = time.time()
    p = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=timeout)
    wall = time.time() - t0
    after = idle_snapshot()
    if p.returncode != 0:
        raise RuntimeError(f"{' '.join(argv)} exited {p.returncode}: {p.stderr[-400:]}")
    return parse_single(engine, cell, p.stdout), wall, idle_share(before, after)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, choices=["pg", "llama", "ik"])
    ap.add_argument("--levels", default="0,1,2,4")
    ap.add_argument("--threads", default="4,8,10")
    ap.add_argument("--cells", default="prompt512,prompt4096,decode")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--pin", default="yes", choices=["yes", "no"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    threads_list = [int(t) for t in args.threads.split(",") if t]
    levels = [int(x) for x in args.levels.split(",") if x != ""]
    cells = [c for c in args.cells.split(",") if c]

    for r in range(args.runs):
        for level in levels:
            disturber = None
            try:
                if level > 0:
                    disturber = subprocess.Popen(
                        ["taskset", "-c", CLUSTER, f"{ROOT}/target/release/examples/loadgen", str(level)],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    )
                    time.sleep(1.0)
                for threads in threads_list:
                    for cell in cells:
                        l1 = load1()
                        try:
                            v, wall, idle = one_run(args.engine, cell, threads, args.pin, args.timeout)
                        except Exception as e:  # noqa: BLE001 - a failed cell is a record
                            rec = {"engine": args.engine, "level": level, "threads": threads,
                                   "cell": cell, "run": r, "error": str(e)[:400], "load1": l1,
                                   "tag": args.tag}
                        else:
                            rec = {"engine": args.engine, "level": level, "threads": threads,
                                   "cell": cell, "pin": args.pin, "run": r, "value": v,
                                   "wall_s": round(wall, 3), "idle": round(idle, 3),
                                   "load1": l1, "tag": args.tag}
                        with open(out, "a") as f:
                            f.write(json.dumps(rec) + "\n")
                            f.flush()
                        print(f'r{r} {rec["engine"]:5} l{level} t{threads:>2} {cell:11} '
                              f'{rec.get("value", 0):9.2f}  load {l1:5.2f} '
                              f'idle {rec.get("idle", 0):.2f} {rec.get("error", "")}', flush=True)
            finally:
                if disturber is not None:
                    disturber.send_signal(signal.SIGTERM)
                    try:
                        disturber.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        disturber.kill()
                        disturber.wait(timeout=5)


if __name__ == "__main__":
    main()
