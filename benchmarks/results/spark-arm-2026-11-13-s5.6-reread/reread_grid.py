#!/usr/bin/env python3
"""The S5.6 re-read: three builds interleaved in every round, one disturbance level per pass.

The S5.6 grid (`../spark-arm-2026-11-13-s5.6/run_grid.py`) measures one binary. This
one measures three in the same window so the arm comparison cannot be a host drift:
for every round, every disturbance level and every cell, the arms run back to back
inside that level, and only then does the next level start. The disturbance generator
(`loadgen`, N busy threads on the ten Cortex-X925 CPUs) runs for one (level, round)
measurement and is stopped by PID in a `finally`; no process outlives its command.

Arms:
  head    `7d46020`, the S5.6 head before S5.4j's join-gap spin window
  merged  `origin/s5.6-pool-contended`, both PRs
  gap0    the merged branch with `GAP_CAP_US = 0` (the stretch off), a local build

  python reread_grid.py --cell prompt4096 --arms head,merged,gap0 \
      --rounds 0,1,2 --out raw/reread-prompt4096-r012.jsonl

Cells are exact mode (the fast prompt mode is off; `PEREGRINE_EMBED_CACHE_MB=0`).
"""

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]
HOME = os.path.expanduser("~")
MODEL = f"{HOME}/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
CORPUS = "/tmp/opencode/corpus-long.txt"
CLUSTER = "5-9,15-19"
IDLE_CPUS = (5, 6, 7, 8, 9, 15, 16, 17, 18, 19)
ARMS = {
    "head": f"{HOME}/wt-s56-head/target/release",
    "merged": f"{HOME}/wt-s56-reread/target/release",
    "gap0": f"{HOME}/wt-s56-gap0/target/release",
    "fixed": f"{HOME}/peregrine/target/release",
}
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


def mask_for(threads):
    if threads == 1:
        return "5"
    if threads == 4:
        return "5-8"
    return CLUSTER


def argv_for(arm, cell, threads):
    b = ARMS[arm]
    if cell == "prompt512":
        argv = [f"{b}/examples/promptbench", MODEL, CORPUS, "512", str(threads)]
    elif cell == "prompt4096":
        argv = [f"{b}/examples/promptbench", MODEL, CORPUS, "4096", str(threads)]
    elif cell == "decode":
        argv = [f"{b}/peregrine", "bench", "-m", MODEL,
                "--prompt-tokens", "16", "--tokens", "128", "--threads", str(threads), "--json"]
    else:
        raise SystemExit(f"cell {cell} is not a single-process cell")
    return ["taskset", "-c", mask_for(threads)] + argv


def one_run(arm, cell, threads, timeout):
    argv = argv_for(arm, cell, threads)
    env = dict(os.environ)
    env.update(BASE_ENV)
    before = idle_snapshot()
    t0 = time.time()
    p = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=timeout)
    wall = time.time() - t0
    idle = idle_share(before, idle_snapshot())
    if p.returncode != 0:
        raise RuntimeError(f"{' '.join(argv)} exited {p.returncode}: {p.stderr[-400:]}")
    d = json.loads(p.stdout)
    v = d["decode_tok_s"] if cell == "decode" else d["prefill_tok_s"]
    return v, wall, idle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True, choices=["prompt512", "prompt4096", "decode"])
    ap.add_argument("--arms", default="head,merged,gap0")
    ap.add_argument("--levels", default="0,1,2,4")
    ap.add_argument("--threads", default="10")
    ap.add_argument("--rounds", default="0,1,2,3,4", help="round indices to run (the driver restarts per command)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--timeout", type=int, default=300)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    arms = [a for a in args.arms.split(",") if a]
    levels = [int(x) for x in args.levels.split(",") if x != ""]
    threads_list = [int(t) for t in args.threads.split(",") if t]

    for r in [int(x) for x in args.rounds.split(",") if x != ""]:
        for level in levels:
            disturber = None
            try:
                if level > 0:
                    disturber = subprocess.Popen(
                        ["taskset", "-c", CLUSTER, f"{ARMS['merged']}/examples/loadgen", str(level)],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    )
                    time.sleep(1.0)
                for threads in threads_list:
                    # Rotate the arm order every round so a within-level drift
                    # (the host's load climbed through the window) cannot favour one arm.
                    order = arms[r % len(arms):] + arms[: r % len(arms)]
                    for arm in order:
                        l1 = load1()
                        try:
                            v, wall, idle = one_run(arm, args.cell, threads, args.timeout)
                        except Exception as e:  # noqa: BLE001 - a failed cell is a record
                            rec = {"arm": arm, "level": level, "threads": threads,
                                   "cell": args.cell, "run": r, "error": str(e)[:400],
                                   "load1": l1}
                        else:
                            rec = {"arm": arm, "level": level, "threads": threads,
                                   "cell": args.cell, "run": r, "value": v,
                                   "wall_s": round(wall, 3), "idle": round(idle, 3),
                                   "load1": l1}
                        with open(out, "a") as f:
                            f.write(json.dumps(rec) + "\n")
                            f.flush()
                        print(f'r{r} {arm:6} l{level} t{threads:>2} {args.cell:11} '
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
