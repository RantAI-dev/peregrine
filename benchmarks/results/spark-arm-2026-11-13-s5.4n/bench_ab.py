#!/usr/bin/env python3
"""The S5.4n A/B: two Peregrine builds (or the fast/exact modes of one) over the same cells.

Each cell is one fresh `promptbench` run (or `llama-bench` for another engine), pinned to
the thread mask, and every arm runs inside every round so a drifting host affects all of
them alike. The arms rotate per round. Ratios are recomputed from the medians by
`summarize.py`, never typed.

  python bench_ab.py --arms old,new --kind pg --modes fast --lengths 4096 \
      --threads 1,10 --rounds 0,1,2,3,4 --out raw/insitu-4096.jsonl
  python bench_ab.py --kind llama --lengths 512,4096 --threads 1,10 ... --out raw/llama.jsonl

Engines:
  pg        `promptbench`; `--modes exact|fast` sets `PEREGRINE_FAST_PROMPT`
  llama     llama.cpp's `llama-bench`
  ik        ik_llama.cpp's `llama-bench`, `--rtr` adds its runtime repack flag
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
HOME = os.path.expanduser("~")
MODELS = {
    "llama1b": f"{HOME}/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
    "qwen05b": f"{HOME}/models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf",
}
CORPUS = "/tmp/opencode/corpus-long.txt"
LLAMA = f"{HOME}/.cache/peregrine-bench/llama.cpp-v0.9.0/build/bin"
IK = f"{HOME}/.cache/peregrine-bench/ik_llama.cpp/build/bin"
PGBINS = {
    "old": f"{HOME}/wt-s54n-off/target/release/examples/promptbench",
    "new": f"{HOME}/peregrine/target/release/examples/promptbench",
}
CLUSTER = "5-9,15-19"


def mask_for(threads):
    if threads == 1:
        return "5"
    if threads == 2:
        return "5,6"
    if threads == 4:
        return "5-8"
    return CLUSTER


def load1():
    return float(open("/proc/loadavg").read().split()[0])


def argv_for(args, arm, model, length, threads):
    if args.kind == "pg":
        argv = [PGBINS[arm], MODELS[model], CORPUS, str(length), str(threads)]
        env = {"PEREGRINE_EMBED_CACHE_MB": "0"}
        if args.modes == "fast":
            env["PEREGRINE_FAST_PROMPT"] = "1"
        return argv, env
    b = LLAMA if args.kind == "llama" else IK
    argv = [f"{b}/llama-bench", "-m", MODELS[model], "-p", str(length), "-n", "0",
            "-t", str(threads), "-r", "1", "-o", "json"]
    if args.rtr:
        argv.append("-rtr")
    return argv, {}


def one(args, arm, model, length, threads, timeout):
    argv, extra = argv_for(args, arm, model, length, threads)
    env = dict(os.environ)
    env.update(extra)
    t0 = time.time()
    p = subprocess.run(["taskset", "-c", mask_for(threads)] + argv, env=env,
                       capture_output=True, text=True, timeout=timeout)
    wall = time.time() - t0
    if p.returncode != 0:
        raise RuntimeError(f"{' '.join(argv)} exited {p.returncode}: {p.stderr[-300:]}")
    if args.kind == "pg":
        return json.loads(p.stdout)["prefill_tok_s"], wall
    d = json.loads(p.stdout)
    rec = d[0] if isinstance(d, list) else d
    return rec["avg_ts"], wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="pg", choices=["pg", "llama", "ik"])
    ap.add_argument("--arms", default="old,new")
    ap.add_argument("--modes", default="fast", choices=["fast", "exact"])
    ap.add_argument("--models", default="llama1b")
    ap.add_argument("--lengths", default="4096")
    ap.add_argument("--threads", default="1,10")
    ap.add_argument("--rounds", default="0,1,2,3,4")
    ap.add_argument("--rtr", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    arms = [a for a in args.arms.split(",") if a]
    models = [m for m in args.models.split(",") if m]
    lengths = [int(x) for x in args.lengths.split(",") if x]
    threads_list = [int(x) for x in args.threads.split(",") if x]
    tag = f"{args.kind}/{args.modes}" + ("/rtr" if args.rtr else "")
    for r in [int(x) for x in args.rounds.split(",") if x != ""]:
        order = arms[r % len(arms):] + arms[: r % len(arms)]
        for m in models:
            for length in lengths:
                for threads in threads_list:
                    for arm in order:
                        l1 = load1()
                        try:
                            v, wall = one(args, arm, m, length, threads, args.timeout)
                        except Exception as e:  # noqa: BLE001 - a failed cell is a record
                            rec = {"kind": tag, "arm": arm, "model": m, "length": length,
                                   "threads": threads, "run": r, "error": str(e)[:300],
                                   "load1": l1}
                        else:
                            rec = {"kind": tag, "arm": arm, "model": m, "length": length,
                                   "threads": threads, "run": r, "value": v,
                                   "wall_s": round(wall, 3), "load1": l1}
                        with open(out, "a") as f:
                            f.write(json.dumps(rec) + "\n")
                            f.flush()
                        print(f'r{r} {tag:10} {arm:4} {m:8} {length:5} t{threads:>2} '
                              f'{rec.get("value", 0):9.2f} load {l1:5.2f} {rec.get("error", "")}',
                              flush=True)


if __name__ == "__main__":
    main()
