#!/usr/bin/env python3
"""The S5.4n attention bucket: two builds, interleaved, `PEREGRINE_PROF2=full`.

Each cell is one `promptbench` fast-prompt run; the driver keeps `StepStats`' attention
and total milliseconds beside the pass rate. The arms rotate per round.

  python profile_ab.py --models llama1b --lengths 4096 --threads 1,10 --rounds 0,1,2 \
      --out raw/buckets.jsonl
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

HOME = os.path.expanduser("~")
MODELS = {
    "llama1b": f"{HOME}/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
    "qwen05b": f"{HOME}/models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf",
}
CORPUS = "/tmp/opencode/corpus-long.txt"
BINS = {
    "old": f"{HOME}/wt-s54n-off/target/release/examples/promptbench",
    "new": f"{HOME}/peregrine/target/release/examples/promptbench",
}


def mask_for(threads):
    return "5" if threads == 1 else "5-9,15-19"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="old,new")
    ap.add_argument("--models", default="llama1b")
    ap.add_argument("--lengths", default="4096")
    ap.add_argument("--threads", default="1,10")
    ap.add_argument("--rounds", default="0,1,2")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    arms = [a for a in args.arms.split(",") if a]
    for r in [int(x) for x in args.rounds.split(",") if x != ""]:
        order = arms[r % len(arms):] + arms[: r % len(arms)]
        for m in [x for x in args.models.split(",") if x]:
            for length in [int(x) for x in args.lengths.split(",") if x]:
                for t in [int(x) for x in args.threads.split(",") if x]:
                    for arm in order:
                        env = dict(os.environ)
                        env["PEREGRINE_FAST_PROMPT"] = "1"
                        env["PEREGRINE_PROF2"] = "full"
                        env["PEREGRINE_EMBED_CACHE_MB"] = "0"
                        t0 = time.time()
                        p = subprocess.run(
                            ["taskset", "-c", mask_for(t), BINS[arm], MODELS[m],
                             CORPUS, str(length), str(t)],
                            env=env, capture_output=True, text=True, timeout=900)
                        if p.returncode != 0:
                            raise SystemExit(f"{arm} {m} {length}/{t}: {p.stderr[-300:]}")
                        d = json.loads(p.stdout)
                        rec = {"arm": arm, "model": m, "length": length, "threads": t,
                               "run": r, "tok_s": d["prefill_tok_s"],
                               "attn_ms": d["stats_ms"]["attention"],
                               "matmul_ms": d["stats_ms"]["matmul"],
                               "total_ms": d["stats_ms"]["total"],
                               "wall_s": round(time.time() - t0, 3),
                               "load1": float(open("/proc/loadavg").read().split()[0])}
                        with open(out, "a") as f:
                            f.write(json.dumps(rec) + "\n")
                            f.flush()
                        print(f'r{r} {arm:4} {m:8} {length:5} t{t:>2} '
                              f'{rec["tok_s"]:9.2f} attn {rec["attn_ms"]:9.1f} '
                              f'total {rec["total_ms"]:9.1f} load {rec["load1"]:5.2f}',
                              flush=True)


if __name__ == "__main__":
    main()
