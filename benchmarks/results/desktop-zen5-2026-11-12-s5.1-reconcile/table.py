#!/usr/bin/env python3
"""Turn the raw JSONL into the tables of docs/design/s5-prefill-avx512.md §10.

Every rate is converted with the model's own multiply-accumulate count per prompt token,
read from the file's tensor shapes by `macs.py` (0.9731 GMAC/token for Llama 3.2 1B,
0.3579 for Qwen 2.5 0.5B). Ratios are Peregrine divided by the fork.

    python3 table.py .
"""
import collections
import glob
import json
import os
import statistics
import sys

MAC = {"llama1b": 0.9731, "qwen05": 0.3579}
NAME = {"llama1b": "Llama 3.2 1B", "qwen05": "Qwen 2.5 0.5B"}


def load(res, engine, threads, model, run):
    stem = "fork" if engine == "ik" else "peregrine"
    pat = os.path.join(res, f"{stem}-{threads}t-*.jsonl")
    out = []
    for f in glob.glob(pat):
        if ("run2" in f) != (run == 2):
            continue
        for line in open(f):
            r = json.loads(line)
            key = "llama1b" if r["model"].startswith("Llama") else "qwen05"
            if key == model:
                out.append(r)
    return out


def med(rows, engine, width):
    vals = []
    for r in rows:
        w = r.get("ub") if engine == "ik" else r.get("chunk")
        v = r.get("median_ts") if engine == "ik" else r.get("prefill_tok_s")
        if w == width and v:
            vals.append(v)
    return (statistics.median(vals) if vals else None), len(vals)


def main(res):
    for threads in (1, 12):
        for model in ("llama1b", "qwen05"):
            print(f"\n**{NAME[model]}, {threads} thread{'s' if threads > 1 else ''}** — "
                  f"pk is the pass width (the fork's `-ub`, Peregrine's chunk)\n")
            print("| pk | fork tok/s | run 2 | fork GMAC/s | P tok/s | run 2 | P GMAC/s | P ÷ fork |")
            print("|---|---|---|---|---|---|---|---|")
            for w in (8, 16, 32, 64, 128, 256, 512):
                f1, n1 = med(load(res, "ik", threads, model, 1), "ik", w)
                f2, _ = med(load(res, "ik", threads, model, 2), "ik", w)
                p1, _ = med(load(res, "peregrine", threads, model, 1), "peregrine", w)
                p2, _ = med(load(res, "peregrine", threads, model, 2), "peregrine", w)
                f = f1 if f1 is not None else f2
                p = p1 if p1 is not None else p2
                fmt = lambda v: "—" if v is None else f"{v:.1f}"
                g = lambda v: "—" if v is None else f"{v * MAC[model]:.1f}"
                ratio = "—" if (f is None or p is None) else f"{p / f:.2f}"
                print(f"| {w} | {fmt(f1)} | {fmt(f2)} | {g(f)} | {fmt(p1)} | {fmt(p2)} | "
                      f"{g(p)} | {ratio} |")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
