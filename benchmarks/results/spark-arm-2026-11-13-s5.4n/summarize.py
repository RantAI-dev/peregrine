#!/usr/bin/env python3
"""S5.4n's tables: the in-situ A/B, the B5 grid, the attention bucket and the kernel bench.

Every ratio is recomputed from the medians here. `new/old` is the fast attention order
on / off (the same branch, the order's tier flag flipped in the `old` build), so a
number above 1.0 is the order's gain. `fast/exact` and `pg/<engine>` compare across
columns of the same table.

  python summarize.py > table.txt
"""

import glob
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(name):
    rows = []
    for f in glob.glob(str(HERE / "raw" / name)):
        for line in open(f):
            line = line.strip()
            if line.startswith("{"):
                rows.append(json.loads(line))
    return rows


def med(vals):
    return st.median(vals) if vals else float("nan")


def spread(vals):
    return (max(vals) - min(vals)) / 2 / st.median(vals) * 100 if vals else float("nan")


def grid(rows, title):
    g = defaultdict(list)
    for r in rows:
        if "value" in r:
            g[(r["model"], r["length"], r["threads"], r["arm"])].append(r["value"])
    print(f"\n== {title}")
    print("| model | length | threads | old | new | new/old |")
    print("|---|---|---|---|---|---|")
    for m in ["llama1b", "qwen05b"]:
        for length in [512, 4096]:
            for t in [1, 10]:
                o = g.get((m, length, t, "old"), [])
                n = g.get((m, length, t, "new"), [])
                if not o or not n:
                    continue
                print(f"| {m} | {length} | {t} | {med(o):.1f} (±{spread(o):.1f} %) | "
                      f"{med(n):.1f} (±{spread(n):.1f} %) | **{med(n) / med(o):.3f}** |")


def buckets(rows):
    g = defaultdict(list)
    for r in rows:
        g[(r["length"], r["threads"], r["arm"])].append((r["attn_ms"], r["total_ms"], r["tok_s"]))
    print("\n== attention bucket (fast mode, llama1b, medians of three)")
    print("| length | threads | arm | attention ms | total ms | attention share | tok/s |")
    print("|---|---|---|---|---|---|---|")
    for length in [512, 4096]:
        for t in [1, 10]:
            for arm in ["old", "new"]:
                v = g.get((length, t, arm), [])
                if not v:
                    continue
                a = med([x[0] for x in v])
                tot = med([x[1] for x in v])
                tok = med([x[2] for x in v])
                print(f"| {length} | {t} | {arm} | {a:.0f} | {tot:.0f} | "
                      f"{a / tot * 100:.1f} % | {tok:.1f} |")


def engines(pg_rows, eng_rows, eng_name):
    """Fast mode against another engine: `pg / engine` (above 1.0 is Peregrine ahead)."""
    g = defaultdict(list)
    for r in pg_rows:
        if "value" in r:
            g[(r["model"], r["length"], r["threads"], r["arm"])].append(r["value"])
    e = defaultdict(list)
    for r in eng_rows:
        if "value" in r:
            e[(r["model"], r["length"], r["threads"])].append(r["value"])
    print(f"\n== fast / {eng_name}")
    for m in ["llama1b", "qwen05b"]:
        for length in [512, 4096]:
            for t in [1, 10]:
                o = g.get((m, length, t, "old"), [])
                n = g.get((m, length, t, "new"), [])
                x = e.get((m, length, t), [])
                if not o or not n or not x:
                    continue
                print(f"| {m} | {length} | {t} | {med(o):.1f} | {med(n):.1f} | {med(x):.1f} | "
                      f"{med(o) / med(x):.3f} | {med(n) / med(x):.3f} |")


def main():
    insitu = load("insitu-*.jsonl")
    print("== in situ, 4 096-token fast prompt (stop-rule cell), five rounds")
    for t in [1, 10]:
        o = [r["value"] for r in insitu if r["length"] == 4096 and r["threads"] == t and r["arm"] == "old"]
        n = [r["value"] for r in insitu if r["length"] == 4096 and r["threads"] == t and r["arm"] == "new"]
        if o and n:
            print(f"t{t}: old {med(o):.1f} new {med(n):.1f} new/old {med(n) / med(o):.4f}")
    grid(load("b5-fast.jsonl"), "B5, fast mode (new/old = the order's gain)")
    grid(load("b5-exact.jsonl"), "B5, exact mode (must be level: the order is off)")
    fast = load("b5-fast.jsonl")
    for name, pattern in [("llama.cpp", "b5-llama.jsonl"), ("ik_llama.cpp", "b5-ik.jsonl"),
                          ("ik_llama.cpp -rtr", "b5-ik-rtr.jsonl")]:
        engines(fast, load(pattern), name)
    buckets(load("buckets.jsonl"))


if __name__ == "__main__":
    main()
