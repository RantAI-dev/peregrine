#!/usr/bin/env python3
"""S4.9w Part 2 lever A/B: row block (orig vs no block) medians from cellprof JSONL."""
import glob
import json
import statistics
from pathlib import Path

P = Path(__file__).resolve().parent / "parts" / "rowblock"
CELLS = ["L128_b1", "L128_b8", "L128_b32", "L512_b1", "L512_b8", "L512_b32", "L2000_b1"]
for model in ("nomic", "jina"):
    print(f"**{model} F16**")
    print("| cell | orig texts/s | no block | ratio | orig matmul ms | no block matmul ms |")
    print("|---|---:|---:|---:|---:|---:|")
    for cell in CELLS:
        vals = {}
        for arm in ("orig", "noblock"):
            t, m = [], []
            for f in glob.glob(str(P / f"r*-{arm}-{model}-*.jsonl")):
                for line in open(f):
                    d = json.loads(line)
                    if d["cell"] == cell:
                        t.append(d["texts_per_s"])
                        m.append(d["stats_ms"]["matmul"])
            vals[arm] = (statistics.median(t), statistics.median(m))
        o, n = vals["orig"], vals["noblock"]
        print(f"| {cell} | {o[0]:.3f} | {n[0]:.3f} | {n[0] / o[0]:.3f} | {o[1]:.1f} | {n[1]:.1f} |")
