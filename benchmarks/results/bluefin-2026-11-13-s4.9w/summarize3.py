#!/usr/bin/env python3
"""S4.9w Part 3: medians from the harness logs, best equal-accuracy engine, ratios."""
import glob
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
P = ROOT / "parts" / "part3"
ROUNDS = (1, 2, 3)
LINE = re.compile(r"^\s+(\S+)\s+([\d.]+) texts/s")
CELLS = ["L16_b8_c1", "L16_b32_c1", "L128_b1_c1", "L128_b8_c1", "L128_b32_c1",
         "L512_b8_c1", "L512_b32_c1", "mixed_b32_c1"]
BEFORE_RATIO = {
    ("nomic", "f16"): {"L16_b8_c1": 1.019, "L16_b32_c1": 1.088, "L128_b1_c1": 0.929,
                       "L128_b8_c1": 1.002, "L128_b32_c1": 0.929, "L512_b8_c1": 1.090,
                       "L512_b32_c1": 1.102, "mixed_b32_c1": 1.717},
    ("jina", "f16"): {"L16_b8_c1": 0.938, "L16_b32_c1": 0.964, "L128_b1_c1": 0.911,
                      "L128_b8_c1": 0.917, "L128_b32_c1": 0.910, "L512_b8_c1": 1.002,
                      "L512_b32_c1": 1.042, "mixed_b32_c1": 1.692},
    ("jina", "q8_0"): {"L16_b8_c1": 0.945, "L16_b32_c1": 0.974, "L128_b1_c1": 0.916,
                       "L128_b8_c1": 0.903, "L128_b32_c1": 0.918},
}


def medians(engine, model, prec):
    vals = {}
    for r in ROUNDS:
        for f in glob.glob(str(P / f"r{r}-*.log")):
            if f"-{model}-{engine}-{prec}.log" not in f:
                continue
            for line in open(f):
                m = LINE.match(line)
                if m:
                    vals.setdefault(m.group(1), {})[r] = float(m.group(2))
    return {c: statistics.median([v[r] for r in ROUNDS if r in v])
            for c, v in vals.items() if len(v) == len(ROUNDS)}


rows = []
for m in ("nomic", "jina"):
    for p in ("f16", "q8_0"):
        pg = medians("peregrine", m, p)
        others = {e: medians(e, m, "fp32") for e in ("onnxruntime", "fastembed", "openvino")}
        for c in CELLS:
            if c not in pg:
                continue
            best = max((o[c] for o in others.values() if c in o), default=0.0)
            if best <= 0:
                continue
            rows.append((m, p, c, pg[c], best, pg[c] / best,
                         BEFORE_RATIO.get((m, p), {}).get(c)))

print("| model | file | cell | before | now | peregrine | best other | after/best |")
print("|---|---|---|---:|---:|---:|---|---:|")
for m, p, c, v, best, ratio, br in rows:
    b = f"{br:.3f}×" if br else "-"
    print(f"| {m} | {p} | {c} | {b} | {ratio:.3f}× | {v:.3f} | {best:.3f} | {ratio:.3f} |")

print("\ncells below 1.0:")
for m, p, c, v, best, ratio, br in rows:
    if ratio < 1.0:
        print(f"- {m} {p} {c}: {ratio:.3f}× ({v:.3f} vs {best:.3f})")

print("\ncount on the table's rule (each row has 12 cells):")
for m in ("nomic", "jina"):
    for p in ("f16", "q8_0"):
        pg = medians("peregrine", m, p)
        others = {e: medians(e, m, "fp32") for e in ("onnxruntime", "fastembed", "openvino")}
        won = n = 0
        for c in CELLS:
            if c not in pg:
                continue
            best = max((o[c] for o in others.values() if c in o), default=0.0)
            if best <= 0:
                continue
            n += 1
            won += pg[c] >= best
        print(f"{m} {p}: {won} of {n} measured (of 12)")
