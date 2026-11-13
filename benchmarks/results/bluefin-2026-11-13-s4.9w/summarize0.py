#!/usr/bin/env python3
"""S4.9w Part 0: medians per arm and AFTER/BEFORE ratios from the raw JSON."""
import glob
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
P = ROOT / "parts" / "part0"
ARMS = ["before", "after", "after_nostall"]
MODELS = ["nomic", "jina", "bge-small", "minilm", "qwen3-embed-0.6b"]
CELLS = ["L16_b8_c1", "L16_b32_c1", "L128_b1_c1", "L128_b8_c1",
         "L128_b32_c1", "L512_b1_c1", "L512_b8_c1"]


def load(arm, model):
    vals = {}
    for f in sorted(glob.glob(str(P / f"r*-{arm}" / f"{model}-peregrine-*-r*.json"))):
        d = json.load(open(f))
        rnd = d["round"]
        for c, v in d["cells"].items():
            vals.setdefault(c, {})[rnd] = v["texts_per_s"]
    return vals


def med(xs):
    return statistics.median(xs)


def spread(xs):
    return (min(xs), max(xs)) if xs else (0, 0)


out = {}
for m in MODELS:
    out[m] = {a: load(a, m) for a in ARMS}

print("| cell | before | after | after/before | nostall | nostall/before |")
print("|---|---:|---:|---:|---:|---:|")
rows = []
for m in MODELS:
    for c in CELLS:
        b = out[m]["before"].get(c, {})
        if not b:
            continue
        bv = [b[r] for r in sorted(b)]
        av = [out[m]["after"][c][r] for r in sorted(out[m]["after"].get(c, {}))]
        nv = [out[m]["after_nostall"][c][r] for r in sorted(out[m]["after_nostall"].get(c, {}))]
        bm, am, nm = med(bv), med(av), med(nv)
        cell = f"{m} {c}"
        print(f"| {cell} | {bm:.3f} | {am:.3f} | {am / bm:.3f} | {nm:.3f} | {nm / bm:.3f} |")
        rows.append((cell, bm, am, am / bm, spread(bv), spread(av), nm, nm / bm, spread(nv)))

print()
print("spreads (min-max of the five rounds):")
for cell, bm, am, ar, bs, as_, nm, nr, ns in rows:
    print(f"- {cell}: before {bs[0]:.2f}-{bs[1]:.2f}, after {as_[0]:.2f}-{as_[1]:.2f}, nostall {ns[0]:.2f}-{ns[1]:.2f}")

print()
print("CPU idle share per round (idle / total ticks):")
for f in sorted(P.glob("r*-cpu.txt")):
    txt = f.read_text().strip()
    parts = txt.split()
    busy0, idle0 = int(parts[3]), int(parts[4])
    busy1, idle1 = int(parts[6]), int(parts[7])
    db, di = busy1 - busy0, idle1 - idle0
    print(f"- {f.name}: idle {di / (db + di) * 100:.1f}%  busy {db} idle {di}")
