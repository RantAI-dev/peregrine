#!/usr/bin/env python3
"""S4.9w Part 1: the row-block three arms (old cap, whole pass, 2 MiB bound).

Reads the per-arm/per-round embed.py JSON and prints, per model and cell, the median
texts/s of each arm, the three ratios recomputed from the medians, and the spread of the
per-round BOUND ratio.  A cell is named when BOUND's median is below OLD's or WHOLE's by
more than the inter-round spread of that same ratio.
"""
import glob
import json
import statistics
from pathlib import Path

P = Path(__file__).resolve().parent / "parts" / "part1"
ARMS = ("old", "whole", "bound")
MODELS = [
    ("nomic", "f16"),
    ("jina", "f16"),
    ("qwen3-embed-0.6b", "f16"),
    ("bge-small", "f32"),
    ("minilm", "f16"),
    ("nemotron-3.5-embed-1b", "f16"),
]


def load(arm, model):
    """cell -> list of (round, texts_per_s)"""
    out = {}
    for f in sorted(glob.glob(str(P / f"r*-{arm}" / f"{model}-peregrine-*-r*.json"))):
        d = json.load(open(f))
        r = d["round"]
        for cell, c in d["cells"].items():
            out.setdefault(cell, []).append((r, c["texts_per_s"]))
    return out


def order(cell):
    if cell.startswith("mixed"):
        return (1, 0)
    n, b = cell[1:].split("_b")
    return (0, int(n), int(b.split("_")[0]))


for model, prec in MODELS:
    data = {a: load(a, model) for a in ARMS}
    cells = sorted(data["bound"], key=order)
    print(f"\n**{model} {prec.upper()}**\n")
    print("| cell | OLD | WHOLE | BOUND | BOUND/OLD | WHOLE/OLD | BOUND/WHOLE | rounds (min–max) |")
    print("|---|---:|---:|---:|---:|---:|---:|---|")
    for cell in cells:
        med = {a: statistics.median([v for _, v in data[a][cell]]) for a in ARMS}
        spread = {}
        for a in ARMS:
            vs = [v for _, v in data[a][cell]]
            spread[a] = (min(vs), max(vs))
        ratio_rounds = []
        for (ra, va), (_, vb) in zip(sorted(data["old"][cell]), sorted(data["bound"][cell])):
            ratio_rounds.append(vb / va)
        rs = (min(ratio_rounds), max(ratio_rounds))
        print(
            f"| {cell} | {med['old']:.3f} | {med['whole']:.3f} | {med['bound']:.3f} "
            f"| {med['bound'] / med['old']:.3f} | {med['whole'] / med['old']:.3f} "
            f"| {med['bound'] / med['whole']:.3f} "
            f"| B/O {rs[0]:.3f}–{rs[1]:.3f}; O {spread['old'][0]:.2f}–{spread['old'][1]:.2f}; "
            f"W {spread['whole'][0]:.2f}–{spread['whole'][1]:.2f}; "
            f"B {spread['bound'][0]:.2f}–{spread['bound'][1]:.2f} |"
        )
    print("\nFLAGS (BOUND below OLD or WHOLE by more than the ratio's round spread):")
    flagged = False
    for cell in cells:
        for other in ("old", "whole"):
            ro = []
            for (_, va), (_, vb) in zip(sorted(data[other][cell]), sorted(data["bound"][cell])):
                ro.append(vb / va)
            med_ratio = statistics.median(
                [v for _, v in data["bound"][cell]]
            ) / statistics.median([v for _, v in data[other][cell]])
            if med_ratio < 1.0 - (max(ro) - min(ro)):
                flagged = True
                print(
                    f"  - {cell}: BOUND/ {'OLD' if other == 'old' else 'WHOLE'} = "
                    f"{med_ratio:.3f}, rounds {min(ro):.3f}–{max(ro):.3f}"
                )
    if not flagged:
        print("  none")

# ---------------------------------------------------------------- 1 MiB / 4 MiB
S = P.parent / "part1-side"
print("\n## The bound's constant (1 MiB / 2 MiB / 4 MiB, 128- and 2 000-token cells)\n")
print("| model | cell | 1 MiB | 2 MiB | 4 MiB | 1MiB/2MiB | 4MiB/2MiB |")
print("|---|---:|---:|---:|---:|---:|---:|")
for model, _prec in MODELS:
    data = {}
    for a in ("1m", "bound", "4m"):
        data[a] = {}
        for f in sorted(glob.glob(str(S / f"r*-{a}" / f"{model}-peregrine-*-r*.json"))):
            d = json.load(open(f))
            for cell, c in d["cells"].items():
                data[a].setdefault(cell, []).append(c["texts_per_s"])
    for cell in sorted(data["bound"], key=order):
        med = {a: statistics.median(data[a][cell]) for a in data}
        print(
            f"| {model} | {cell} | {med['1m']:.3f} | {med['bound']:.3f} | {med['4m']:.3f} "
            f"| {med['1m'] / med['bound']:.3f} | {med['4m'] / med['bound']:.3f} |"
        )
