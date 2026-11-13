#!/usr/bin/env python3
"""Medians, spreads and the arm ratios of the S5.6 re-read.

Reads the JSONL the two drivers append and prints one table per phase. Every ratio
is recomputed from the medians here; none is typed. A ratio is `after / before`:
`merged/gap0` is the gap-stretched window's cost, `merged/head` is whether the merge
kept S5.6's numbers, `fixed/gap0` is what the review fix (no stretch while the
run-queue delay is at or above the stall gate) recovered.
"""

import glob
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
CELLS = ["prompt512", "prompt4096", "decode", "embed32x128"]
LEVELS = [0, 1, 2, 4]


def load(pat):
    recs = []
    for f in glob.glob(str(HERE / pat)):
        for line in open(f):
            r = json.loads(line)
            if "value" in r:
                recs.append(r)
    return recs


def table(recs, arms, title):
    g = defaultdict(list)
    for r in recs:
        g[(r["cell"], r["level"], r["arm"])].append(r["value"])
    print(f"=== {title}")
    for cell in CELLS:
        print(f"-- {cell}")
        print("| arm | " + " | ".join(f"L{lv}" for lv in LEVELS) + " |")
        print("|---|" + "---|" * len(LEVELS))
        for arm in arms:
            cells = []
            for lv in LEVELS:
                v = g.get((cell, lv, arm), [])
                if not v:
                    cells.append("—")
                    continue
                med = st.median(v)
                half = (max(v) - min(v)) / 2 / med * 100
                cells.append(f"{med:.1f} (±{half:.1f} %)")
            print(f"| {arm} | " + " | ".join(cells) + " |")
        for lv in LEVELS:
            def m(a):
                v = g.get((cell, lv, a), [])
                return st.median(v) if v else None

            parts = []
            h, a, b = m("head"), m("merged"), m("gap0")
            if a and b:
                parts.append(f"merged/gap0 {a / b:.3f}")
            if a and h:
                parts.append(f"merged/head {a / h:.3f}")
            if m("fixed") and b:
                parts.append(f"fixed/gap0 {m('fixed') / b:.3f}")
            if m("fixed") and a:
                parts.append(f"fixed/merged {m('fixed') / a:.3f}")
            if parts:
                print(f"  L{lv}: " + ", ".join(parts))


def main():
    table(load("raw/reread-*.jsonl"), ["head", "merged", "gap0"], "before the fix")
    table(load("raw/after-*.jsonl"), ["head", "merged", "fixed", "gap0"], "after the fix")


if __name__ == "__main__":
    main()
