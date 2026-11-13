#!/usr/bin/env python3
"""Medians and spreads of the S5.6 raw grid, and the Part 3 loss table.

Reads the raw records (one measurement each), groups by
(engine, pin, cell, threads, level) and prints median [min, max]. `before` and
`after` are separate: Peregrine's before files are `raw/before-pg-*.jsonl`, its
after files `raw/after-pg-*.jsonl`, and the other engines have one file each
(they did not change). Ratios are recomputed from these medians; nothing is typed
by hand into the README.

  python summarize.py            # both grids, medians and spreads
  python summarize.py --part3    # each grid's own-loss ratio and Peregrine ÷ engine
"""

import glob
import json
import statistics
import sys
from collections import defaultdict


def load(phase):
    groups = defaultdict(list)
    meta = defaultdict(list)
    for fn in sorted(glob.glob("raw/*.jsonl")):
        name = fn.split("/")[-1]
        if name.startswith("before-pg") and phase != "before":
            continue
        if name.startswith("after-pg") and phase != "after":
            continue
        if name.startswith(("before-pg-l", "after-pg")):
            pass
        for line in open(fn):
            r = json.loads(line)
            if "value" not in r:
                continue
            if r["engine"] == "pg" and not (
                name.startswith(phase + "-pg") or (phase == "before" and name.startswith("before-pg"))
            ):
                continue
            key = (r["engine"], r.get("pin", "yes"), r["cell"], r["threads"], r["level"])
            groups[key].append(r["value"])
            meta[key].append((r.get("load1", 0.0), r.get("idle", 0.0)))
    return groups, meta


def med(v):
    return statistics.median(v)


def table(phase, groups, meta):
    print(f"== {phase}")
    print(f'{"engine":5} {"pin":3} {"cell":11} {"thr":>3} {"level":>5} '
          f'{"n":>2} {"median":>9} {"min":>8} {"max":>8} {"load":>5} {"idle":>5}')
    for key in sorted(groups):
        e, p, c, t, l = key
        v = groups[key]
        loads = [m[0] for m in meta[key]]
        idles = [m[1] for m in meta[key]]
        print(f"{e:5} {p:3} {c:11} {t:3} {l:5} {len(v):2} {med(v):9.2f} "
              f"{min(v):8.2f} {max(v):8.2f} {med(loads):5.1f} {med(idles):5.2f}")


def part3(phase, groups):
    print(f"== {phase}: Peregrine's fraction of its own quiet cell, and its ratio to each engine")
    for cell in ["prompt512", "prompt4096", "decode", "embed32x128"]:
        for t in [4, 8, 10]:
            quiet = groups.get(("pg", "yes", cell, t, 0))
            if not quiet:
                continue
            base = med(quiet)
            for l in [0, 1, 2, 4]:
                row = groups.get(("pg", "yes", cell, t, l))
                if not row:
                    continue
                text = f"{cell:11} t{t:<2} l{l} pg {med(row):8.2f} own {med(row) / base:5.3f}"
                for e in ["llama", "ik"]:
                    other = groups.get((e, "yes", cell, t, l))
                    if other:
                        text += f"  ÷{e} {med(row) / med(other):5.2f}x"
                print(text)


def main():
    for phase in ["before", "after"]:
        groups, meta = load(phase)
        if "--part3" in sys.argv:
            part3(phase, groups)
        else:
            table(phase, groups, meta)


if __name__ == "__main__":
    main()
