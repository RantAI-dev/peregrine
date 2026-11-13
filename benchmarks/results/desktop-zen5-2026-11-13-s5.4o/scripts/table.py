#!/usr/bin/env python3
"""The S5.4o tables from the raw files: medians per (cell, arm), the ratios, the spread.

Every ratio is recomputed from the medians in the same table, and a cell the run
skipped is `—`, never 0. Empty and zero records (a failed arm, re-run separately) are
dropped and their count printed, so a table never silently averages a failure in.
"""
import glob
import json
import statistics
import sys

ARMS = ["exact", "fast-before", "fast", "ik", "ik-rtr", "llama"]


def load(files):
    d = {}
    zeros = 0
    for f in files:
        for line in open(f):
            r = json.loads(line)
            if not r["tok_s"]:
                zeros += 1
                continue
            d.setdefault((r["model"], r["p"], r["t"], r["arm"]), []).append(r["tok_s"])
    return d, zeros


def main():
    files = sys.argv[1:] or sorted(glob.glob("raw/sweep-*.jsonl")) + sorted(
        glob.glob("raw/a2-*.jsonl")
    )
    d, zeros = load(files)
    cells = sorted({k[:3] for k in d})
    arms = [a for a in ARMS if any(k[3] == a for k in d)]
    print("| model | prompt / threads | " + " | ".join(arms) + " |")
    print("|---|---|" + "---|" * len(arms))
    for c in cells:
        med = {}
        for a in arms:
            v = d.get(c + (a,), [])
            med[a] = statistics.median(v) if v else 0.0
        print(
            f"| {c[0].split('-')[0]} | {c[1]} / {c[2]} | "
            + " | ".join(f"{med[a]:.1f}" if med[a] else "—" for a in arms)
            + " |"
        )
    print()
    for c in cells:
        for a in arms:
            v = d.get(c + (a,), [])
            if not v:
                continue
            spread = 1 - min(v) / max(v)
            print(
                f"{c[0][:12]} {c[1]:>5}/{c[2]:>2} {a:11s} n={len(v)} "
                f"med={statistics.median(v):7.1f} min={min(v):7.1f} max={max(v):7.1f} "
                f"spread={spread:.3f}"
            )
    if zeros:
        print(f"\n{zeros} failed/zero records dropped (each was re-run separately)")


if __name__ == "__main__":
    main()
