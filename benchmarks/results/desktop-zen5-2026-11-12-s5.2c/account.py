#!/usr/bin/env python3
"""S5.2c: the phase account of one prompt pass, from `promptbench`'s own slots.

    python3 account.py <files...>

Each file is one `promptbench` JSON line (`PEREGRINE_PROF2=1`). Prints, per cell, the
share of the engine's `prefill_s` for every caller-side slot and the count of
fork-joins, so the serial remainder is named step by step. Medians over the files
given for the same cell.
"""

import json
import sys
from collections import defaultdict

SLOTS = ["matmul qkv", "matmul wo", "matmul gate/up", "matmul down", "output head",
         "attention", "norms", "quantise", "rope+bias", "kv write", "residual",
         "swiglu", "barrier wait", "other"]
SUB = ["  attn scores", "  attn softmax", "  attn mix", "  attn kv decode"]


def main():
    cells = defaultdict(list)
    for path in sys.argv[1:]:
        for line in open(path):
            d = json.loads(line)
            cells[(d["model"] if "model" in d else path, d["len"], d["threads"])].append(d)
    for (tag, ln, th), rows in sorted(cells.items(), key=lambda kv: (str(kv[0][0]), kv[0][1], kv[0][2])):
        med = {n: sorted(r["prof_ms"].get(n, 0) for r in rows)[len(rows) // 2] / 1e6 for n in SLOTS + SUB}
        totals = sorted(r["prefill_s"] * 1e3 for r in rows)
        tot = totals[len(totals) // 2]
        toks = sorted(r["prefill_tok_s"] for r in rows)[len(rows) // 2]
        barriers = sorted(r["stats_ms"]["barriers"] for r in rows)[len(rows) // 2]
        print(f"{tag} L{ln} t{th}: {tot:.0f} ms  {toks:.0f} tok/s  {barriers} joins")
        for n in SLOTS:
            if med[n] > 0.05:
                print(f"   {n:14s} {med[n]:8.1f} ms {med[n] / tot * 100:5.2f} %")
        mix = med["attention"] - sum(med[n] for n in SUB)
        print(f"   attention's own fork+prologue {mix:8.1f} ms {mix / tot * 100:5.2f} %")
        for n in SUB:
            if med[n] > 0.05:
                print(f"     {n:18s} {med[n]:8.1f} ms {med[n] / tot * 100:5.2f} %")


if __name__ == "__main__":
    main()
