#!/usr/bin/env python3
"""The S5.2b tables, from the JSONL this directory's drivers wrote."""
import glob
import json
import statistics
import sys


def rows(pat):
    out = []
    for f in sorted(glob.glob(pat)):
        for line in open(f):
            out.append(json.loads(line))
    return out


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def spread(xs):
    return (max(xs) - min(xs)) / med(xs) * 100 if xs else float("nan")


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    pg = rows("pgrid-*.jsonl")
    pp = rows("pprof-*.jsonl")
    ik = rows("ik-*.jsonl")
    ab = rows("ab-*.jsonl")

    if what in ("all", "grid"):
        cells = {}
        for r in pg:
            if r["series"] != "pgrid":
                continue
            cells.setdefault((r["model"], r["len"], r["threads"]), []).append(
                r["peregrine"]["prefill_tok_s"])
        abc = {}
        for r in ab:
            if r["decode"] or r["slots"] > 1:
                continue
            for arm in ("before", "after"):
                abc.setdefault((r["model"], r["len"], r["threads"], arm), []).append(
                    r["arms"][arm]["prefill_tok_s"])
        iki = {}
        for r in ik:
            if r["series"] != "ikgrid":
                continue
            iki.setdefault((r["model"], r["len"], r["threads"], r["ik"]["ub"]), []).append(
                r["ik"]["ts"])
        print("| model | prompt | threads | P before | P after | ik ub512 | ik ub64 | after÷ik512 |")
        print("|---|---|---|---|---|---|---|---|")
        for model in ("llama1b", "qwen05"):
            for ln in (512, 4096):
                for th in (1, 6, 12, 24):
                    p = med(cells.get((model, ln, th), []))
                    a = med(abc.get((model, ln, th, "after"), []))
                    b = med(abc.get((model, ln, th, "before"), []))
                    i512 = med(iki.get((model, ln, th, 512), []))
                    i64 = med(iki.get((model, ln, th, 64), []))
                    print(f"| {model} | {ln} | {th} | {p:.1f} | {a:.1f} | {i512:.1f} | "
                          f"{i64:.1f} | {a / i512:.2f} |")
        print()
        print("spreads (max-min)/median, %:")
        for model in ("llama1b", "qwen05"):
            for ln in (512, 4096):
                for th in (1, 6, 12, 24):
                    p = cells.get((model, ln, th), [])
                    a = abc.get((model, ln, th, "after"), [])
                    b = abc.get((model, ln, th, "before"), [])
                    if not p:
                        continue
                    print(f"  {model} L{ln} t{th}: P {spread(p):.1f} "
                          f"(n={len(p)}) | before {spread(b):.1f} after {spread(a):.1f} "
                          f"(n={len(a)}) | ratio {a and (med(a) / med(b)):.3f}")

    if what in ("all", "stages"):
        print()
        print("stage split, % of the measured prefill wall (prof2=full, median of runs):")
        cells = {}
        for r in pp:
            if r["series"] != "pprof":
                continue
            cells.setdefault((r["model"], r["len"], r["threads"]), []).append(r["peregrine"])
        for k in sorted(cells):
            runs = cells[k]
            wall = med([x["prefill_s"] for x in runs])
            w = runs[0]["width"]
            acc = {}
            for x in runs:
                for n, v in x["prof_ms"].items():
                    if n.startswith("dec"):
                        continue
                    acc[n] = acc.get(n, 0) + v / 1e9
            acc = {n: v / len(runs) for n, v in acc.items()}
            core = {n: v / w for n, v in acc.items() if n.startswith("  ")}
            mat = sum(acc.get(n, 0) for n in
                      ("matmul qkv", "matmul wo", "matmul gate/up", "matmul down"))
            attn = acc.get("attention", 0)
            rest = sum(acc.get(n, 0) for n in
                       ("norms", "quantise", "rope+bias", "kv write", "residual", "swiglu",
                        "other", "output head"))
            wait = acc.get("barrier wait", 0)
            print(f"{k[0]} L{k[1]} t{k[2]}: wall {wall:.2f}s tok/s "
                  f"{med([x['prefill_tok_s'] for x in runs]):.1f} | matmul {mat / wall * 100:.1f}% "
                  f"attn {attn / wall * 100:.1f}% (scores {core.get('  attn scores', 0) / wall * 100:.1f}"
                  f" mix {core.get('  attn mix', 0) / wall * 100:.1f}"
                  f" dec {core.get('  attn kv decode', 0) / wall * 100:.1f})"
                  f" rest {rest / wall * 100:.1f}% wait {wait / wall * 100:.1f}%"
                  f" | per-core GMAC/s: scores {attn / wall:.2f}?")

    if what in ("all", "ab"):
        print()
        print("A/B (before/after), from ab-*.jsonl:")
        for key in sorted({(r["model"], r["len"], r["threads"], r["decode"], r["slots"])
                           for r in ab}):
            if key[3] or key[4] > 1:
                continue
            bs = [r["arms"]["before"]["prefill_tok_s"] for r in ab
                  if (r["model"], r["len"], r["threads"], r["decode"], r["slots"]) == key]
            as_ = [r["arms"]["after"]["prefill_tok_s"] for r in ab
                   if (r["model"], r["len"], r["threads"], r["decode"], r["slots"]) == key]
            print(f"  {key}: before {med(bs):.1f} [{min(bs):.0f}-{max(bs):.0f}] "
                  f"after {med(as_):.1f} [{min(as_):.0f}-{max(as_):.0f}] "
                  f"ratio {med(as_) / med(bs):.3f} n={len(as_)}")


if __name__ == "__main__":
    main()
