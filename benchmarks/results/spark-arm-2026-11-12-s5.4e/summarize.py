#!/usr/bin/env python3
"""Summarize S5.4e's runs: medians and half-ranges per cell, from runs.jsonl and
the llama.cpp / ik_llama.cpp blocks in llama.txt and ik.txt."""
import json
import re
import statistics as st
import sys
from collections import defaultdict

here = sys.argv[1] if len(sys.argv) > 1 else "."
rows = [json.loads(l) for l in open(f"{here}/runs.jsonl")]
cells = defaultdict(list)
for r in rows:
    key = (r["model"], r["prompt_tokens"], r["threads"], r["mode"], r["chunk"])
    cells[key].append((r["prefill_tok_s"], r["load1"], r["round"]))


def med(vals):
    v = sorted(vals)
    m = st.median(v)
    return m, m - v[0], v[-1] - m


print("Peregrine")
for key in sorted(cells):
    model, prompt, thr, mode, chunk = key
    vals = [v for v, _, _ in cells[key]]
    loads = [l for _, l, _ in cells[key]]
    m, lo, hi = med(vals)
    print(
        f"{model[:12]:12} {prompt:>4} tok {thr:>2} th {mode:5} "
        f"chunk {str(chunk):>4}  {m:7.1f} -{lo:4.1f} +{hi:4.1f}  "
        f"n={len(vals)} load {min(loads):.1f}-{max(loads):.1f}"
    )

for name, pat in (("llama.cpp", "llama.txt"), ("ik_llama.cpp", "ik.txt")):
    print(name)
    eng = defaultdict(list)
    text = open(f"{here}/{pat}").read()
    for block in text.split("=== ")[1:]:
        head = block.splitlines()[0]
        m = re.search(r"prompt (\d+), threads (\d+), load1 ([\d.]+)", head)
        tm = re.search(r"\|\s*([\d.]+) ± ([\d.]+) \|", block)
        mm = re.search(r"\| (llama|qwen2|qwen3) .*?\|", block)
        if not (m and tm):
            continue
        prompt, thr, load = int(m.group(1)), int(m.group(2)), float(m.group(3))
        model = "Qwen2.5-0.5B" if "qwen2" in block else "Llama-3.2-1B"
        eng[(model, prompt, thr)].append((float(tm.group(1)), load))
    for key in sorted(eng):
        model, prompt, thr = key
        vals = [v for v, _ in eng[key]]
        loads = [l for _, l in eng[key]]
        m, lo, hi = med(vals)
        print(
            f"{model[:12]:12} {prompt:>4} tok {thr:>2} th {name:12} "
            f"{m:7.1f} -{lo:4.1f} +{hi:4.1f}  n={len(vals)} "
            f"load {min(loads):.1f}-{max(loads):.1f}"
        )
