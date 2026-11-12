#!/usr/bin/env python3
"""S5.4e's tables: medians of each mode and chunk, and per-round ratios
(Peregrine ÷ engine inside the round, then median), from runs.jsonl and
llama.txt / ik.txt."""
import json
import re
import statistics as st
import sys
from collections import defaultdict

here = sys.argv[1] if len(sys.argv) > 1 else "."
rows = [json.loads(l) for l in open(f"{here}/runs.jsonl")]
pg = defaultdict(list)
loads = defaultdict(list)
for r in rows:
    name = (
        r["model"]
        .replace("-Instruct-Q4_K_M.gguf", "")
        .replace("Qwen2.5-0.5B", "Qwen 2.5 0.5B")
        .replace("Llama-3.2-1B", "Llama 3.2 1B")
    )
    key = (name, r["prompt_tokens"], r["threads"], r["mode"], r["chunk"])
    pg[key].append((r["prefill_tok_s"], r["round"]))
    loads[key].append(r["load1"])

eng = {name: {} for name in ("llama.cpp", "ik_llama.cpp")}
for name, pat in (("llama.cpp", "llama.txt"), ("ik_llama.cpp", "ik.txt")):
    for block in open(f"{here}/{pat}").read().split("=== ")[1:]:
        head = block.splitlines()[0]
        m = re.search(r"round (\d+), prompt (\d+), threads (\d+)", head)
        tm = re.search(r"\|\s*([\d.]+) ±", block)
        if not (m and tm):
            continue
        model = "Qwen 2.5 0.5B" if "qwen2" in block else "Llama 3.2 1B"
        eng[name][(model, int(m.group(2)), int(m.group(3)), int(m.group(1)))] = float(
            tm.group(1)
        )


def med(v):
    return st.median(v)


def half(v):
    v = sorted(v)
    m = st.median(v)
    return m - v[0], v[-1] - m


models = ["Llama 3.2 1B", "Qwen 2.5 0.5B"]
for model in models:
    print(f"\n## {model}\n")
    print("| threads | prompt | exact | fast 64 | fast 128 | fast 256 | fast 512 | llama.cpp | ik |")
    print("|---|---|---|---|---|---|---|---|---|")
    for prompt in (512, 4096):
        for thr in (1, 10):
            def g(mode, chunk):
                v = [x for x, _ in pg.get((model, prompt, thr, mode, chunk), [])]
                return f"{med(v):.1f}" if v else "-"

            lv = [
                v
                for (mm, pp, tt, _), v in eng["llama.cpp"].items()
                if (mm, pp, tt) == (model, prompt, thr)
            ]
            iv = [
                v
                for (mm, pp, tt, _), v in eng["ik_llama.cpp"].items()
                if (mm, pp, tt) == (model, prompt, thr)
            ]
            print(
                f"| {thr} | {prompt} | {g('exact', None)} | {g('fast', 64)} | {g('fast', 128)} | "
                f"{g('fast', 256)} | {g('fast', 512)} | "
                f"{med(lv):.1f} | {med(iv):.1f} |"
            )


def per_round(cell, engname):
    """Median of (peregrine value ÷ engine value) over the rounds both saw."""
    ratios = []
    for v, rnd in cell:
        key = (cell_model, cell_prompt, cell_thr, rnd)
        if key in eng[engname]:
            ratios.append(v / eng[engname][key])
    return med(ratios) if ratios else float("nan")


print("\n## ratios: exact ÷ engine (per round, median)\n")
print("| model | threads | prompt | ÷ llama.cpp | ÷ ik |")
print("|---|---|---|---|---|")
for model in models:
    for prompt in (512, 4096):
        for thr in (1, 10):
            cell_model, cell_prompt, cell_thr = model, prompt, thr
            e = per_round(pg[(model, prompt, thr, "exact", None)], "llama.cpp")
            i = per_round(pg[(model, prompt, thr, "exact", None)], "ik_llama.cpp")
            print(f"| {model} | {thr} | {prompt} | {e:.2f}× | {i:.2f}× |")

print("\n## ratios: fast (best chunk by median) ÷ engine (per round, median)\n")
print("| model | threads | prompt | best chunk | fast ÷ llama.cpp | fast ÷ ik | fast ÷ exact |")
print("|---|---|---|---|---|---|---|")
for model in models:
    for prompt in (512, 4096):
        for thr in (1, 10):
            cell_model, cell_prompt, cell_thr = model, prompt, thr
            best, bc = 0, None
            for c in (64, 128, 256, 512):
                v = med([x for x, _ in pg[(model, prompt, thr, "fast", c)]])
                if v > best:
                    best, bc = v, c
            e = med([x for x, _ in pg[(model, prompt, thr, "exact", None)]])
            l = per_round(pg[(model, prompt, thr, "fast", bc)], "llama.cpp")
            i = per_round(pg[(model, prompt, thr, "fast", bc)], "ik_llama.cpp")
            print(
                f"| {model} | {thr} | {prompt} | {bc} | {l:.2f}× | {i:.2f}× | "
                f"{best / e:.2f}× |"
            )

print("\n## spread and load\n")
for key in sorted(pg):
    model, prompt, thr, mode, chunk = key
    vals = [x for x, _ in pg[key]]
    lo, hi = half(vals)
    print(
        f"- {model}, {prompt} tok, {thr} th, {mode} {chunk}: "
        f"median {med(vals):.1f}, -{lo:.1f}/+{hi:.1f}, n={len(vals)}, "
        f"load {min(loads[key]):.1f}-{max(loads[key]):.1f}"
    )
