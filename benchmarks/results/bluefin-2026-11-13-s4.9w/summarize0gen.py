#!/usr/bin/env python3
"""S4.9w Part 0 generation: medians per arm and AFTER/BEFORE for Llama 3.2 1B."""
import glob
import json
import statistics
from pathlib import Path

P = Path(__file__).resolve().parent / "parts" / "part0-gen"
MED_A = {"decode_tok_s": ["single_user", "decode_tok_s"],
         "prefill_tok_s": ["single_user", "prefill_tok_s"],
         "ten_user_tok_s": ["multi_user_10", "aggregate_tok_s"]}


def val(d, path):
    for p in path:
        d = d["summary"]["512"][p[0]][p[1]] if p == path[-1] else d["results"]["512"]
    return d if not isinstance(d, dict) else d.get("median")


def load(arm, key):
    vals = {k: [] for k in MED_A}
    for r in (1, 2, 3, 4, 5):
        fs = glob.glob(str(P / f"r{r}-{arm}" / "*peregrine*.json"))
        if not fs:
            continue
        d = json.load(open(fs[0]))
        s = d["summary"]["512"]
        vals["decode_tok_s"].append(s["single_user"]["decode_tok_s"]["median"])
        vals["prefill_tok_s"].append(s["single_user"]["prefill_tok_s"]["median"])
        vals["ten_user_tok_s"].append(s["multi_user_10"]["aggregate_tok_s"]["median"])
    return {k: statistics.median(v) for k, v in vals.items() if v}, vals


print("| cell | before | after | after/before | nostall | nostall/before |")
print("|---|---:|---:|---:|---:|---:|")
for k in MED_A:
    med = {}
    raw = {}
    for arm in ("before", "after", "after_nostall"):
        med[arm], raw[arm] = load(arm, k)
    b, a, n = med["before"][k], med["after"][k], med["after_nostall"][k]
    print(f"| {k} | {b:.2f} | {a:.2f} | {a / b:.3f} | {n:.2f} | {n / b:.3f} |")
    for arm in ("before", "after", "after_nostall"):
        print(f"  - {arm}: {raw[arm][k]}")
