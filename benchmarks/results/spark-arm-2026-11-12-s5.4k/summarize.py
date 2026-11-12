#!/usr/bin/env python3
"""Rebuild the S5.4k tables from runs.jsonl, llama.txt and ik.txt."""
import json, re, statistics as st

runs = [json.loads(l) for l in open("runs.jsonl")]

def engine(path):
    """{(model-tag, prompt, threads): [round values]}"""
    cells, cur = {}, None
    for line in open(path):
        line = line.strip()
        m = re.match(r"=== round (\d+), prompt (\d+), threads (\d+), load1 ([\d.]+)", line)
        if m:
            cur = (int(m.group(2)), int(m.group(3)))
            continue
        if line.startswith("|") and "pp" in line:
            parts = [p.strip() for p in line.split("|") if p.strip()]
            nums = [p for p in parts if "±" in p]
            if cur and nums:
                tag = "Qwen" if parts[0].startswith("qwen2") else "Llama"
                v = float(nums[-1].split("±")[0])
                cells.setdefault((tag, *cur), []).append(v)
    return cells

llama = engine("llama.txt")
ik = engine("ik.txt")

models = ["Qwen2.5-0.5B-Instruct-Q4_K_M.gguf", "Llama-3.2-1B-Instruct-Q4_K_M.gguf"]
short = {"Qwen2.5-0.5B-Instruct-Q4_K_M.gguf": "Qwen 0.5B",
         "Llama-3.2-1B-Instruct-Q4_K_M.gguf": "Llama 1B"}

print(f"{'model':<10} {'cell':>9} {'exact':>8} {'fast':>8} {'f/exact':>8} "
      f"{'llama':>8} {'ik':>8} {'fast/llama':>10} {'fast/ik':>8}  loads")
rows = []
for m in models:
    for prompt in (512, 4096):
        for threads in (1, 10):
            def med(mode):
                v = [r["prefill_tok_s"] for r in runs
                     if r["model"] == m and r["prompt_tokens"] == prompt
                     and r["threads"] == threads and r["mode"] == mode]
                return (st.median(v), v) if v else (float("nan"), [])
            (e, ev), (f, fv) = med("exact"), med("fast")
            tag = "Qwen" if m.startswith("Qwen") else "Llama"
            key = (tag, prompt, threads)
            lv = llama.get(key, [])
            iv = ik.get(key, [])
            lm = st.median(lv) if lv else float("nan")
            im = st.median(iv) if iv else float("nan")
            loads = [r["load1"] for r in runs if r["model"] == m
                     and r["prompt_tokens"] == prompt and r["threads"] == threads]
            cell = f"{prompt}/{threads}"
            print(f"{short[m]:<10} {cell:>9} {e:8.1f} {f:8.1f} {f/e:8.3f} "
                  f"{lm:8.1f} {im:8.1f} {f/lm:10.3f} {f/im:8.3f}  "
                  f"{min(loads):.1f}-{max(loads):.1f}")
            rows.append(dict(model=short[m], cell=cell, exact=e, fast=f,
                             llama=lm, ik=im, f_over_exact=f/e,
                             f_over_llama=f/lm, f_over_ik=f/im,
                             fast_rounds=fv, llama_rounds=lv, ik_rounds=iv,
                             loads=(min(loads), max(loads))))
print()
print("raw rounds per cell:")
for r in rows:
    print(f"  {r['model']:<10} {r['cell']:>9}  fast {['%.1f' % v for v in r['fast_rounds']]}"
          f"  llama {['%.1f' % v for v in r['llama_rounds']]}"
          f"  ik {['%.1f' % v for v in r['ik_rounds']]}")
json.dump(rows, open("summary.json", "w"), indent=1)
