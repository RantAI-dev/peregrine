#!/usr/bin/env python3
"""The pass-width table: PASS_ROWS=32 against 128, per model, thread count and cell.

The per-round ratio (128's tokens/s over 32's, same round, alternating order) is the
claim; the medians and the round-to-round spread are beside it. Files are
`pass-rows-<model>-<threads>t*.jsonl`.
"""
import glob, json, statistics as st

CELLS = ["16", "128", "512", "16x32", "32x128", "8x512", "mixed"]
models = []
for f in sorted(glob.glob("pass-rows-*-1t*.jsonl")):
    m = f[len("pass-rows-"):].split("-1t")[0]
    if m not in models:
        models.append(m)
for f in sorted(glob.glob("pass-rows-*-10t.jsonl")):
    m = f[len("pass-rows-"):].split("-10t")[0]
    if m not in models:
        models.append(m)
models.sort()

print(f"{'model':<28} {'th':>3} {'cell':>8} {'32 tok/s':>9} {'128 tok/s':>10} "
      f"{'med':>7} {'range':>13} {'round ratios':>30} {'loads':>13}")
rows = []
for m in models:
    for th in (1, 10):
        runs = []
        for f in glob.glob(f"pass-rows-{m}-{th}t*.jsonl"):
            runs += [json.loads(l) for l in open(f)]
        if not runs:
            continue
        for cell in CELLS:
            by = {32: [], 128: []}
            rounds = sorted({r["round"] for r in runs if r["cell"] == cell})
            ratios, loads = [], []
            for rnd in rounds:
                a = [r for r in runs if r["cell"] == cell and r["round"] == rnd and r["rows"] == 32]
                b = [r for r in runs if r["cell"] == cell and r["round"] == rnd and r["rows"] == 128]
                if not a or not b:
                    continue
                by[32].append(a[0]["tokens_per_s"])
                by[128].append(b[0]["tokens_per_s"])
                ratios.append(b[0]["tokens_per_s"] / a[0]["tokens_per_s"])
                loads += [a[0].get("load1", 0), b[0].get("load1", 0)]
            if not ratios:
                continue
            a32, a128 = st.median(by[32]), st.median(by[128])
            med = st.median(ratios)
            print(f"{m:<28} {th:>3} {cell:>8} {a32:9.1f} {a128:10.1f} "
                  f"{med:7.3f} {min(ratios):6.3f}-{max(ratios):6.3f} "
                  f"{' '.join(f'{x:.3f}' for x in ratios):>30} "
                  f"{min(loads):4.1f}-{max(loads):4.1f}")
            rows.append(dict(model=m, threads=th, cell=cell, rows32=a32, rows128=a128,
                             med_ratio=med, ratio=a128 / a32, ratios=ratios, n=len(ratios),
                             loads=(min(loads), max(loads))))
json.dump(rows, open("summary.json", "w"), indent=1)
print()
worst = sorted(rows, key=lambda r: r["med_ratio"])[:8]
print("lowest 128/32 ratios:")
for r in worst:
    print(f"  {r['model']} {r['threads']}t {r['cell']}: {r['med_ratio']:.3f} "
          f"({['%.3f' % x for x in r['ratios']]})")
