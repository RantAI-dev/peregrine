#!/usr/bin/env python3
"""The S5.4j standings from the raw sweep files: medians per (cell, arm), the ratios,
and the rounds' spread. Every ratio is recomputed from the medians in the row."""
import glob, json, statistics, sys

files = sys.argv[1:] or sorted(glob.glob('raw/sweep-*.jsonl'))
d = {}
for f in files:
    for line in open(f):
        r = json.loads(line)
        d.setdefault((r['model'], r['p'], r['t'], r['arm']), []).append(r['tok_s'])
cells = sorted({k[:3] for k in d})
arms = ['exact', 'fast-before', 'fast', 'ik', 'ik-rtr', 'llama']
print('| model | p/t | ' + ' | '.join(arms) + ' | fast÷exact | fast÷ik | fast÷ik-rtr | fast÷llama |')
print('|---|---|' + '---|' * len(arms) + '---|---|---|---|')
for c in cells:
    med = {}
    for a in arms:
        v = d.get(c + (a,), [])
        med[a] = statistics.median(v) if v else 0.0
    def q(a, b):
        return f'{med[a]/med[b]:.3f}' if med[a] and med[b] else '—'
    print(f'| {c[0].split("-")[0]} | {c[1]}/{c[2]} | ' +
          ' | '.join(f'{med[a]:.1f}' if med[a] else '—' for a in arms) +
          f' | {q("fast","exact")} | {q("fast","ik")} | {q("fast","ik-rtr")} | {q("fast","llama")} |')
print()
for c in cells:
    for a in arms:
        v = d.get(c + (a,), [])
        if not v:
            continue
        rng = (min(v) / max(v)) if max(v) else 0
        print(f'{c[0][:6]} {c[1]}/{c[2]} {a:11s} n={len(v)} med={statistics.median(v):7.1f} '
              f'min={min(v):7.1f} max={max(v):7.1f} spread={1-rng:.3f}')
