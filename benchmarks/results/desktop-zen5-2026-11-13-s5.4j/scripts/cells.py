#!/usr/bin/env python3
"""The eight cells' medians and every ratio, from the raw sweep files."""
import glob, json, statistics, sys

d = {}
for f in sorted(glob.glob('raw/sweep-*.jsonl')):
    for line in open(f):
        r = json.loads(line)
        d.setdefault((r['model'], r['p'], r['t'], r['arm']), []).append(r['tok_s'])
arms = ['exact', 'fast-before', 'fast', 'ik', 'ik-rtr', 'llama']
for c in sorted({k[:3] for k in d}):
    med = {a: (statistics.median(d[c + (a,)]) if c + (a,) in d else 0.0) for a in arms}
    n = {a: len(d[c + (a,)]) for a in arms}
    def q(a, b):
        return f'{med[a]/med[b]:.3f}' if med[a] and med[b] else '—'
    print(f"{c[0][:6]:7s} {c[1]:5d}/{c[2]:2d} " +
          ' '.join(f'{a}={med[a]:.1f}' for a in arms) +
          f" | exact {q('fast','exact')} ik {q('fast','ik')} rtr {q('fast','ik-rtr')} "
          f"llama {q('fast','llama')} | before->after {q('fast','fast-before')} "
          f"n={[n[a] for a in arms]}")
