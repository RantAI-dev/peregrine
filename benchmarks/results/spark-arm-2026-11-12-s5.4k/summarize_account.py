#!/usr/bin/env python3
"""Rebuild the S5.4k stage account from account.jsonl.

The caller's slots (`matmul qkv`, ... `other`) are wall time on the thread that drives
the pool, so a ten-thread cell's matmul already contains the wait for its workers. The
worker-side slots (`fmt ...`, `  attn ...`) are summed over the pool, so a ten-thread
figure is up to ten times the wall time they took.
"""
import json, statistics as st, sys

runs = [json.loads(l) for l in open(sys.argv[1] if len(sys.argv) > 1 else "account.jsonl")]
keys = ["matmul qkv", "matmul wo", "matmul gate/up", "matmul down", "attention",
        "  attn scores", "  attn softmax", "  attn mix", "norms", "quantise",
        "rope+bias", "kv write", "residual", "swiglu", "barrier wait", "other",
        "output head", "  fmt Q5_0", "  fmt Q8_0", "  fmt Q6_K", "  fmt Q4_K"]
print(f"{'model':<8} {'cell':>9} {'tok/s':>8} {'wall':>7} " +
      " ".join(f"{k.split()[-1][:7]:>8}" for k in keys))
for m in sorted({r["model"] for r in runs}):
    for length in (512, 4096):
        for th in (1, 10):
            cells = [r for r in runs if r["model"] == m and r["len"] == length
                     and r["threads"] == th]
            if not cells:
                continue
            med = lambda f: st.median([f(c) for c in cells])
            tps = med(lambda c: c["prefill_tok_s"])
            wall = med(lambda c: c["stats_ms"]["total"]) / 1e3
            vals = []
            for k in keys:
                v = med(lambda c, k=k: c["prof_ms"].get(k, 0)) / 1e6
                vals.append(v)
            print(f"{m[:8]:<8} {length:>4}/{th:<4} {tps:8.1f} {wall:7.2f} " +
                  " ".join(f"{v:8.3f}" for v in vals))
