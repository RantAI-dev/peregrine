#!/usr/bin/env python3
"""The S5.4k account table: the engine's own stage timers, split where the timer is.

MAC counts per token (the 2-D matrices a prompt pass multiplies; the token
embedding is gathered and the head reads one row):
  Qwen 2.5 0.5B (24 layers, d 896, ff 4864, 14 q / 2 kv heads of 64):
    Q5_0 253.2 M (q, k, v, wo, gate, up), Q8_0 2.75 M (attn_v),
    Q4_K 52.3 M and Q6_K 52.3 M (ffn_down, twelve layers each).
  Llama 3.2 1B (16 layers, d 2048, ff 8192, 32 q / 8 kv heads of 64):
    Q4_K 704.6 M (q, k, v, wo, gate, up), Q6_K 268.4 M (ffn_down).

The matmul groups are the caller's walls; a group's GMAC/s is its own MAC count
over its own wall. Attention is causal: `n_head · hd · n(n+1)/2` per layer for
the scores and the same again for the mix; its "scores" and "mix" subtimers are
worker-side, so a ten-thread figure is summed over the pool (the wall is `attention`).
"""
import json, statistics as st, sys

MODELS = {
    "Qwen2.5-0.5B-Instruct-Q4_K_M.gguf": dict(
        name="Qwen 0.5B", n_layer=24, n_head=14, hd=64,
        groups=[("qkv", ["matmul qkv"], "Q5_0", 253.2 * 896 / 1282.0),
                ("wo", ["matmul wo"], "Q5_0", 253.2 * 896 / 1282.0),
                ("gate/up", ["matmul gate/up"], "Q5_0", 253.2 * 4864 * 2 / 1282.0),
                ("down", ["matmul down"], "Q4_K+Q6_K", 104.6)],
        q5=253.2 * (896 + 896 + 4864 * 2) / 1282.0),
    "Llama-3.2-1B-Instruct-Q4_K_M.gguf": dict(
        name="Llama 1B", n_layer=16, n_head=32, hd=64,
        groups=[("qkv", ["matmul qkv"], "Q4_K", 85.4),
                ("wo", ["matmul wo"], "Q4_K", 67.1),
                ("gate/up", ["matmul gate/up"], "Q4_K", 268.4 * 2),
                ("down", ["matmul down"], "Q6_K", 268.4)],
        q5=0.0),
}
# Qwen's per-layer Q5_0 total is 10.551 M weights over 24 layers; the fractions
# above split it as qkv 0.896/10.551·? -- the grouped counts are written directly
# in the cells below from the architecture, not from the profile.
MODELS["Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"]["groups"] = [
    ("qkv", ["matmul qkv"], "Q5_0", 24 * (896 * 896 + 896 * 128 * 2) / 1e6),
    ("wo", ["matmul wo"], "Q5_0", 24 * 896 * 896 / 1e6),
    ("gate/up", ["matmul gate/up"], "Q5_0", 24 * 896 * 4864 * 2 / 1e6),
    ("down", ["matmul down"], "Q4_K+Q6_K", 24 * 4864 * 896 / 1e6),
]
MODELS["Llama-3.2-1B-Instruct-Q4_K_M.gguf"]["groups"] = [
    ("qkv", ["matmul qkv"], "Q4_K", 16 * (2048 * 2048 + 2048 * 512 * 2) / 1e6),
    ("wo", ["matmul wo"], "Q4_K", 16 * 2048 * 2048 / 1e6),
    ("gate/up", ["matmul gate/up"], "Q4_K", 16 * 2048 * 8192 * 2 / 1e6),
    ("down", ["matmul down"], "Q6_K", 16 * 8192 * 2048 / 1e6),
]

runs = [json.loads(l) for l in open(sys.argv[1] if len(sys.argv) > 1 else "account.jsonl")]
print("weight matmul by group (rows multiplied 511 / 4 095)")
for m, d in MODELS.items():
    for length in (512, 4096):
        for th in (1, 10):
            cells = [r for r in runs if r["model"] == m and r["len"] == length
                     and r["threads"] == th]
            if not cells:
                continue
            med = lambda f: st.median([f(c) for c in cells])
            rows = 511 if length == 512 else 4095
            cells_s = " ".join(
                f"{g[0]}:{g[2]} {g[3] * rows / 1e3 / (med(lambda c, k=g[1][0]: c['prof_ms'].get(k, 0)) / 1e9):.0f}"
                f"/{med(lambda c, k=g[1][0]: c['prof_ms'].get(k, 0)) / 1e9:.2f}s"
                for g in d["groups"])
            print(f"  {d['name']:<9} {f'{length}/{th}':>9} {med(lambda c: c['prefill_tok_s']):7.1f} tok/s"
                  f"  {cells_s}")
print()
print("attention and the rest (caller walls; scores/mix are worker sums)")
for m, d in MODELS.items():
    for length in (512, 4096):
        for th in (1, 10):
            cells = [r for r in runs if r["model"] == m and r["len"] == length
                     and r["threads"] == th]
            if not cells:
                continue
            med = lambda f: st.median([f(c) for c in cells]) / 1e9
            n = length
            gm = d["n_head"] * d["hd"] * (n + 1) / 2 * n * d["n_layer"] / 1e9  # scores = mix = gm
            wall = st.median([c["stats_ms"]["total"] for c in cells]) / 1e3
            a = med(lambda c: c["prof_ms"].get("attention", 0))
            sc = med(lambda c: c["prof_ms"].get("  attn scores", 0))
            mx = med(lambda c: c["prof_ms"].get("  attn mix", 0))
            rest = {k: med(lambda c, k=k: c["prof_ms"].get(k, 0)) for k in
                    ("norms", "rope+bias", "kv write", "residual", "swiglu",
                     "quantise", "barrier wait", "other")}
            print(f"  {d['name']:<9} {f'{length}/{th}':>9} wall {wall:6.2f}s  "
                  f"attn {a:6.2f}s ({100*a/wall:4.1f}%, {2*gm/a:4.0f} GMAC/s; "
                  f"scores {gm/sc:4.0f}, mix {gm/mx:4.0f})  "
                  + " ".join(f"{k.replace(' ', '')} {v:5.2f}s" for k, v in rest.items()))
