#!/usr/bin/env python3
"""The account of part A: where a fast prompt pass spends its time, and what each stage
runs at, from `promptbench`'s own counters (`PEREGRINE_PROF2=full`).

Reads `raw/account/*.jsonl` (one JSON line per cell) and prints, per (model, length,
threads, mode):

  * the pass wall (`prefill_s`), prompt tok/s and end-to-end GMAC/s;
  * the caller's StepStats buckets and the `prof` slots, as shares of the pass wall;
  * the weight matmul split by matrix group (qkv / wo / gate-up / down / head) and by
    weight format (Q4_K / Q6_K), with the **in-situ GMAC/s** of each, from the model's own
    tensor table;
  * attention's three stages, the caller's serial steps, and the barrier wait.

The matrices' MAC counts come from the GGUF tensor table (`gguf_macs.py`), so the in-situ
rate is the engine's, not a microbenchmark's.
"""

import json
import os
import statistics as st
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

MODELS = {
    "llama:Llama-3.2-1B-Instruct-Q4_K_M.gguf": (
        "Llama 3.2 1B",
        os.path.expanduser("~/models-ab/Llama-3.2-1B-Instruct-Q4_K_M.gguf"),
    ),
    "qwen3:Qwen3-1.7B-Q4_K_M.gguf": (
        "Qwen3 1.7B",
        "/var/tmp/pg/s54c/models/Qwen3-1.7B-Q4_K_M.gguf",
    ),
}

GROUPS = [("qkv", "matmul qkv"), ("wo", "matmul wo"), ("gate/up", "matmul gate/up"),
          ("down", "matmul down")]
FORMATS = [("Q4_K", "  fmt Q4_K"), ("Q6_K", "  fmt Q6_K")]


def med(v):
    return st.median(v)


def main():
    files = sys.argv[1:] or [os.path.join(ROOT, "raw/account/rounds-a.jsonl")]
    raw = json.loads(
        subprocess.check_output(
            [sys.executable, os.path.join(HERE, "gguf_macs.py")]
            + [p for _, p in MODELS.values()]
        )
    )
    for path in files:
        rows = [json.loads(line) for line in open(path) if line.strip()]
        cells = {}
        for r in rows:
            cells.setdefault((r["model"], r["mode"], r["len"], r["threads"]), []).append(r)
        print(f"=== {os.path.basename(path)} ===")
        for key in sorted(cells):
            rows_c = cells[key]
            _, mode, ln, th = key
            name, mpath = MODELS[rows_c[0]["model"]]
            m = raw[rows_c[0]["model"]]
            cfg = m["cfg"]
            prompt_rows = med([r["prefill_rows"] for r in rows_c])
            wall = med([r["prefill_s"] for r in rows_c])
            toks = med([r["prefill_tok_s"] for r in rows_c])
            total = med([r["macs"] for r in rows_c])
            share = {}
            names = set()
            for r in rows_c:
                names.update(r["prof_ms"].keys())
            for n in names:
                if n.startswith("dec "):
                    continue
                share[n] = med([r["prof_ms"].get(n, 0.0) / 1e9 / r["prefill_s"] for r in rows_c])
            st_ = med([r["stats_ms"]["matmul"] for r in rows_c]) / 1e3
            head_s = med([r["stats_ms"]["head"] for r in rows_c]) / 1e3
            attn_s = med([r["stats_ms"]["attention"] for r in rows_c]) / 1e3
            other_s = med([r["stats_ms"]["other"] for r in rows_c]) / 1e3
            width = rows_c[0]["width"]
            load = (min(r["load1"] for r in rows_c), max(r["load1"] for r in rows_c))
            print(
                f"\n--- {name}  prompt {ln}  threads {th}  {mode}  {toks:.1f} tok/s  "
                f"wall {wall:.3f} s  end to end {total / wall / 1e9:.1f} GMAC/s  "
                f"{prompt_rows:.0f} rows  load {load[0]:.2f}-{load[1]:.2f} ---"
            )
            print(
                f"  StepStats: matmul {st_ / wall * 100:5.1f} %  head {head_s / wall * 100:5.1f} %"
                f"  attention {attn_s / wall * 100:5.1f} %  other {other_s / wall * 100:5.1f} %"
                f"  barriers {med([r['stats_ms']['barriers'] for r in rows_c]):.0f}"
            )
            print("  weight matmul by group (per prompt row):")
            for label, slot in GROUPS:
                g = share.get(slot, 0.0)
                macs = m["group_macs"].get(label, 0) * prompt_rows
                rate = macs / (g * wall) / 1e9 if g else 0.0
                print(
                    f"    {label:>8}  {g * 100:5.1f} %  {g * wall:7.3f} s  "
                    f"{macs / 1e9:6.1f} GMAC  {rate:5.1f} GMAC/s in situ"
                )
            hm = cfg["head_macs"]
            print(
                f"    {'head':>8}  {head_s / wall * 100:5.1f} %  {head_s:7.3f} s  "
                f"{hm / 1e9:6.3f} GMAC  {hm / head_s / 1e9 if head_s else 0:5.1f} GMAC/s"
                f" in situ (1 row of {cfg['vocab']})"
            )
            print("  weight matmul by format:")
            for label, slot in FORMATS:
                g = share.get(slot, 0.0)
                macs = m["format_macs"].get(label, 0) * prompt_rows
                rate = macs / (g * wall) / 1e9 if g else 0.0
                print(
                    f"    {label:>5}  {g * 100:5.1f} %  {g * wall:7.3f} s  "
                    f"{macs / 1e9:6.1f} GMAC  {rate:5.1f} GMAC/s in situ  "
                    f"({m['format_bytes'].get(label, 0) / 1e6:.0f} MB)"
                )
            n = ln
            sc = [l * cfg["n_head"] * cfg["head_dim"] * (n * (n + 1) // 2) for l in (cfg["n_layer"],)]
            sc = sc[0]
            mix = sc
            print(
                f"  attention {attn_s / wall * 100:5.1f} %:"
                f"  scores {share.get('  attn scores', 0) * 100:4.1f} %"
                f"  softmax {share.get('  attn softmax', 0) * 100:4.1f} %"
                f"  mix {share.get('  attn mix', 0) * 100:4.1f} %"
                f"  (worker sums / {width} workers)"
            )
            # The subtimers are worker-side sums, i.e. core-seconds, so the per-core
            # rate is the MACs over that sum.
            sr = sc / (share.get("  attn scores", 1e-9) * wall) / 1e9
            mr = mix / (share.get("  attn mix", 1e-9) * wall) / 1e9
            print(
                f"    scores {sc / 1e9:7.2f} GMAC  {sr:5.1f} GMAC/s/core;  "
                f"mix {mix / 1e9:7.2f} GMAC  {mr:5.1f} GMAC/s/core"
            )
            print(
                f"  caller: quantise {share.get('quantise', 0) * 100:4.1f} %"
                f"  swiglu {share.get('swiglu', 0) * 100:4.1f} %"
                f"  residual {share.get('residual', 0) * 100:4.1f} %"
                f"  rope {share.get('rope+bias', 0) * 100:4.1f} %"
                f"  kv write {share.get('kv write', 0) * 100:4.1f} %"
                f"  norms {share.get('norms', 0) * 100:4.1f} %"
                f"  barrier wait {share.get('barrier wait', 0) * 100:4.1f} %"
                f"  other {share.get('other', 0) * 100:4.1f} %"
            )


if __name__ == "__main__":
    main()
