#!/usr/bin/env python3
"""S4.9w Part 1's cellprof control: the engine's own counters per pass, three arms.

Reads every `parts/part1-account/r*-*.jsonl` line, keys it by (model, cell, arm) from the
file name, and prints the median `bytes_acts`, `bytes_weights` and matmul bucket of the
three row-block arms.  `bytes_acts` is the engine's activation-traffic counter; the weight
bytes are counted once per matrix, so the block's weight re-read is read off the block
count (see the report).
"""
import glob
import json
import re
import statistics
from pathlib import Path

P = Path(__file__).resolve().parent / "parts" / "part1-account"
ARMS = ("old", "whole", "bound")
MODELS = ("nomic", "jina", "qwen3-embed-0.6b", "bge-small", "minilm", "nemotron")
CELLS = {
    "nomic": ["L128_b1", "L128_b32", "L512_b1", "L512_b8", "L2000_b1", "mixed_b32"],
    "jina": ["L128_b1", "L2000_b1"],
    "qwen3-embed-0.6b": ["L128_b1", "L512_b1", "L2000_b1"],
    "bge-small": ["L128_b32"],
    "minilm": ["L128_b32"],
    "nemotron": ["L512_b1"],
}

rows = {}
for f in glob.glob(str(P / "*.jsonl")):
    name = Path(f).name
    arm = next((a for a in ARMS if f"-{a}-" in f or f"-{a}." in f), None)
    if arm is None:
        continue
    model = next((m for m in MODELS if m in name), None)
    if "qwen3" in name:
        model = "qwen3-embed-0.6b"
    if "mixed" in name:
        model = "nomic"
    if model is None:
        continue
    for line in open(f):
        if not line.strip():
            continue
        d = json.loads(line)
        rows.setdefault((model, d["cell"], arm), []).append(d)

for model in MODELS:
    print(f"\n**{model}**\n")
    print("| cell | OLD acts MiB | WHOLE | BOUND | OLD matmul ms | WHOLE | BOUND | OLD texts/s | WHOLE | BOUND |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for cell in CELLS[model]:
        val = {}
        for a in ARMS:
            ds = rows.get((model, cell, a), [])
            if not ds:
                val[a] = None
                continue
            val[a] = (
                statistics.median(d["bytes_acts"] / 2**20 for d in ds),
                statistics.median(d["stats_ms"]["matmul"] for d in ds),
                statistics.median(d["texts_per_s"] for d in ds),
                len(ds),
            )
        if any(v is None for v in val.values()):
            print(f"| {cell} | — | — | — | — | — | — | — | — | — |")
            continue
        print(
            f"| {cell} | {val['old'][0]:.0f} | {val['whole'][0]:.0f} | {val['bound'][0]:.0f} "
            f"| {val['old'][1]:.1f} | {val['whole'][1]:.1f} | {val['bound'][1]:.1f} "
            f"| {val['old'][2]:.2f} | {val['whole'][2]:.2f} | {val['bound'][2]:.2f} |"
        )
