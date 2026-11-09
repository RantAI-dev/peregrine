#!/usr/bin/env python3
"""Rebuild the before/after medians of this directory from its JSON run sets."""
import glob
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS = ("Llama-3.2-1B", "Qwen2.5-0.5B")
METRICS = ("decode_tok_s", "prefill_tok_s", "aggregate_tok_s",
           "decode_aggregate_tok_s", "ttft_s")


def medians(phase, sub, model, ctx, users):
    samples = {}
    for path in sorted(glob.glob(str(HERE / phase / sub / "*.json"))):
        run = json.load(open(path))
        if model not in run["model"]["filename"]:
            continue
        engine = run["engine"]["name"]
        result = run["results"][str(ctx)]
        key = "single_user" if users == 1 else f"multi_user_{users}"
        for metric in METRICS:
            for value in result.get(key, {}).get(metric, []):
                samples.setdefault((engine, metric), []).append(value)
    for (engine, metric), values in samples.items():
        med = statistics.median(values)
        spread = (max(values) - min(values)) / 2
        print(f"{phase:6} {sub:14} {model:14} {ctx:4} u{users:<2} "
              f"{engine:10} {metric:24} {med:8.2f} +- {spread:6.2f} (n={len(values)})")


if __name__ == "__main__":
    for phase in ("before", "after"):
        for ctx, users in ((512, 1), (512, 4), (512, 10), (4096, 10)):
            for model in MODELS:
                medians(phase, "512" if ctx == 512 else "4096", model, ctx, users)
    for ctx, users in ((512, 1), (512, 4), (512, 10)):
        medians("before", "512-threads24", "Llama-3.2-1B", ctx, users)
        medians("after", "512-threads24", "Llama-3.2-1B", ctx, users)
    for model in MODELS:
        medians("before", "512-fa-off", model, 512, 10)
    sys.exit(0)
