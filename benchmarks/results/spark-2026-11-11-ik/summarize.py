#!/usr/bin/env python3
"""Rebuild the tables in README.md from the raw run sets.

Every JSON in this directory is one engine's arm of one harness invocation.
Invocations are split by context and user count to keep each command inside
the ten-minute cap; this script concatenates the per-phase sample arrays and
reports the median and half-range, exactly as run.py's report does.
"""
import glob
import json
import os
import statistics
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))

MODELS = [
    ("Llama-3.2-1B-Instruct-Q4_K_M", "Llama 3.2 1B Instruct Q4_K_M"),
    ("Qwen2.5-0.5B-Instruct-Q4_K_M", "Qwen 2.5 0.5B Instruct Q4_K_M"),
    ("SmolLM2-360M-Instruct.Q4_K_M", "SmolLM2 360M Instruct Q4_K_M"),
    ("google_gemma-3-1b-it-Q4_K_M", "gemma-3-1b-it Q4_K_M"),
]
ENGINES = [("peregrine", "Peregrine"), ("llama_cpp", "llama.cpp"), ("ik_llama_cpp", "ik_llama.cpp")]
PHASES = {"single_user": "single", "multi_user_4": "4 users", "multi_user_10": "10 users"}


def median_half_range(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None, 0
    m = statistics.median(xs)
    return m, (max(xs) - min(xs)) / 2.0, len(xs)


def main():
    # key: (model, engine, ctx, phase) -> {metric: [samples]}
    data = defaultdict(lambda: defaultdict(list))
    for path in sorted(glob.glob(os.path.join(HERE, "*.json"))):
        with open(path, encoding="utf-8") as fh:
            rec = json.load(fh)
        model = rec["model"]["filename"].removesuffix(".gguf")
        engine = {"peregrine": "peregrine", "llama.cpp": "llama_cpp",
                  "ik_llama.cpp": "ik_llama_cpp"}.get(rec["engine"]["name"],
                                                      rec["engine"]["name"])
        for ctx, phases in rec["results"].items():
            for phase, metrics in phases.items():
                for metric, samples in metrics.items():
                    data[(model, engine, int(ctx), phase)][metric].extend(samples)

    out = {}
    for model, label in MODELS:
        out[label] = {}
        for ctx in (512, 4096):
            out[label][ctx] = {}
            for phase in ("single_user", "multi_user_4", "multi_user_10"):
                out[label][ctx][phase] = {}
                for eng, _ in ENGINES:
                    m = data.get((model, eng, ctx, phase), {})
                    row = {}
                    for metric in ("decode_tok_s", "prefill_tok_s", "aggregate_tok_s", "ttft_s"):
                        med, half, n = median_half_range(m.get(metric, []))
                        if med is not None:
                            row[metric] = {"median": med, "half_range": half, "n": n}
                    if row:
                        out[label][ctx][phase][eng] = row
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
