#!/usr/bin/env python3
"""The task-accuracy table of S5.4c, from the scored runs in accuracy/.

    python3 accuracy.py

Prints the score, the point difference on the same items and the exact McNemar
p-value for every model and task, with the fast mode off against on. The two
"off" runs are compared against each other as well: that is the suite's
run-to-run spread on this host, and it is zero because greedy replies are
deterministic.
"""
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
TASKS = ["gsm8k", "mmlu", "humaneval"]
MODELS = [
    ("qwen25-05b", "Qwen2.5 0.5B", ["off-r1", "off-r2", "on"]),
    ("llama32-1b", "Llama 3.2 1B", ["off-r1", "off-r2", "on"]),
    ("gemma3-1b", "Gemma 3 1B", ["off-r1", "on"]),
    ("qwen3-17b", "Qwen3 1.7B", ["off-r1", "off-r2", "on"]),
]


def load(run, task):
    path = os.path.join(HERE, "accuracy", run, f"{task}.scored.jsonl")
    return {r["id"]: r["correct"] for r in (json.loads(l) for l in open(path, encoding="utf-8"))}


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p from the discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def score(correct):
    return 100.0 * sum(correct.values()) / len(correct)


def main():
    print("| Model | Task | n | off (r1 / r2) | on | Δ on−off (points) | flips | McNemar p |")
    print("|---|---|---|---|---|---|---|---|")
    for key, name, configs in MODELS:
        runs = {c: f"{key}-{c}" for c in configs}
        for task in TASKS:
            off = load(runs[configs[0]], task)
            on = load(runs["on"], task)
            same_items = sorted(set(off) & set(on))
            a = {i: off[i] for i in same_items}
            b = {i: on[i] for i in same_items}
            diff = score(b) - score(a)
            b01 = sum(1 for i in same_items if not a[i] and b[i])
            b10 = sum(1 for i in same_items if a[i] and not b[i])
            spread = ""
            if "off-r2" in configs:
                r2 = load(runs["off-r2"], task)
                spread = f"{score(off):.1f} / {score(r2):.1f}"
            else:
                spread = f"{score(off):.1f} / —"
            print(f"| {name} | {task} | {len(same_items)} | {spread} | {score(on):.1f} | {diff:+.1f} "
                  f"| {b10} lost / {b01} gained | {mcnemar_exact(b10, b01):.2f} |")


if __name__ == "__main__":
    main()
