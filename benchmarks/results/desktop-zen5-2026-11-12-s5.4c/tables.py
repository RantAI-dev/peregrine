#!/usr/bin/env python3
"""Build the S5.4c tables from the committed raw files.

    python3 tables.py            # prints every table as Markdown

A, B: parse the `kld` reports in raw/.  D: parse the `peregrine bench` and
llama-bench JSONL in raw/speed/.  The numbers in README.md come from here.
"""
import collections
import glob
import json
import os
import re
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))

MODELS = [
    ("llama32-1b", "Llama 3.2 1B Instruct"),
    ("qwen25-05b", "Qwen2.5 0.5B Instruct"),
    ("smollm2-360m", "SmolLM2 360M Instruct"),
    ("gemma3-1b", "Gemma 3 1B IT"),
    ("qwen3-17b", "Qwen3 1.7B"),
]


def kld(path):
    """The statistics block of one kld report."""
    text = open(path, encoding="utf-8").read()
    out = {}
    for line in text.splitlines():
        m = re.match(r"\s*(Mean    KLD|Median  KLD|99\.0%   KLD|Maximum KLD|RMS Δp|Same top p"
                     r"|Mean PPL\(Q\)|Mean PPL\(base\))\s*:\s+([-0-9.]+)\s*(?:±\s*([-0-9.]+))?", line)
        if m:
            out[m.group(1).strip()] = float(m.group(2))
            if m.group(3):
                out[m.group(1).strip() + " ±"] = float(m.group(3))
    chunks = []
    for line in text.splitlines():
        m = re.match(r"\s*\d+\s+PPL\s+[-0-9.]+\s+KLD\s+[-0-9.]+\s+chunk KLD\s+([-0-9.]+)", line)
        if m:
            chunks.append(float(m.group(1)))
    return out, chunks


def fmt(v, n=6):
    return f"{v:.{n}f}"


def table_ab():
    rows = []
    for key, name in MODELS:
        a, chunks = kld(os.path.join(HERE, "raw", f"a-{key}-fast.txt"))
        b, _ = kld(os.path.join(HERE, "raw", f"b-{key}-exact.txt"))
        c, _ = kld(os.path.join(HERE, "raw", f"b-{key}-fast.txt"))
        share = (c["Mean    KLD"] - b["Mean    KLD"]) / b["Mean    KLD"] * 100.0
        rows.append((name, a, chunks, b, c, share))
    print("| Model (Q4_K_M) | A: fast vs exact | median / 99 % | RMS Δp | same top | B: exact vs F16 | B: fast vs F16 | share |")
    print("|---|---|---|---|---|---|---|---|")
    for name, a, _, b, c, share in rows:
        print(f"| {name} | {fmt(a['Mean    KLD'])} ± {a['Mean    KLD ±']:.6f} | {fmt(a['Median  KLD'])} / {fmt(a['99.0%   KLD'])} "
              f"| {a['RMS Δp']:.3f} % | {a['Same top p']:.3f} % | {fmt(b['Mean    KLD'])} | {fmt(c['Mean    KLD'])} "
              f"| {share:+.2f} % |")
    print()
    print("Per-chunk KLD (A):")
    for name, a, chunks, _, _, _ in rows:
        print(f"| {name} | {len(chunks)} | {min(chunks):.5f} | {statistics.median(chunks):.5f} "
              f"| {max(chunks):.5f} |")


def speed_cell(files, drop_zero=True, load_max=None):
    by = collections.defaultdict(dict)
    loads = []
    for f in files:
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            if drop_zero and r["tok_s"] <= 0:
                continue
            if load_max is not None and r["load1"] > load_max:
                continue
            by[r["engine"]][r["round"]] = r["tok_s"]
            loads.append(r["load1"])
    out = {}
    for e, v in by.items():
        vals = sorted(v.values())
        out[e] = (statistics.median(vals), min(vals), max(vals), len(vals))
    return out, (min(loads) if loads else 0.0, max(loads) if loads else 0.0)


CELLS = [
    ("Llama 3.2 1B", 512, 12, ["llama32-1b-p512-t12-v2b.jsonl"], None),
    ("Llama 3.2 1B", 512, 1, ["llama32-1b-p512-t1-v2.jsonl"], 1.5),
    ("Llama 3.2 1B", 4096, 12, ["llama32-1b-p4096-t12-v2b.jsonl"], None),
    ("Llama 3.2 1B", 4096, 1, ["llama32-1b-p4096-t1-v2.jsonl"], None),
    ("Qwen3 1.7B", 512, 12, ["qwen3-17b-p512-t12-v2.jsonl"], None),
    ("Qwen3 1.7B", 512, 1, ["qwen3-17b-p512-t1-v2.jsonl"], None),
    ("Qwen3 1.7B", 4096, 12, ["qwen3-17b-p4096-t12-v2.jsonl", "qwen3-17b-p4096-t12-v2b.jsonl"], None),
    ("Qwen3 1.7B", 4096, 1, ["qwen3-17b-p4096-t1-v2.jsonl"], None),
]

ENGINES = ["pg-exact", "pg-fast", "ik-default", "ik-rtr", "llama"]


def table_speed():
    print("| Model | prompt | threads | Peregrine exact | Peregrine fast | ik (defaults) | ik (-rtr) | llama.cpp | fast ÷ exact | fast ÷ ik | fast ÷ ik -rtr | fast ÷ llama | load |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name, p, t, files, load_max in CELLS:
        paths = [os.path.join(HERE, "raw", "speed", f) for f in files]
        data, (lmin, lmax) = speed_cell(paths, load_max=load_max)
        med = {e: data[e][0] for e in ENGINES if e in data}
        rng = {e: (data[e][1], data[e][2], data[e][3]) for e in ENGINES if e in data}
        print(f"| {name} | {p} | {t} | {med['pg-exact']:.1f} | {med['pg-fast']:.1f} | {med['ik-default']:.1f} "
              f"| {med['ik-rtr']:.1f} | {med['llama']:.1f} | {med['pg-fast']/med['pg-exact']:.2f}× "
              f"| {med['pg-fast']/med['ik-default']:.2f}× | {med['pg-fast']/med['ik-rtr']:.2f}× "
              f"| {med['pg-fast']/med['llama']:.2f}× | {lmin:.1f}-{lmax:.1f} |")
    print()
    print("Medians. Spreads (min-max, n):")
    for name, p, t, files, load_max in CELLS:
        paths = [os.path.join(HERE, "raw", "speed", f) for f in files]
        data, _ = speed_cell(paths, load_max=load_max)
        cells = "; ".join(f"{e} {data[e][1]:.0f}-{data[e][2]:.0f} (n={data[e][3]})" for e in ENGINES if e in data)
        print(f"- {name} {p} tokens, {t} threads: {cells}")


if __name__ == "__main__":
    table_ab()
    print()
    table_speed()
