#!/usr/bin/env python3
"""Summarise the bert-family standings: speed, accuracy, profile and the gap.

Reads the raw JSON that `benchmarks/embed.py` writes under `<dir>/raw` and the
`cellprof` JSON lines under `<dir>/profile`, takes the median of the three
rounds, and prints the Markdown tables. Standard library only.

  python benchmarks/bert_report.py --dir benchmarks/results/bluefin-2026-11-11-bert-standings
"""

import argparse
import json
from pathlib import Path
import statistics

# The Peregrine row each model's gap analysis starts from.
PEREGRINE_ROW = {
    "minilm": "f16",
    "bge-small": "q8_0",
    "bge-m3": "q8_0",
}
# A competitor is "equal accuracy" when its cosine mean is at least this. The
# int8/int8 exports are at 0.97-0.98 and are compared by speed only.
EQUAL_ACCURACY = 0.999

CELL_ORDER = [
    "L16_b1_c1", "L16_b8_c1", "L16_b32_c1",
    "L128_b1_c1", "L128_b8_c1", "L128_b32_c1",
    "L512_b1_c1", "L512_b8_c1", "L512_b32_c1",
    "L2000_b1_c1", "L2000_b8_c1", "L2000_b32_c1",
    "mixed_b32_c1",
]
CELL_LABEL = {
    "L16_b1_c1": "L16 b1", "L16_b8_c1": "L16 b8", "L16_b32_c1": "L16 b32",
    "L128_b1_c1": "L128 b1", "L128_b8_c1": "L128 b8", "L128_b32_c1": "L128 b32",
    "L512_b1_c1": "L512 b1", "L512_b8_c1": "L512 b8", "L512_b32_c1": "L512 b32",
    "L2000_b1_c1": "L2000 b1", "L2000_b8_c1": "L2000 b8",
    "L2000_b32_c1": "L2000 b32", "mixed_b32_c1": "mixed b32",
}


def load_raw(directory):
    speed = {}
    latency = {}
    accuracy = {}
    for f in sorted((directory / "raw").glob("*-r*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        model = d["model"]["name"]
        engine = d["row"]["engine"]
        precision = d["row"]["precision"]
        key = (model, engine, precision)
        for cell, c in d.get("cells", {}).items():
            if "texts_per_s" in c:
                speed.setdefault(key, {}).setdefault(cell, []).append(c["texts_per_s"])
                latency.setdefault(key, {}).setdefault(cell, []).append(c["latency_s"]["p50"])
    for f in sorted((directory / "raw").glob("accuracy-*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        key = (d["model"]["name"], d["row"]["engine"], d["row"]["precision"])
        accuracy[key] = d["cosine"]
    return speed, latency, accuracy


def cells_of(speed, model):
    seen = {c for (m, _, _), cs in speed.items() if m == model for c in cs}
    return [c for c in CELL_ORDER if c in seen]


def row_label(key):
    return f"{key[1]} {key[2]}"


def speed_table(speed, latency, accuracy, model):
    cells = cells_of(speed, model)
    lines = ["| engine / file | " + " | ".join(CELL_LABEL[c] for c in cells) + " |",
             "|---|" + "---:|" * len(cells)]
    rows = sorted((k for k in speed if k[0] == model), key=lambda k: (k[1], k[2]))
    for key in rows:
        acc = accuracy.get(key, {}).get("mean")
        label = row_label(key)
        if acc is not None and acc < EQUAL_ACCURACY:
            label += f" ({acc:.3f})"
        vals = []
        for cell in cells:
            xs = speed[key].get(cell)
            vals.append(f"{statistics.median(xs):.2f}" if xs else "–")
        lines.append(f"| {label} | " + " | ".join(vals) + " |")
    return "\n".join(lines)


def latency_table(latency, model):
    cells = [c for c in ("L16_b1_c1", "L128_b1_c1", "L512_b1_c1", "L2000_b1_c1")
             if any(c in cs for (m, _, _), cs in latency.items() if m == model)]
    rows = sorted((k for k in latency if k[0] == model), key=lambda k: (k[1], k[2]))
    lines = ["| engine / file | " + " | ".join(CELL_LABEL[c] for c in cells) + " |",
             "|---|" + "---:|" * len(cells)]
    for key in rows:
        vals = []
        for cell in cells:
            xs = latency[key].get(cell)
            vals.append(f"{statistics.median(xs) * 1e3:.1f}" if xs else "–")
        lines.append(f"| {row_label(key)} | " + " | ".join(vals) + " |")
    return "\n".join(lines)


def accuracy_table(accuracy):
    lines = ["| model | engine / file | cosine mean | cosine min |",
             "|---|---|---:|---:|"]
    for key in sorted(accuracy, key=lambda k: (k[0], k[1], k[2])):
        a = accuracy[key]
        lines.append(f"| {key[0]} | {row_label(key)} | {a['mean']:.6f} | {a['min']:.6f} |")
    return "\n".join(lines)


def load_profile(directory):
    out = {}
    for f in sorted((directory / "profile").glob("*.jsonl")):
        model = f.stem.rsplit("-", 1)[0]
        rows = {}
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                rows[d["cell"]] = d
        out[model] = rows
    return out


def profile_table(profile, model, latency, precision):
    rows = profile.get(model, {})
    cells = [c for c in CELL_ORDER if c.replace("_c1", "") in rows]
    lines = ["| cell | HTTP ms (residual) | engine ms | tokenise ms | matmul | attention | element-wise | quantise | embedding | barrier wait | other |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for cell in cells:
        d = rows[cell.replace("_c1", "")]
        st = d["stats_ms"]
        ph = d.get("phases_ms", {})
        prof = d.get("prof_ms", {})
        total = st["total"] or 1.0
        q_ln = ph.get("quantise ln", 0.0)
        q_gelu = ph.get("quantise gelu", 0.0)
        q = q_ln + q_gelu
        ln = ph.get("layer norm", 0.0) - q_ln
        gelu = ph.get("gelu", 0.0) - q_gelu
        ew = max(0.0, ln + gelu)
        emb = ph.get("embed+norm", 0.0)
        pool = ph.get("pooling", 0.0)
        wait = prof.get("barrier wait", 0.0)
        other = max(0.0, st["total"] - st["matmul"] - st["attention"] - ew - q - emb - pool - wait)
        pct = lambda x: f"{100.0 * x / total:.1f}%"
        tok_ms = d["tokenise_ms_per_text"] * d["batch"]
        p50s = latency.get((model, "peregrine", precision), {}).get(cell)
        # The server p50 is the whole request; subtract the direct engine call and
        # tokenisation. Where the two runs differ by more than the framing (the
        # 2 000-token cells), the residual is negative and is reported as 0.
        http = (f"{max(0.0, statistics.median(p50s) * 1e3 - tok_ms - total):.2f}"
                if p50s else "–")
        lines.append(
            f"| {CELL_LABEL[cell]} | {http} | {total:.1f} | {tok_ms:.2f} | "
            f"{pct(st['matmul'])} | {pct(st['attention'])} | {pct(ew)} | {pct(q)} | "
            f"{pct(emb)} | {pct(wait)} | {pct(other)} |")
    return "\n".join(lines)


def gap_table(speed, accuracy, model):
    prec = PEREGRINE_ROW[model]
    mine = speed.get((model, "peregrine", prec), {})
    if not mine:
        return ""
    mine_acc = accuracy.get((model, "peregrine", prec), {}).get("mean", 0.0)
    rows = []
    for cell in cells_of(speed, model):
        rivals = []
        for key, cs in speed.items():
            if key[0] != model or key[1] == "peregrine":
                continue
            # llama.cpp's Q8_0 is a different kernel and lands 3e-4 below ours
            # on the same file; a half-thousandth is the same accuracy here.
            if accuracy.get(key, {}).get("mean", 0.0) < max(EQUAL_ACCURACY, mine_acc - 0.0005):
                continue
            if cell in cs:
                rivals.append((statistics.median(cs[cell]), key))
        if not rivals:
            continue
        best, key = max(rivals)
        p = statistics.median(mine[cell])
        rows.append((p / best, cell, p, best, key))
    rows.sort()
    lines = ["| cell | peregrine texts/s | best equal-accuracy texts/s | engine | ratio |",
             "|---|---:|---:|---|---:|"]
    for ratio, cell, p, best, key in rows:
        lines.append(f"| {CELL_LABEL[cell]} | {p:.2f} | {best:.2f} | {row_label(key)} | "
                     f"**{ratio:.2f}×** |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    args = ap.parse_args()
    speed, latency, accuracy = load_raw(args.dir)
    profile = load_profile(args.dir)
    for model in ("minilm", "bge-small", "bge-m3"):
        if not any(k[0] == model for k in speed):
            continue
        print(f"## {model}\n")
        print("### texts/s (median of three rounds)\n")
        print(speed_table(speed, latency, accuracy, model))
        print("\n### p50 latency, batch 1 (ms)\n")
        print(latency_table(latency, model))
        print("\n### Peregrine profile\n")
        print(profile_table(profile, model, latency, PEREGRINE_ROW[model]))
        print("\n### Peregrine against the best equal-accuracy engine\n")
        print(gap_table(speed, accuracy, model))
        print()
    print("## Accuracy\n")
    print(accuracy_table(accuracy))


if __name__ == "__main__":
    main()
