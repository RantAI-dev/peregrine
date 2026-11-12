#!/usr/bin/env python3
"""Tables for the S4.9l proof run: speed, accuracy, cache and stage profile.

Reads the raw JSON `benchmarks/embed.py` wrote under `<dir>/raw` (three rounds),
the cache-on rows under `<dir>/raw-cache`, the `cellprof` lines under
`<dir>/profile`, and the ignored real-file tests' log under `<dir>/tests`.
Prints the Markdown tables; standard library only.

  python benchmarks/proof_report.py --dir benchmarks/results/bluefin-2026-11-12-s4.9l-proof

The ratio is always Peregrine / the best other engine at equal accuracy. A
competitor is "equal accuracy" when its measured cosine mean is within 0.0005
of Peregrine's row for that model (the same rule as `bert_report.py`); a row
with no cosine record is included and named in the README when that matters.
"""

import argparse
import json
from pathlib import Path
import statistics

# The Peregrine row each model's ratio starts from: the fastest row that is at
# the fp32/F16 accuracy class the model is served at.
PEREGRINE_ROW = {
    "minilm": "f16",
    "bge-small": "q8_0",
    "bge-m3": "q8_0",
    "embeddinggemma": "q8_0",
    "qwen3-embed-0.6b": "q8_0",
    "nemotron-3.5-embed-1b": "q8_0",
}
MODEL_ORDER = list(PEREGRINE_ROW)
MODEL_LABEL = {
    "minilm": "all-MiniLM-L6-v2",
    "bge-small": "bge-small-en-v1.5",
    "bge-m3": "bge-m3",
    "embeddinggemma": "EmbeddingGemma 300M",
    "qwen3-embed-0.6b": "Qwen3-Embedding 0.6B",
    "nemotron-3.5-embed-1b": "Nemotron 3.5 Embed 1B",
}
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


def load_raw(directory, sub="raw"):
    speed, latency, accuracy = {}, {}, {}
    base = directory / sub
    for f in sorted(base.glob("*-r[0-9].json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        key = (d["model"]["name"], d["row"]["engine"], d["row"]["precision"])
        for cell, c in d.get("cells", {}).items():
            if "texts_per_s" in c:
                speed.setdefault(key, {}).setdefault(cell, []).append(c["texts_per_s"])
                latency.setdefault(key, {}).setdefault(cell, []).append(c["latency_s"]["p50"])
    for f in sorted(base.glob("accuracy-*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        if "model" not in d:
            continue  # accuracy-real-file.json, read by load_real_file
        key = (d["model"]["name"], d["row"]["engine"], d["row"]["precision"])
        a = dict(d["cosine"])
        a["backend"] = (d.get("reference") or {}).get("backend") or "fp32"
        a["count"] = (d.get("reference") or {}).get("count")
        accuracy[key] = a
    return speed, latency, accuracy


def load_real_file(directory):
    path = directory / "raw" / "accuracy-real-file.json"
    if not path.exists():
        return {}
    return {(r["model"], r["engine"], r["precision"]): r
            for r in json.loads(path.read_text(encoding="utf-8"))["records"]}


def load_profile(directory):
    out = {}
    for f in sorted((directory / "profile").glob("*.jsonl")):
        rows = {}
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                rows[d["cell"]] = d
        out[f.stem] = rows
    return out


def cells_of(speed, model):
    seen = {c for (m, _, _), cs in speed.items() if m == model for c in cs}
    return [c for c in CELL_ORDER if c in seen]


def row_label(key):
    return f"{key[1]} {key[2]}"


def median(xs):
    return statistics.median(xs) if xs else None


def speed_table(speed, latency, accuracy, real, model):
    cells = cells_of(speed, model)
    lines = ["| engine / file | " + " | ".join(CELL_LABEL[c] for c in cells) + " |",
             "|---|" + "---:|" * len(cells)]
    rows = sorted((k for k in speed if k[0] == model), key=lambda k: (k[1], k[2]))
    for key in rows:
        acc = accuracy.get(key, {}).get("mean")
        if acc is None:
            acc = real.get(key, {}).get("cosine_mean")
        label = row_label(key)
        if acc is not None and acc < EQUAL_ACCURACY:
            label += f" ({acc:.3f})"
        vals = [f"{median(speed[key][c]):.2f}" if c in speed[key] else "–" for c in cells]
        lines.append(f"| {label} | " + " | ".join(vals) + " |")
    return "\n".join(lines)


def latency_table(latency, model):
    cells = [c for c in ("L16_b1_c1", "L128_b1_c1", "L512_b1_c1", "L2000_b1_c1")
             if any(c in cs for (m, _, _), cs in latency.items() if m == model)]
    rows = sorted((k for k in latency if k[0] == model), key=lambda k: (k[1], k[2]))
    lines = ["| engine / file | " + " | ".join(CELL_LABEL[c] for c in cells) + " |",
             "|---|" + "---:|" * len(cells)]
    for key in rows:
        vals = [f"{median(latency[key][c]) * 1e3:.1f}" if c in latency[key] else "–"
                for c in cells]
        lines.append(f"| {row_label(key)} | " + " | ".join(vals) + " |")
    return "\n".join(lines)


def profile_table(profile, model, precision, latency):
    rows = profile.get(f"{model}-{precision}", {})
    cells = [c for c in CELL_ORDER if c.replace("_c1", "") in rows]
    lines = ["| cell | request ms | tokenise ms | matmul | attention | other | "
             "barrier wait | steps |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for cell in cells:
        d = rows[cell.replace("_c1", "")]
        st = d["stats_ms"]
        total = st["total"] or 1.0
        prof = d.get("prof_ms", {})
        wait = prof.get("barrier wait", 0.0)
        other = max(0.0, total - st["matmul"] - st["attention"])
        pct = lambda x: f"{100.0 * x / total:.1f}%"
        p50 = latency.get((model, "peregrine", precision), {}).get(cell)
        request = f"{median(p50) * 1e3:.1f}" if p50 else "–"
        tok_ms = d["tokenise_ms_per_text"] * d["batch"]
        lines.append(
            f"| {CELL_LABEL[cell]} | {request} | {tok_ms:.2f} | {pct(st['matmul'])} | "
            f"{pct(st['attention'])} | {pct(other)} | {pct(wait)} | {st['barriers']} |")
    return "\n".join(lines)


def gap_table(speed, accuracy, real, model):
    prec = PEREGRINE_ROW[model]
    mine = speed.get((model, "peregrine", prec), {})
    if not mine:
        return ""
    mine_acc = accuracy.get((model, "peregrine", prec), {}).get("mean")
    if mine_acc is None:
        mine_acc = real.get((model, "peregrine", prec), {}).get("cosine_mean")
    rows = []
    for cell in cells_of(speed, model):
        rivals = []
        for key, cs in speed.items():
            if key[0] != model or key[1] == "peregrine" or cell not in cs:
                continue
            acc = accuracy.get(key, {}).get("mean")
            if acc is None:
                acc = real.get(key, {}).get("cosine_mean")
            if acc is not None and mine_acc is not None:
                # Equal accuracy: within a half-thousandth of Peregrine's row, and
                # never below the absolute bar that excludes the int8 exports.
                if acc < min(EQUAL_ACCURACY, mine_acc - 0.0005):
                    continue
            rivals.append((median(cs[cell]), key))
        if not rivals:
            continue
        best, key = max(rivals)
        p = median(mine[cell])
        rows.append((p / best, cell, p, best, key))
    rows.sort()
    lines = ["| cell | peregrine texts/s | best equal-accuracy texts/s | engine | ratio |",
             "|---|---:|---:|---|---:|"]
    for ratio, cell, p, best, key in rows:
        lines.append(f"| {CELL_LABEL[cell]} | {p:.2f} | {best:.2f} | {row_label(key)} | "
                     f"**{ratio:.2f}×** |")
    return "\n".join(lines)


def cache_table(speed, cache, model):
    prec = PEREGRINE_ROW[model]
    off = speed.get((model, "peregrine", prec), {})
    on = cache.get((model, "peregrine", prec), {})
    if not off or not on:
        return ""
    cells = [c for c in CELL_ORDER if c in off and c in on]
    lines = ["| cell | cache off texts/s | cache on, repeated texts/s | speed-up |",
             "|---|---:|---:|---:|"]
    for cell in cells:
        a, b = median(off[cell]), median(on[cell])
        lines.append(f"| {CELL_LABEL[cell]} | {a:.2f} | {b:.2f} | {b / a:.1f}× |")
    return "\n".join(lines)


def model_accuracy_table(accuracy, real, model):
    lines = ["| engine / file | cosine mean | cosine min | reference |",
             "|---|---:|---:|---|"]
    keys = [k for k in accuracy if k[0] == model]
    for key in sorted(keys, key=lambda k: (k[1], k[2])):
        a = accuracy[key]
        lines.append(f"| {row_label(key)} | {a['mean']:.6f} | {a['min']:.6f} | "
                     f"{a.get('backend', 'fp32')}, {a.get('count')} texts |")
    for key in sorted((k for k in real if k[0] == model), key=lambda k: (k[1], k[2])):
        r = real[key]
        if r.get("cosine_mean") is not None:
            lines.append(f"| {row_label(key)} | {r['cosine_mean']:.6f} | "
                         f"{r['cosine_min']:.6f} | real-file test, {r.get('reference', '')} |")
    return "\n".join(lines)


def accuracy_table(accuracy, real):
    lines = ["| model | engine / file | cosine mean | cosine min | reference |",
             "|---|---|---:|---:|---|"]
    for key in sorted(accuracy, key=lambda k: (MODEL_ORDER.index(k[0]), k[1], k[2])):
        a = accuracy[key]
        lines.append(f"| {key[0]} | {row_label(key)} | {a['mean']:.6f} | {a['min']:.6f} | "
                     f"{a.get('backend', 'fp32')}, {a.get('count')} texts |")
    for key in sorted(real, key=lambda k: (MODEL_ORDER.index(k[0]), k[1], k[2])):
        r = real[key]
        ref = r.get("reference", "real-file fixtures")
        if r.get("cosine_mean") is None:
            continue
        lines.append(f"| {key[0]} | {row_label(key)} | {r['cosine_mean']:.6f} | "
                     f"{r['cosine_min']:.6f} | {ref} |")
    return "\n".join(lines)


def real_file_table(real):
    lines = ["| model | check | cosine min | cosine mean |", "|---|---|---:|---:|"]
    for key in sorted(real, key=lambda k: (MODEL_ORDER.index(k[0]), k[1], k[2])):
        r = real[key]
        for check in r.get("checks", []):
            if check.get("cosine_mean") is None:
                continue
            lines.append(f"| {key[0]} | {key[2]}: {check['name']} | {check['cosine_min']:.7f} | "
                         f"{check['cosine_mean']:.7f} |")
    return "\n".join(lines)


def load_decisions(directory):
    out = {}
    for f in sorted((directory / "decide").glob("decide-*-r[0-9].json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        for cell, c in d.get("cells", {}).items():
            out.setdefault((d["engine"], cell), []).append(c["questions_per_s"])
    return out


def decisions_table(decisions):
    cells = sorted({cell for (_, cell) in decisions},
                   key=lambda c: (int(c.split("_")[0][1:]), int(c.split("_")[1][1:])))
    lines = ["| state tokens | questions | peregrine q/s | llama.cpp q/s | ratio |",
             "|---:|---:|---:|---:|---:|"]
    for cell in cells:
        p = median(decisions.get(("peregrine", cell), []))
        l = median(decisions.get(("llama_cpp", cell), []))
        if p is None or l is None:
            continue
        state, q = cell[1:].split("_q")
        lines.append(f"| {state} | {q} | {p:.2f} | {l:.2f} | **{p / l:.2f}×** |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    args = ap.parse_args()
    speed, latency, accuracy = load_raw(args.dir)
    cache, _, _ = load_raw(args.dir, sub="raw-cache")
    real = load_real_file(args.dir)
    profile = load_profile(args.dir)
    for model in MODEL_ORDER:
        if not any(k[0] == model for k in speed):
            continue
        print(f"## {MODEL_LABEL[model]}\n")
        print("### texts/s (median of three rounds)\n")
        print(speed_table(speed, latency, accuracy, real, model))
        print("\n### p50 latency, batch 1 (ms)\n")
        print(latency_table(latency, model))
        print("\n### Peregrine against the best equal-accuracy engine\n")
        print(gap_table(speed, accuracy, real, model))
        print("\n### Accuracy\n")
        print(model_accuracy_table(accuracy, real, model))
        print("\n### Peregrine stage profile\n")
        print(profile_table(profile, model, PEREGRINE_ROW[model], latency))
        print("\n### Peregrine, cache on (repeated texts)\n")
        print(cache_table(speed, cache, model) or "_not measured_")
        print()
    print("## Decisions, /v1/systemone\n")
    print(decisions_table(load_decisions(args.dir)))
    print()
    print("## Accuracy, harness\n")
    print(accuracy_table(accuracy, {}))
    print("\n## Accuracy, real-file tests\n")
    print(real_file_table(real))


if __name__ == "__main__":
    main()
