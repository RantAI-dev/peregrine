#!/usr/bin/env python3
"""Summarise a standings run: Peregrine, llama.cpp and the ratio per cell.

Reads the JSON that `benchmarks/embed.py` writes (three interleaved rounds),
takes the median of the per-round `texts_per_s`, and prints one Markdown table
per model/precision. It also prints the decision cells from `benchmarks/decide.py`
and the stage profile from `benchmarks/../crates/peregrine-bench/examples/cellprof.rs`.
Standard library only.

  python benchmarks/standings_report.py --dir benchmarks/results/bluefin-...-s4.9-standings
"""

import argparse
import json
from pathlib import Path
import statistics


def load_cells(path):
    d = json.loads(path.read_text(encoding="utf-8"))
    return d.get("cells", {})


def collect(directory):
    """(model, precision) -> engine -> cell -> [per-round cells]."""
    out = {}
    for p in sorted(directory.glob("*-r*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        model = d["model"]["name"]
        row = d["row"]
        key = (model, row["precision"])
        out.setdefault(key, {}).setdefault(row["engine"], {})
        for cell, c in d.get("cells", {}).items():
            out[key][row["engine"]].setdefault(cell, []).append(c)
    return out


def cell_order(cell):
    if cell.startswith("mixed"):
        return (99999, 0, 0)
    length, rest = cell[1:].split("_b")
    batch = rest.split("_")[0]
    return (int(length), int(batch), 0)


def median_field(rounds, name, default=None):
    vals = [r.get(name) for r in rounds if isinstance(r.get(name), (int, float))]
    return statistics.median(vals) if vals else default


def table_for(cells):
    rows = sorted(cells.get("peregrine", {}) | cells.get("llama_cpp", {}), key=cell_order)
    lines = ["| cell | peregrine texts/s | llama.cpp texts/s | ratio | peregrine p50 ms | llama.cpp p50 ms |",
             "|---|---:|---:|---:|---:|---:|"]
    for cell in rows:
        p = cells.get("peregrine", {}).get(cell, [])
        l = cells.get("llama_cpp", {}).get(cell, [])
        ps = median_field(p, "texts_per_s")
        ls = median_field(l, "texts_per_s")
        pp = median_field([r.get("latency_s", {}) for r in p], "p50")
        lp = median_field([r.get("latency_s", {}) for r in l], "p50")
        if ps is None or ls is None:
            lines.append(f"| {cell} | {ps or 'skip'} | {ls or 'skip'} | – | – | – |")
            continue
        lines.append(
            f"| {cell} | {ps:.2f} | {ls:.2f} | **{ps / ls:.2f}×** | "
            f"{pp * 1e3:.1f} | {lp * 1e3:.1f} |"
        )
    return "\n".join(lines)


def profile_tables(directory):
    files = sorted((directory / "profile").glob("*profile.json"))
    out = []
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        name = Path(d["model"]).stem
        lines = [f"### {name}, {d['threads']} threads", "",
                 "| cell | request ms | tokenise ms | engine ms | matmul | attention | quantise | barrier wait | other |",
                 "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for cell, c in sorted(d["cells"].items(), key=lambda kv: cell_order(kv[0])):
            st = c["stats_ms"]
            prof = c.get("prof_ms", {})
            total = st["total"] or 1.0
            matmul = prof.get("matmul qkv", 0) + prof.get("matmul wo", 0) + \
                prof.get("matmul gate/up", 0) + prof.get("matmul down", 0)
            if matmul == 0:
                # The bidirectional encoder's step statistics split matmul and
                # attention; the prof buckets it does fill (norms, quantise,
                # barrier wait) come out of `other`.
                matmul = st["matmul"]
                attn = st["attention"]
                quant = prof.get("quantise")
                wait = prof.get("barrier wait")
                other = max(0.0, st["other"] - (quant or 0.0) - (wait or 0.0))
            else:
                attn = prof.get("attention", 0)
                quant = prof.get("quantise", 0)
                wait = prof.get("barrier wait", 0)
                other = max(0.0, st["other"] - quant - wait)
            def pct(x):
                return "–" if x is None else f"{100.0 * x / total:.1f}%"
            lines.append(
                f"| {cell} | {c['wall_s'] * 1e3:.1f} | "
                f"{c['tokenise_ms_per_text'] * c['batch']:.2f} | "
                f"{st['total']:.1f} | {pct(matmul)} | {pct(attn)} | "
                f"{pct(quant)} | {pct(wait)} | {pct(other)} |"
            )
        out.append("\n".join(lines))
    return "\n\n".join(out)


def decision_tables(directory):
    out = []
    for engine in ("peregrine", "llama_cpp"):
        files = sorted((directory / "decide").glob(f"decide-{engine}-r*.json"))
        if not files:
            continue
        cells = {}
        for f in files:
            d = json.loads(f.read_text(encoding="utf-8"))
            for cell, c in d["cells"].items():
                cells.setdefault(cell, []).append(c["questions_per_s"])
        lines = [f"### {engine}, 4 threads", "",
                 "| state tokens | questions | questions/s |", "|---:|---:|---:|"]
        for cell in sorted(cells, key=lambda c: (int(c.split("_")[0][1:]), int(c.split("_q")[1]))):
            state, q = cell[1:].split("_q")
            lines.append(f"| {state} | {q} | {statistics.median(cells[cell]):.2f} |")
        out.append("\n".join(lines))
    return "\n\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    args = ap.parse_args()
    data = collect(args.dir)
    print("# Embedding standings\n")
    for (model, precision), engines in sorted(data.items()):
        print(f"## {model} {precision}\n")
        print(table_for(engines))
        print()
    print("## Stage profile (Peregrine)\n")
    print(profile_tables(args.dir))
    print("\n## Decisions (/v1/systemone)\n")
    print(decision_tables(args.dir))


if __name__ == "__main__":
    main()
