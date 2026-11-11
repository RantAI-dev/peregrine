#!/usr/bin/env python3
"""Merge embedding benchmark rounds into medians and Markdown tables (S4.2).

`embed.py` writes one JSON per (model, engine, precision, round).  This script
takes the median across rounds for every cell, keeps the min/max spread, joins
the accuracy records, and emits:

  summary.json   every cell, medians and spreads, machine and engine records
  tables.md      the tables used in the results README

Standard library only.

  python benchmarks/embed_report.py --results-dir benchmarks/results/bluefin-embed-2026-11-10
"""

import argparse
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Dict, List, Optional, Tuple

CELL_ORDER = [
    (16, 1), (16, 8), (16, 32), (16, 128),
    (128, 1), (128, 8), (128, 32), (128, 128),
    (512, 1), (512, 8), (512, 32), (512, 128),
]


def cell_key(L: int, b: int, c: int = 1) -> str:
    return f"L{L}_b{b}_c{c}"


def median(vals: List[float]) -> float:
    return statistics.median(vals) if vals else 0.0


def spread(vals: List[float]) -> Tuple[float, float]:
    return (min(vals), max(vals)) if vals else (0.0, 0.0)


def load_runs(results_dir: Path) -> Dict[Tuple[str, str, str], List[Dict[str, Any]]]:
    runs: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for f in sorted(results_dir.glob("*.json")):
        name = f.name
        if name.startswith("accuracy-") or name in ("summary.json",):
            continue
        if "-r" not in name:
            continue
        try:
            rec = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if rec.get("benchmark") != "peregrine-s4.2-embed":
            continue
        key = (rec["model"]["name"], rec["row"]["engine"], rec["row"]["precision"])
        runs.setdefault(key, []).append(rec)
    return runs


def load_accuracy(results_dir: Path) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    out: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for f in sorted(results_dir.glob("accuracy-*.json")):
        try:
            rec = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        key = (rec["model"]["name"], rec["row"]["engine"], rec["row"]["precision"])
        out[key] = rec
    return out


def merge_cell(records: List[Dict[str, Any]], key: str) -> Optional[Dict[str, Any]]:
    cells = [r["cells"][key] for r in records if key in r["cells"]]
    if not cells:
        return None
    if all("skipped" in c for c in cells):
        return dict(cells[0])
    live = [c for c in cells if "skipped" not in c]
    tps = [c["texts_per_s"] for c in live]
    toks = [c["tokens_per_s"] for c in live]
    p50 = [c["latency_s"]["p50"] for c in live]
    p99 = [c["latency_s"]["p99"] for c in live]
    return {
        "texts_per_s": round(median(tps), 3),
        "texts_per_s_spread": [round(x, 3) for x in spread(tps)],
        "tokens_per_s": round(median(toks), 1),
        "tokens_per_s_spread": [round(x, 1) for x in spread(toks)],
        "p50_s": round(median(p50), 6),
        "p50_s_spread": [round(x, 6) for x in spread(p50)],
        "p99_s": round(median(p99), 6),
        "p99_s_spread": [round(x, 6) for x in spread(p99)],
        "rounds": len(live),
        "requests": [c["requests"] for c in live],
        "tokens_source": live[0]["tokens_source"],
    }


def fmt(x: float, nd: int = 1) -> str:
    if x is None:
        return "—"
    if x == 0:
        return "0"
    if abs(x) >= 1000:
        return f"{x:,.0f}"
    if abs(x) >= 100:
        return f"{x:.0f}"
    return f"{x:.{nd}f}"


def engine_label(engine: str, precision: str, notes: str = "") -> str:
    name = {
        "llama_cpp": "llama.cpp",
        "tei": "TEI",
        "fastembed": "fastembed",
        "openvino": "OpenVINO",
        "peregrine": "Peregrine",
    }.get(engine, engine)
    return f"{name} {precision}"


def build_summary(runs, accs) -> Dict[str, Any]:
    models: Dict[str, Any] = {}
    for (model, engine, precision), records in sorted(runs.items()):
        records = sorted(records, key=lambda r: r.get("round", 0))
        cells = {}
        for L, b in CELL_ORDER:
            c = merge_cell(records, cell_key(L, b))
            if c:
                cells[cell_key(L, b)] = c
        for L, _ in CELL_ORDER:
            for c in (10,):
                cell = merge_cell(records, cell_key(L, 1, c))
                if cell:
                    cells[cell_key(L, 1, c)] = cell
        entry = {
            "engine": engine,
            "precision": precision,
            "rounds": len(records),
            "row": records[0]["row"],
            "model": records[0]["model"],
            "engine_info": records[0].get("engine", {}),
            "start": records[0].get("start", {}),
            "settings": records[0].get("settings", {}),
            "host_before": [r.get("host_before") for r in records],
            "host_after": [r.get("host_after") for r in records],
            "machine": records[0].get("machine", {}),
            "cells": cells,
            "accuracy": None,
        }
        a = accs.get((model, engine, precision))
        if a:
            entry["accuracy"] = {
                "cosine_mean": a["cosine"]["mean"],
                "cosine_min": a["cosine"]["min"],
                "cosine_p1": a["cosine"].get("p1"),
                "norm_mean": a["norm"]["mean"],
                "dim": a["dim"],
                "n": a["n"],
                "reference": a["reference"],
            }
        models.setdefault(model, []).append(entry)
    return models


def table_throughput(models: Dict[str, Any], field: str, nd: int) -> str:
    out = []
    for model, rows in models.items():
        header = "| engine · precision | cos | " + " | ".join(
            f"L{L} b{b}" for L, b in CELL_ORDER
        ) + " |"
        sep = "|" + "---|" * (len(CELL_ORDER) + 2)
        out.append(f"\n**{model}** — {field.replace('_', ' ')}\n")
        out.append(header)
        out.append(sep)
        for r in rows:
            label = engine_label(r["engine"], r["precision"])
            cos = r["accuracy"]["cosine_mean"] if r["accuracy"] else None
            cells = []
            for L, b in CELL_ORDER:
                c = r["cells"].get(cell_key(L, b))
                if not c:
                    cells.append("·")
                elif "skipped" in c:
                    cells.append("skip")
                else:
                    cells.append(fmt(c[field], nd))
            out.append(f"| {label} | " +
                       (f"{cos:.4f}" if cos is not None else "—") + " | " +
                       " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def table_latency(models: Dict[str, Any]) -> str:
    out = [
        "\n| model | engine · precision | p50 1×16 (ms) | p99 1×16 (ms) | p50 10×1×16 (ms) | p99 10×1×16 (ms) |",
        "|---|---|---|---|---|---|",
    ]
    for model, rows in models.items():
        for r in rows:
            c1 = r["cells"].get(cell_key(16, 1, 1))
            c10 = r["cells"].get(cell_key(16, 1, 10))
            if not c1:
                continue
            def ms(c, k):
                return f"{c[k] * 1000:.1f}" if c and "skipped" not in c else "—"
            out.append(
                f"| {model} | {engine_label(r['engine'], r['precision'])} | "
                f"{ms(c1, 'p50_s')} | {ms(c1, 'p99_s')} | "
                f"{ms(c10, 'p50_s')} | {ms(c10, 'p99_s')} |"
            )
    return "\n".join(out) + "\n"


def best_per_row(models: Dict[str, Any]) -> str:
    out = [
        "\n| model | cell | best competitor | texts/s | tokens/s | worst-cell note |",
        "|---|---|---|---|---|---|",
    ]
    for model, rows in models.items():
        for L, b in CELL_ORDER:
            key = cell_key(L, b)
            best = None
            for r in rows:
                c = r["cells"].get(key)
                if not c or "skipped" in c:
                    continue
                if r["engine"] == "peregrine":
                    continue
                if best is None or c["texts_per_s"] > best[1]["texts_per_s"]:
                    best = (r, c)
            if best is None:
                continue
            r, c = best
            out.append(
                f"| {model} | L{L} b{b} | {engine_label(r['engine'], r['precision'])} | "
                f"{fmt(c['texts_per_s'])} | {fmt(c['tokens_per_s'], 0)} | "
                f"{c['rounds']} rounds |"
            )
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="benchmarks/results/bluefin-embed-2026-11-10")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    results_dir = Path(args.results_dir)
    runs = load_runs(results_dir)
    if not runs:
        raise SystemExit(f"no round JSON in {results_dir}")
    accs = load_accuracy(results_dir)
    models = build_summary(runs, accs)
    summary = {"results_dir": str(results_dir), "models": models}
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")

    md = ["# S4.2 embedding baselines — generated tables\n"]
    md.append(table_throughput(models, "texts_per_s", 2))
    md.append(table_throughput(models, "tokens_per_s", 0))
    md.append("\n## single-text latency (batch 1)\n")
    md.append(table_latency(models))
    md.append("\n## fastest competitor per cell\n")
    md.append(best_per_row(models))
    out = Path(args.out) if args.out else results_dir / "tables.md"
    out.write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {results_dir / 'summary.json'} and {out}")


if __name__ == "__main__":
    main()
