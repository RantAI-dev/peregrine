#!/usr/bin/env python3
"""Medians across the three interleaved rounds, for every cell in this folder.

`embed_report.py` prints the fixed 16/128/512 grid; this run also has the
2000-token cells and the mixed batch, so this script reports every cell the
raw JSONs hold. A cell is the median of its per-round `texts_per_s`,
`tokens_per_s` and batch-1 p50; a cell skipped by the harness is reported once.
"""
import glob
import json
import os
import statistics
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
CELLS = [
    "L16_b1_c1", "L16_b32_c1", "L128_b1_c1", "L128_b32_c1",
    "L512_b1_c1", "L512_b32_c1", "L2000_b1_c1", "L2000_b32_c1",
    "mixed_b32_c1",
]


def main():
    rows = defaultdict(lambda: defaultdict(list))
    skipped = {}
    for path in sorted(glob.glob(os.path.join(HERE, "*-r[123].json"))):
        with open(path, encoding="utf-8") as fh:
            rec = json.load(fh)
        key = (rec["model"]["name"], rec["row"]["engine"], rec["row"]["precision"])
        for cell, value in rec["cells"].items():
            if "skipped" in value:
                skipped[(key, cell)] = value["skipped"]
                continue
            rows[(key, cell)]["texts_per_s"].append(value["texts_per_s"])
            rows[(key, cell)]["tokens_per_s"].append(value["tokens_per_s"])
            rows[(key, cell)]["p50_ms"].append(value["latency_s"]["p50"] * 1000.0)

    out = defaultdict(dict)
    for (key, cell), vals in rows.items():
        model, engine, precision = key
        out[model].setdefault(f"{engine}:{precision}", {})[cell] = {
            "texts_per_s": statistics.median(vals["texts_per_s"]),
            "tokens_per_s": statistics.median(vals["tokens_per_s"]),
            "p50_ms": statistics.median(vals["p50_ms"]),
            "rounds": len(vals["texts_per_s"]),
        }
    for (key, cell), reason in skipped.items():
        model, engine, precision = key
        out[model].setdefault(f"{engine}:{precision}", {})[cell] = {"skipped": reason}
    print(json.dumps(out, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
