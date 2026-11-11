#!/usr/bin/env python3
"""Rebuild the medians in this folder's README from the raw run sets.

The 512 and 4096 rows were measured in several invocations (the host caps a
command at ten minutes), so one engine/model/context can have its samples split
over more than one JSON file. This script concatenates the per-phase raw sample
arrays across files before taking the median and half-range, exactly as
``benchmarks/run.py`` does within one invocation.

Usage: python3 summarize.py [--ratios] [directory]
"""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def stats(samples):
    if not samples:
        return None
    s = sorted(samples)
    return {
        "median": statistics.median(s),
        "spread": (s[-1] - s[0]) / 2.0,
        "min": s[0],
        "max": s[-1],
        "n": len(s),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratios", action="store_true")
    ap.add_argument("directory", nargs="?", default=str(Path(__file__).parent))
    args = ap.parse_args()

    rows = defaultdict(lambda: defaultdict(list))
    meta = {}
    for path in sorted(Path(args.directory).glob("*.json")):
        try:
            d = json.load(open(path))
        except Exception:
            continue
        if not isinstance(d, dict) or "engine" not in d:
            continue
        eng = d["engine"]["name"]
        model = d["model"]["filename"]
        key = (model, d["model"]["sha256"][:8], eng)
        meta.setdefault(key, d["engine"])
        for ctx, r in d.get("results", {}).items():
            for phase in ("single_user", "multi_user_4", "multi_user_10"):
                for metric, values in r.get(phase, {}).items():
                    if metric in ("ttft_s", "decode_tok_s", "prefill_tok_s",
                                  "aggregate_tok_s"):
                        rows[(model, ctx, eng)][(phase, metric)].extend(values)

    for (model, ctx, eng), metrics in sorted(rows.items()):
        if not metrics.get(("single_user", "decode_tok_s")):
            continue
        def m(phase, metric):
            st = stats(metrics.get((phase, metric), []))
            if not st:
                return "—"
            return f"{st['median']:.2f} ± {st['spread']:.2f} (n={st['n']})"
        print(f"\n{model}  ctx={ctx}  {eng}")
        print(f"  decode 1u     {m('single_user','decode_tok_s')}")
        print(f"  prompt 1u     {m('single_user','prefill_tok_s')}")
        print(f"  agg 4u        {m('multi_user_4','aggregate_tok_s')}")
        print(f"  agg 10u       {m('multi_user_10','aggregate_tok_s')}")
        print(f"  ttft 10u (s)  {m('multi_user_10','ttft_s')}")

    if args.ratios:
        print("\n\nRatios (Peregrine / other), same model+context:")
        for (model, ctx, eng) in sorted(rows):
            if eng == "peregrine":
                continue
            p = rows.get((model, ctx, "peregrine"))
            o = rows.get((model, ctx, eng))
            if not p or not o:
                continue
            out = [f"{model} ctx={ctx} vs {eng}"]
            for phase, metric, label in (
                ("single_user", "decode_tok_s", "decode"),
                ("single_user", "prefill_tok_s", "prompt"),
                ("multi_user_4", "aggregate_tok_s", "4u"),
                ("multi_user_10", "aggregate_tok_s", "10u"),
                ("multi_user_10", "ttft_s", "ttft10"),
            ):
                ps = stats(p.get((phase, metric), []))
                os_ = stats(o.get((phase, metric), []))
                if ps and os_ and os_["median"]:
                    out.append(f"{label} {ps['median']/os_['median']:.2f}x")
            print("  " + "  ".join(out))


if __name__ == "__main__":
    main()
