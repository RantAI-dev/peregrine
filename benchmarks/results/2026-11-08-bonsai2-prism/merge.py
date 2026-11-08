#!/usr/bin/env python3
"""Combine the per-invocation run sets of the Bonsai 2 / PrismML benchmark.

The host caps each command at ten minutes, so the 3 interleaved rounds of a
(model, engine) pair were measured in several invocations: one for the 512-token
phases (1 and 10 users), one for the 4096-token phase (1 user), and one per MTP
configuration. This script concatenates the raw sample arrays of those runs into
the same one-manifest-per-engine shape the harness writes for a single
`--runs 3` invocation, and recomputes the summary with the harness's own
median/spread rule.

    python3 merge.py            # reads raw/, writes the manifests beside it
"""

import datetime
import json
from pathlib import Path
import statistics

RAW = Path(__file__).resolve().parent / "raw"
OUT = Path(__file__).resolve().parent


def stats(samples):
    if not samples:
        return {"median": 0.0, "spread": 0.0, "min": 0.0, "max": 0.0}
    s = [float(x) for x in samples]
    return {
        "median": round(statistics.median(s), 2),
        "spread": round((max(s) - min(s)) / 2.0, 2),
        "min": round(min(s), 2),
        "max": round(max(s), 2),
    }


def config_label(engine_name, engine_args):
    """Name the speculative-decoding configuration of one manifest."""
    args = engine_args.get(engine_name, [])
    if engine_name == "peregrine":
        return "mtp-off" if "--opt MTP=0" in " ".join(args) else "mtp-on"
    return "mtp-on" if "--spec-type draft-mtp" in " ".join(args) else "mtp-off"


def merge(group):
    runs = sorted(group, key=lambda r: r["timestamp"])
    base = json.loads(json.dumps(runs[0]))  # deep copy
    resolved = base["model"]["path"]
    marker = "/models/"
    if marker in resolved:
        # The harness resolves the checkout's models/ symlink; record the stable
        # checkout-relative path instead of the host path it resolves to.
        base["model"]["path"] = "models/" + resolved.split(marker, 1)[1]
    engine_name = base["engine"]["name"]
    label = config_label(engine_name, base["settings"].get("engine_args", {}))
    model_and_engine = (base["model"]["filename"], engine_name)

    contexts = sorted({int(c) for r in runs for c in r["results"] if r["results"][c]["single_user"]["decode_tok_s"] or r["results"][c]["multi_user_10"]["aggregate_tok_s"]})
    base["settings"]["contexts"] = contexts
    base["settings"]["runs"] = len(runs)
    base["timestamp"] = runs[0]["timestamp"]
    base["merged_from"] = [Path(r["_source"]).name for r in runs]
    base["results"] = {}
    base["summary"] = {}

    bandwidth = base["machine"]["bandwidth_ceiling_bytes_per_s"]
    bytes_per_token = base["model"].get("bytes_per_token_telemetry", 0.0)

    for ctx in contexts:
        ctx_k = str(ctx)
        merged = {
            "single_user": {"decode_tok_s": [], "prefill_tok_s": [], "ttft_s": [], "prompt_tokens": []},
            "multi_user_4": {"aggregate_tok_s": [], "decode_aggregate_tok_s": [], "ttft_s": [], "prompt_tokens": []},
            "multi_user_10": {"aggregate_tok_s": [], "decode_aggregate_tok_s": [], "ttft_s": [], "prompt_tokens": []},
        }
        for r in runs:
            data = r["results"].get(ctx_k)
            if not data:
                continue
            for phase in merged:
                for key in merged[phase]:
                    merged[phase][key].extend(data.get(phase, {}).get(key, []))

        su_dec = stats(merged["single_user"]["decode_tok_s"])
        su_pref = stats(merged["single_user"]["prefill_tok_s"])
        su_ttft = stats(merged["single_user"]["ttft_s"])
        su_pts = stats(merged["single_user"]["prompt_tokens"])
        m4_agg = stats(merged["multi_user_4"]["aggregate_tok_s"])
        m4_ttft = stats(merged["multi_user_4"]["ttft_s"])
        m4_pts = stats(merged["multi_user_4"]["prompt_tokens"])
        m10_agg = stats(merged["multi_user_10"]["aggregate_tok_s"])
        m10_ttft = stats(merged["multi_user_10"]["ttft_s"])
        m10_pts = stats(merged["multi_user_10"]["prompt_tokens"])

        actual_prompt_tokens = su_pts["median"] if su_pts["median"] > 0 else ctx
        frac = 0.0
        if bandwidth > 0 and bytes_per_token > 0:
            frac = (bytes_per_token * su_dec["median"]) / bandwidth

        base["results"][ctx_k] = merged
        base["summary"][ctx_k] = {
            "context": ctx,
            "actual_prompt_tokens": actual_prompt_tokens,
            "single_user": {
                "decode_tok_s": su_dec,
                "prefill_tok_s": su_pref,
                "ttft_s": su_ttft,
                "prompt_tokens": su_pts,
                "bytes_per_token": bytes_per_token,
                "fraction_of_ceiling": round(frac, 4),
                "percent_of_ceiling": round(frac * 100.0, 2),
                "raw_samples": merged["single_user"],
            },
            "multi_user_4": {
                "aggregate_tok_s": m4_agg,
                "decode_aggregate_tok_s": stats(merged["multi_user_4"]["decode_aggregate_tok_s"]),
                "ttft_s": m4_ttft,
                "prompt_tokens": m4_pts,
                "raw_samples": merged["multi_user_4"],
            },
            "multi_user_10": {
                "aggregate_tok_s": m10_agg,
                "decode_aggregate_tok_s": stats(merged["multi_user_10"]["decode_aggregate_tok_s"]),
                "first_token_time_s": m10_ttft,
                "prompt_tokens": m10_pts,
                "raw_samples": merged["multi_user_10"],
            },
        }

    stem = Path(model_and_engine[0]).stem
    is_mtp_model = "MTP" in stem
    suffix = f"-{label}" if is_mtp_model else ""
    stamp = datetime.datetime.fromisoformat(runs[0]["timestamp"]).strftime("%Y-%m-%d-%H%M%S")
    out = OUT / f"{stamp}-{engine_name}-{stem}{suffix}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(base, f, indent=2)
    print(f"{out.name}: {len(runs)} runs, contexts {contexts}")
    return model_and_engine + (label,)


def main():
    groups = {}
    for path in sorted(RAW.glob("*.json")):
        with open(path, encoding="utf-8") as f:
            run = json.load(f)
        run["_source"] = str(path)
        engine_name = run["engine"]["name"]
        label = config_label(engine_name, run["settings"].get("engine_args", {}))
        key = (run["model"]["filename"], engine_name, label)
        groups.setdefault(key, []).append(run)

    for key in sorted(groups):
        merge(groups[key])


if __name__ == "__main__":
    main()
