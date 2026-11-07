#!/usr/bin/env python3
"""Peregrine Benchmark Harness: report generator.

Converts benchmark JSON run manifests into Markdown tables adhering to the format
defined in docs/targets.md (Section 2 & 4: the 'four numbers', medians, spreads,
and fraction-of-ceiling).
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional


def format_stat(stat_dict: Dict[str, Any], unit: str = "") -> str:
    """Format median and spread as 'median ± spread unit'."""
    med = stat_dict.get("median", 0.0)
    spread = stat_dict.get("spread", 0.0)
    u_str = f" {unit}" if unit else ""
    if spread > 0.0:
        return f"{med:.1f} ± {spread:.1f}{u_str}"
    return f"{med:.1f}{u_str}"


def format_ttft(stat_dict: Dict[str, Any]) -> str:
    """Format time-to-first-token in seconds (e.g. 0.025 ± 0.002 s)."""
    med = stat_dict.get("median", 0.0)
    spread = stat_dict.get("spread", 0.0)
    if spread > 0.0:
        return f"{med:.3f} ± {spread:.3f} s"
    return f"{med:.3f} s"


def _render_model_tables(model_runs: List[Dict[str, Any]], heading_level: int = 2) -> List[str]:
    """Render Results and Head-to-Head tables for a specific model's runs."""
    lines: List[str] = []
    h2 = "#" * heading_level
    h3 = "#" * (heading_level + 1)

    # Collect all contexts across runs for this model
    all_contexts: set = set()
    for r in model_runs:
        summary = r.get("summary", {})
        for c in summary.keys():
            all_contexts.add(int(c))
    sorted_contexts = sorted(list(all_contexts))

    # Verify actual prompt token counts across all runs and fail loudly if >5% off target
    for r in model_runs:
        eng_name = r.get("engine", {}).get("name", "unknown")
        for ctx in sorted_contexts:
            ctx_k = str(ctx)
            s = r.get("summary", {}).get(ctx_k)
            if not s:
                continue
            target_ctx = int(ctx)
            act_tokens = s.get("actual_prompt_tokens")
            if act_tokens is None and "prompt_tokens" in s.get("single_user", {}):
                act_tokens = s["single_user"]["prompt_tokens"].get("median")
            if act_tokens is None:
                raise ValueError(
                    f"Engine '{eng_name}' context {ctx} is missing recorded actual prompt token count in summary!"
                )
            off_pct = abs(act_tokens - target_ctx) / float(target_ctx)
            if off_pct > 0.05:
                raise ValueError(
                    f"Engine '{eng_name}' context {ctx} actual prompt tokens ({act_tokens:.1f}) "
                    f"is {off_pct * 100.0:.2f}% off target ({target_ctx}) > 5% allowed threshold!"
                )
            for phase_name in ["single_user", "multi_user_4", "multi_user_10"]:
                raw_pts = s.get(phase_name, {}).get("raw_samples", {}).get("prompt_tokens", [])
                for pt in raw_pts:
                    off = abs(pt - target_ctx) / float(target_ctx)
                    if off > 0.05:
                        raise ValueError(
                            f"Engine '{eng_name}' context {ctx} {phase_name} request prompt tokens ({pt}) "
                            f"is {off * 100.0:.2f}% off target ({target_ctx}) > 5% allowed threshold!"
                        )

    lines.append(f"{h2} Benchmark Results: The Four Numbers")
    lines.append("")
    lines.append(
        "Per `docs/targets.md`: single-user decode tok/s, prompt tok/s, total tok/s at 4 and 10 users, "
        "and first-token time at 10 users; medians reported with spread (± half-range)."
    )
    lines.append("")
    lines.append("| Context | Actual Tokens | Engine | Single decode tok/s | % of ceiling | Prompt tok/s | 4 users tok/s | 10 users tok/s | 10 users TTFT |")
    lines.append("|---|---|---|---|---|---|---|---|---|")

    for ctx in sorted_contexts:
        ctx_k = str(ctx)
        for r in model_runs:
            eng_name = r.get("engine", {}).get("name", "unknown")
            s = r.get("summary", {}).get(ctx_k)
            if not s:
                continue
            su = s.get("single_user", {})
            m4 = s.get("multi_user_4", {})
            m10 = s.get("multi_user_10", {})

            act_tokens = s.get("actual_prompt_tokens", ctx)
            act_str = f"{act_tokens:.0f}" if isinstance(act_tokens, (int, float)) else str(act_tokens)

            dec_str = format_stat(su.get("decode_tok_s", {}))
            pct_ceil = f"{su.get('percent_of_ceiling', 0.0):.1f}%" if su.get("percent_of_ceiling", 0.0) > 0 else "—"
            pref_str = format_stat(su.get("prefill_tok_s", {}))
            m4_str = format_stat(m4.get("aggregate_tok_s", {}))
            m10_str = format_stat(m10.get("aggregate_tok_s", {}))
            ttft_str = format_ttft(m10.get("first_token_time_s", {}))

            lines.append(
                f"| {ctx} | {act_str} | {eng_name} | {dec_str} | {pct_ceil} | {pref_str} | {m4_str} | {m10_str} | {ttft_str} |"
            )
    lines.append("")

    # Head-to-Head Comparison Table if multiple engines exist
    engines_in_runs = {r.get("engine", {}).get("name") for r in model_runs}
    if len(engines_in_runs) > 1 and "peregrine" in engines_in_runs:
        lines.append(f"{h2} Head-to-Head Comparison")
        lines.append("")
        lines.append("| Context | Metric | Peregrine | Competitor | Speedup | Peregrine % Ceiling |")
        lines.append("|---|---|---|---|---|---|")

        p_run = next(r for r in model_runs if r.get("engine", {}).get("name") == "peregrine")
        comp_runs = [r for r in model_runs if r.get("engine", {}).get("name") != "peregrine"]

        for comp_run in comp_runs:
            comp_name = comp_run.get("engine", {}).get("name", "competitor")
            for ctx in sorted_contexts:
                ctx_k = str(ctx)
                p_s = p_run.get("summary", {}).get(ctx_k)
                c_s = comp_run.get("summary", {}).get(ctx_k)
                if not p_s or not c_s:
                    continue

                # 1. Single decode
                p_dec = p_s["single_user"]["decode_tok_s"]["median"]
                c_dec = c_s["single_user"]["decode_tok_s"]["median"]
                ratio_dec = f"{p_dec / c_dec:.2f}x" if c_dec > 0 else "—"
                pct_ceil = f"{p_s['single_user'].get('percent_of_ceiling', 0.0):.1f}%"
                lines.append(f"| {ctx} | Single Decode | {format_stat(p_s['single_user']['decode_tok_s'])} | {format_stat(c_s['single_user']['decode_tok_s'])} ({comp_name}) | {ratio_dec} | {pct_ceil} |")

                # 2. Prefill
                p_pref = p_s["single_user"]["prefill_tok_s"]["median"]
                c_pref = c_s["single_user"]["prefill_tok_s"]["median"]
                ratio_pref = f"{p_pref / c_pref:.2f}x" if c_pref > 0 else "—"
                lines.append(f"| {ctx} | Prompt (Prefill) | {format_stat(p_s['single_user']['prefill_tok_s'])} | {format_stat(c_s['single_user']['prefill_tok_s'])} | {ratio_pref} | — |")

                # 3. 4 Users Aggregate
                p_m4 = p_s["multi_user_4"]["aggregate_tok_s"]["median"]
                c_m4 = c_s["multi_user_4"]["aggregate_tok_s"]["median"]
                ratio_m4 = f"{p_m4 / c_m4:.2f}x" if c_m4 > 0 else "—"
                lines.append(f"| {ctx} | 4 Users Throughput | {format_stat(p_s['multi_user_4']['aggregate_tok_s'])} | {format_stat(c_s['multi_user_4']['aggregate_tok_s'])} | {ratio_m4} | — |")

                # 4. 10 Users Aggregate
                p_m10 = p_s["multi_user_10"]["aggregate_tok_s"]["median"]
                c_m10 = c_s["multi_user_10"]["aggregate_tok_s"]["median"]
                ratio_m10 = f"{p_m10 / c_m10:.2f}x" if c_m10 > 0 else "—"
                lines.append(f"| {ctx} | 10 Users Throughput | {format_stat(p_s['multi_user_10']['aggregate_tok_s'])} | {format_stat(c_s['multi_user_10']['aggregate_tok_s'])} | {ratio_m10} | — |")

                # 5. 10 Users TTFT (lower is better)
                p_ttft = p_s["multi_user_10"]["first_token_time_s"]["median"]
                c_ttft = c_s["multi_user_10"]["first_token_time_s"]["median"]
                ratio_ttft = f"{c_ttft / p_ttft:.2f}x faster" if p_ttft > 0 and c_ttft > 0 else "—"
                lines.append(f"| {ctx} | 10 Users TTFT | {format_ttft(p_s['multi_user_10']['first_token_time_s'])} | {format_ttft(c_s['multi_user_10']['first_token_time_s'])} | {ratio_ttft} | — |")
        lines.append("")

    return lines


def generate_report(runs: List[Dict[str, Any]]) -> str:
    """Generate Markdown report from a list of run set dictionaries."""
    lines: List[str] = []

    if not runs:
        return "# Benchmark Report\n\nNo run data found."

    # Group runs by model filename
    models_dict: Dict[str, List[Dict[str, Any]]] = {}
    for r in runs:
        m_filename = r.get("model", {}).get("filename", "unknown")
        models_dict.setdefault(m_filename, []).append(r)

    # Use first run for system metadata
    sample_run = runs[0]
    mach = sample_run.get("machine", {})

    is_synth = any(
        r.get("sample_type") == "synthetic_smoke"
        or "synthetic" in str(r.get("model", {}).get("filename", "")).lower()
        or "smoke" in str(r.get("model", {}).get("path", "")).lower()
        for r in runs
    )
    if is_synth:
        lines.append("# Benchmark Report (Synthetic Smoke Sample)")
        lines.append("")
        lines.append("> **Note**: This is a synthetic smoke sample generated using a tiny synthetic GGUF model for CI/development verification, not real model weights.")
    else:
        lines.append("# Benchmark Report")
    lines.append("")
    lines.append(f"**Date**: {sample_run.get('timestamp', 'unknown')}")

    if len(models_dict) == 1:
        single_m = sample_run.get("model", {})
        lines.append(f"**Model**: `{single_m.get('filename', 'unknown')}` (SHA-256: `{single_m.get('sha256', '')[:16]}...`)")
    lines.append("")

    # Machine Record Table
    lines.append("## Machine Record")
    lines.append("")
    lines.append("| Property | Value |")
    lines.append("|---|---|")
    lines.append(f"| Host / CPU | {mach.get('cpu_model', 'unknown')} |")
    features_str = ", ".join(mach.get("cpu_features", [])) or "standard"
    lines.append(f"| CPU Features | {features_str} |")
    lines.append(f"| Memory Bandwidth Ceiling | **{mach.get('bandwidth_ceiling_gb_s', 0.0):.2f} GB/s** (Peregrine telemetry probe) |")
    lines.append(f"| Compute Ceiling | {mach.get('gflops_ceiling', 0.0):.1f} GFLOP/s (Ridge: {mach.get('ridge_point', 0.0):.1f}) |")
    lines.append(f"| Kernel / OS | {mach.get('kernel', 'unknown')} ({mach.get('os', 'unknown')}) |")
    lines.append(f"| CPU Governor | {mach.get('governor', 'unknown')} |")
    lines.append(f"| Benchmark Threads | {mach.get('threads_tested', 'unknown')} |")
    lines.append("")

    # Engine Builds Table (deduplicated)
    lines.append("## Engines Tested")
    lines.append("")
    lines.append("| Engine | Version / Commit | Build Details |")
    lines.append("|---|---|---|")
    seen_engines = set()
    for r in runs:
        eng = r.get("engine", {})
        eng_key = (eng.get("name"), eng.get("version"), eng.get("commit"))
        if eng_key in seen_engines:
            continue
        seen_engines.add(eng_key)
        lines.append(f"| {eng.get('name', 'unknown')} | `{eng.get('version', '')}` (`{eng.get('commit', '')[:8]}`) | {eng.get('build_command', eng.get('build_type', 'default'))} |")
    lines.append("")

    if len(models_dict) == 1:
        # Single model: render tables directly
        lines.extend(_render_model_tables(runs, heading_level=2))
    else:
        # Multiple models: render a dedicated section per model
        for m_filename, m_runs in models_dict.items():
            m_info = m_runs[0].get("model", {})
            sha = m_info.get("sha256", "")[:16]
            lines.append(f"## Model: `{m_filename}` (SHA-256: `{sha}...`)")
            lines.append("")
            lines.extend(_render_model_tables(m_runs, heading_level=3))

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Format Peregrine benchmark JSON run manifests into Markdown tables.",
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="JSON file(s) or directory containing run set JSONs.",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output Markdown file (default: stdout).",
    )
    return parser.parse_args()


def load_runs(input_paths: List[str]) -> List[Dict[str, Any]]:
    """Load all JSON files from provided paths or directories."""
    files: List[Path] = []
    for p_str in input_paths:
        p = Path(p_str)
        if p.is_dir():
            files.extend(sorted(p.glob("*.json")))
        elif p.is_file():
            files.append(p)

    runs: List[Dict[str, Any]] = []
    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
                if isinstance(data, dict) and "benchmark" in data:
                    runs.append(data)
        except Exception as e:
            print(f"[warning] failed to read {f}: {e}", file=sys.stderr)
    return runs


def main():
    args = parse_args()
    runs = load_runs(args.inputs)
    if not runs:
        print("Error: no valid benchmark JSON runs found.", file=sys.stderr)
        sys.exit(1)

    markdown = generate_report(runs)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown, encoding="utf-8")
        print(f"Report written to: {args.output}")
    else:
        print(markdown)


if __name__ == "__main__":
    main()
