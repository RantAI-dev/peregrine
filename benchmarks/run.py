#!/usr/bin/env python3
"""Peregrine Benchmark Harness: runner for reproducible engine benchmarks.

Measures the 'four numbers' across reference and competing engines at pinned versions:
1. Single-user decode throughput (tok/s)
2. Prefill / prompt throughput (tok/s)
3. Multi-user aggregate throughput at 4 and 10 users (tok/s)
4. Time-to-first-token (TTFT) at 10 users (s)

Interleaves engines ABAB..., computes fraction-of-ceiling against measured memory bandwidth,
and writes structured JSON run manifests.
"""

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import uuid
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from engines import (
    EngineAdapter,
    PeregrineAdapter,
    LlamaCppAdapter,
    IkLlamaCppAdapter,
)
from engines.base import send_chat_completion, run_concurrent_requests


def get_cpu_info() -> Dict[str, Any]:
    """Collect host CPU model and hardware feature flags from Linux /proc/cpuinfo."""
    cpu_model = "unknown"
    flags: List[str] = []
    if Path("/proc/cpuinfo").exists():
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("model name") and cpu_model == "unknown":
                        cpu_model = line.split(":", 1)[1].strip()
                    elif line.startswith("flags"):
                        flags = line.split(":", 1)[1].strip().split()
        except Exception:
            pass

    key_features = [
        f for f in flags if f in {
            "avx", "avx2", "fma", "avx512f", "avx512vnni", "avx_vnni",
            "avx512_bf16", "neon", "asimd",
        }
    ]

    return {
        "model": cpu_model,
        "features": key_features,
        "cores_logical": os.cpu_count() or 1,
    }


def get_governor() -> str:
    """Read CPU scaling governor if accessible."""
    gov_path = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    if gov_path.exists():
        try:
            return gov_path.read_text(encoding="utf-8").strip()
        except Exception:
            pass
    return "unknown (VM or fixed frequency)"


def get_machine_record(peregrine: PeregrineAdapter, threads: int) -> Dict[str, Any]:
    """Probe machine ceilings via Peregrine's telemetry and gather system facts."""
    cpu_info = get_cpu_info()
    probe_data = {}
    try:
        probe_data = peregrine.probe_machine(threads=threads)
    except Exception as e:
        print(f"[warning] telemetry probe failed: {e}", file=sys.stderr)
        probe_data = {
            "bandwidth_bytes_per_s": 0.0,
            "bandwidth_gb_s": 0.0,
            "flops": 0.0,
            "gflops": 0.0,
            "simd": "unknown",
            "ridge": 0.0,
        }

    return {
        "hostname": platform.node(),
        "os": platform.system(),
        "kernel": platform.release(),
        "platform": platform.platform(),
        "cpu_model": cpu_info["model"],
        "cpu_features": cpu_info["features"],
        "threads_tested": threads,
        "governor": get_governor(),
        "bandwidth_ceiling_bytes_per_s": probe_data.get("bandwidth_bytes_per_s", 0.0),
        "bandwidth_ceiling_gb_s": probe_data.get("bandwidth_gb_s", 0.0),
        "flops_ceiling": probe_data.get("flops", 0.0),
        "gflops_ceiling": probe_data.get("gflops", 0.0),
        "simd_tier": probe_data.get("simd", "unknown"),
        "ridge_point": probe_data.get("ridge", 0.0),
    }


def compute_file_sha256(path: Path) -> str:
    """Calculate SHA-256 digest of a model file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def load_prompt_corpus(prompt_file: Optional[Path]) -> str:
    """Load text corpus for generating benchmark prompts."""
    if prompt_file and prompt_file.exists():
        return prompt_file.read_text(encoding="utf-8")
    default_corpus = Path(__file__).resolve().parent / "prompts" / "corpus.txt"
    if default_corpus.exists():
        return default_corpus.read_text(encoding="utf-8")
    return "The quick brown fox jumps over the lazy dog. Benchmarking inference engines on CPUs."


def make_prompt(corpus: str, char_count: int, run_nonce: str) -> str:
    """Build a prompt with approx char_count characters of corpus and a randomized first line to avoid prefix cache reuse."""
    text = corpus
    while len(text) < char_count:
        text += "\n" + corpus
    base_text = text[:char_count].strip()
    return f"// Benchmark run nonce: {run_nonce}\n{base_text}"


def calibrate_prompt_chars(
    url: str,
    corpus: str,
    target_tokens: int,
    model_alias: str = "peregrine",
    max_iters: int = 10,
    tolerance: float = 0.02,
) -> int:
    """Calibrate character count for a target prompt token length by probing the server.

    Sends probe requests with ignore_eos: true, measures usage.prompt_tokens,
    scales character count, and repeats until within ~2% of target_tokens.
    """
    print(f"  Calibrating prompt characters for target {target_tokens} tokens...")
    # Start conservatively (e.g. target_tokens // 2) so we never exceed server_ctx
    # on byte tokenizers (where 1 char ~= 1 token).
    nonce = f"probe-{uuid.uuid4().hex[:8]}"
    nonce_line = f"// Benchmark run nonce: {nonce}\n"
    char_count = max(20, (target_tokens // 2) - len(nonce_line))

    best_chars = char_count
    best_diff = float("inf")
    best_tokens = 0

    for iteration in range(max_iters):
        test_nonce = f"probe-{iteration}-{uuid.uuid4().hex[:8]}"
        prompt = make_prompt(corpus, char_count, test_nonce)
        try:
            res = send_chat_completion(
                url=url,
                prompt=prompt,
                max_tokens=1,
                stream=False,
                model_alias=model_alias,
                ignore_eos=True,
            )
        except Exception as e:
            if char_count > 40:
                char_count = char_count // 2
                continue
            raise e

        actual_tokens = res.get("prompt_tokens", 0)
        if actual_tokens <= 0:
            break

        diff = abs(actual_tokens - target_tokens)
        if diff < best_diff:
            best_diff = diff
            best_chars = char_count
            best_tokens = actual_tokens

        ratio = diff / float(target_tokens)
        if ratio <= tolerance:
            print(f"  [calibrated] {char_count} chars -> {actual_tokens} tokens ({ratio * 100:.1f}% error <= 2%)")
            return char_count

        scale = float(target_tokens) / float(actual_tokens)
        new_char_count = max(10, int(char_count * scale))
        if new_char_count == char_count:
            if actual_tokens < target_tokens:
                new_char_count += 4
            else:
                new_char_count = max(10, new_char_count - 4)
        char_count = new_char_count

    print(f"  [calibrated: best] {best_chars} chars -> {best_tokens} tokens (closest diff: {best_diff})")
    return best_chars


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Peregrine Benchmark Harness (Roadmap S3.1)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--models", "-m",
        nargs="+",
        help="Path(s) to GGUF model files to benchmark.",
    )
    parser.add_argument(
        "--engines", "-e",
        default="peregrine",
        help="Comma-separated engine names (peregrine, llama_cpp, ik_llama_cpp).",
    )
    parser.add_argument(
        "--contexts", "-c",
        default="512,4096,16384",
        help="Comma-separated context lengths to test.",
    )
    parser.add_argument(
        "--runs", "-r",
        type=int,
        default=3,
        help="Number of interleaved repeat runs (minimum 3 for reliable medians).",
    )
    parser.add_argument(
        "--tokens", "-n",
        type=int,
        default=256,
        help="Greedy decode tokens per request (enough that decode, not prefill, dominates).",
    )
    parser.add_argument(
        "--threads", "-t",
        type=int,
        default=8,
        help="Worker thread count for engines.",
    )
    parser.add_argument(
        "--user-counts",
        default="1,4,10",
        help="Comma-separated user counts (phases) to run; lets a slow host split a "
        "context across several commands under a wall-clock cap.",
    )
    parser.add_argument(
        "--llama-fa",
        choices=["auto", "on", "off"],
        default="auto",
        help="llama.cpp flash attention setting, passed as -fa to both llama-server and "
        "llama-bench. The default reproduces llama.cpp's own default and older runs; "
        "benchmark reports compare against the faster setting per row (see benchmarks/README.md).",
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8201,
        help="HTTP server port.",
    )
    parser.add_argument(
        "--mode",
        choices=["server", "cli"],
        default="server",
        help="Benchmarking mode: full server (measures all 4 numbers) or CLI.",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        help="Optional path to a custom text file for prompt generation.",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("benchmarks/results"),
        help="Directory to save raw JSON run sets.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("~/.cache/peregrine-bench"),
        help="Cache directory for third-party engine clones and builds.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print build and execution commands without running them.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke test mode: generates synthetic model if none provided and runs 1 quick round.",
    )
    return parser.parse_args()


def calculate_stats(samples: List[float]) -> Dict[str, float]:
    """Calculate median and spread (half-range: (max - min) / 2) from samples."""
    if not samples:
        return {"median": 0.0, "spread": 0.0, "min": 0.0, "max": 0.0}
    sorted_s = sorted(samples)
    med = statistics.median(sorted_s)
    spread = (max(sorted_s) - min(sorted_s)) / 2.0
    return {
        "median": round(med, 2),
        "spread": round(spread, 2),
        "min": round(min(sorted_s), 2),
        "max": round(max(sorted_s), 2),
    }


def main():
    args = parse_args()
    workspace_root = Path(__file__).resolve().parent.parent
    cache_dir = str(args.cache_dir.expanduser())
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Instantiate engine adapters
    peregrine_adapter = PeregrineAdapter(workspace_root=workspace_root)
    engine_map: Dict[str, EngineAdapter] = {
        "peregrine": peregrine_adapter,
        "llama_cpp": LlamaCppAdapter(flash_attn=args.llama_fa),
        "ik_llama_cpp": IkLlamaCppAdapter(),
    }

    selected_engine_names = [e.strip() for e in args.engines.split(",") if e.strip()]
    selected_engines: List[EngineAdapter] = []
    for name in selected_engine_names:
        if name not in engine_map:
            print(f"Error: unknown engine '{name}'. Available: {list(engine_map.keys())}", file=sys.stderr)
            sys.exit(1)
        selected_engines.append(engine_map[name])

    # 2. Build or locate selected engines
    print("=== Building / Locating Engines ===")
    for eng in selected_engines:
        print(f"[{eng.name}] Checking binary...")
        eng.build_or_locate(cache_dir=cache_dir, dry_run=args.dry_run)

    # 3. Model file handling
    model_paths: List[Path] = []
    if args.models:
        model_paths = [Path(m) for m in args.models]
    elif args.smoke:
        print("=== Generating Synthetic Model for Smoke Test ===")
        synth_dir = Path("/tmp/peregrine-smoke-model")
        synth_dir.mkdir(parents=True, exist_ok=True)
        # Smoke test synthetic model with 2048 context
        synth_path = peregrine_adapter.write_synthetic_model(synth_dir, ctx=2048)
        model_paths = [synth_path]
        print(f"Synthetic model generated at: {synth_path}")
    else:
        print("Error: No --models specified. Pass --models <path.gguf> or --smoke.", file=sys.stderr)
        sys.exit(1)

    # 4. Contexts and settings
    selected_user_counts = [int(c.strip()) for c in args.user_counts.split(",") if c.strip()]
    if not selected_user_counts or any(c not in (1, 4, 10) for c in selected_user_counts):
        print("Error: --user-counts must be a subset of 1,4,10.", file=sys.stderr)
        sys.exit(1)
    if args.smoke:
        contexts = [512]
        runs = 1
        decode_tokens = 16
    else:
        contexts = [int(c.strip()) for c in args.contexts.split(",") if c.strip()]
        runs = max(1, args.runs)
        decode_tokens = args.tokens

    # 5. Machine Record
    print("=== Probing Hardware Ceilings and System Record ===")
    machine_rec = get_machine_record(peregrine_adapter, threads=args.threads)
    print(f"Host: {machine_rec['cpu_model']} ({machine_rec['simd_tier']})")
    print(f"Streaming Bandwidth Ceiling: {machine_rec['bandwidth_ceiling_gb_s']:.2f} GB/s")
    print(f"FMA Compute Ceiling: {machine_rec['gflops_ceiling']:.2f} GFLOP/s (Ridge: {machine_rec['ridge_point']:.2f})")

    corpus = load_prompt_corpus(args.prompt_file)

    # 6. Execute benchmark runs per model
    for model_path in model_paths:
        if not model_path.exists():
            print(f"Error: model file not found at {model_path}", file=sys.stderr)
            continue

        print(f"\n========================================================")
        print(f"Benchmarking Model: {model_path.name}")
        print(f"========================================================")
        model_sha256 = compute_file_sha256(model_path)
        model_size = model_path.stat().st_size

        # Obtain bytes_per_token telemetry from Peregrine bench
        bytes_per_token = 0.0
        try:
            cli_res = peregrine_adapter.run_cli_bench(
                model_path=str(model_path),
                prompt_tokens=32,
                decode_tokens=16,
                threads=args.threads,
            )
            bytes_per_token = cli_res.get("bytes_per_token", 0.0)
            print(f"Peregrine Telemetry: {bytes_per_token:.1f} bytes moved per token")
        except Exception as e:
            print(f"[warning] could not get CLI bytes_per_token: {e}", file=sys.stderr)

        # Dictionary to store all engine run results
        engine_results: Dict[str, Any] = {
            eng.name: {
                "benchmark": "peregrine-s3.1",
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "model": {
                    "path": str(model_path.resolve()),
                    "filename": model_path.name,
                    "sha256": model_sha256,
                    "size_bytes": model_size,
                    "bytes_per_token_telemetry": bytes_per_token,
                },
                "machine": machine_rec,
                "engine": eng.get_version_info(),
                "settings": {
                    "contexts": contexts,
                    "runs": runs,
                    "decode_tokens": decode_tokens,
                    "threads": args.threads,
                    "mode": args.mode,
                    "user_counts": selected_user_counts,
                    "interleaved": True,
                    "llama_flash_attn": args.llama_fa,
                },
                "results": {
                    str(ctx): {
                        "single_user": {"decode_tok_s": [], "prefill_tok_s": [], "ttft_s": [], "prompt_tokens": []},
                        "multi_user_4": {"aggregate_tok_s": [], "decode_aggregate_tok_s": [], "ttft_s": [], "prompt_tokens": []},
                        "multi_user_10": {"aggregate_tok_s": [], "decode_aggregate_tok_s": [], "ttft_s": [], "prompt_tokens": []},
                    }
                    for ctx in contexts
                },
            }
            for eng in selected_engines
        }

        calibrated_chars_cache: Dict[Any, int] = {}

        # Interleaved repeat rounds: ABAB...
        for round_idx in range(runs):
            print(f"\n--- Interleaved Run {round_idx + 1} of {runs} ---")
            for eng in selected_engines:
                print(f"[{eng.name}] Starting run {round_idx + 1}...")

                if args.dry_run:
                    print(f"[dry-run] skipping execution for {eng.name}")
                    continue

                if args.mode == "cli":
                    # CLI-only benchmark mode
                    for ctx in contexts:
                        res = eng.run_cli_bench(
                            model_path=str(model_path),
                            prompt_tokens=ctx,
                            decode_tokens=decode_tokens,
                            threads=args.threads,
                        )
                        ctx_k = str(ctx)
                        engine_results[eng.name]["results"][ctx_k]["single_user"]["decode_tok_s"].append(res["decode_tok_s"])
                        engine_results[eng.name]["results"][ctx_k]["single_user"]["prefill_tok_s"].append(res["prefill_tok_s"])
                        engine_results[eng.name]["results"][ctx_k]["single_user"]["ttft_s"].append(res.get("prefill_s", 0.0))
                        engine_results[eng.name]["results"][ctx_k]["single_user"]["prompt_tokens"].append(res.get("prompt_tokens", ctx))
                    continue

                # Server mode: start HTTP server per (context, user count) phase
                for ctx in contexts:
                    ctx_k = str(ctx)
                    server_ctx = ctx + decode_tokens + 64

                    phases = [
                        (users, phase_key, phase_label)
                        for users, phase_key, phase_label in [
                            (1, "single_user", "1 user"),
                            (4, "multi_user_4", "4 users"),
                            (10, "multi_user_10", "10 users"),
                        ]
                        if users in selected_user_counts
                    ]

                    for users, phase_key, phase_label in phases:
                        print(f"[{eng.name}] Context {ctx} tokens, {phase_label} (slots={users}, ctx={server_ctx}):")
                        server_proc = eng.start_server(
                            model_path=str(model_path),
                            port=args.port,
                            threads=args.threads,
                            ctx=server_ctx,
                            slots=users,
                        )
                        server_url = f"http://127.0.0.1:{args.port}"

                        try:
                            # Warm-up request after every server start
                            print(f"[{eng.name}] Warming up...")
                            send_chat_completion(
                                url=server_url,
                                prompt="Hello world",
                                max_tokens=8,
                                stream=True,
                                ignore_eos=True,
                            )

                            # Calibrate prompt character count for this engine, model, and context
                            cache_key = (eng.name, model_path.name, ctx)
                            if cache_key not in calibrated_chars_cache:
                                calibrated_chars_cache[cache_key] = calibrate_prompt_chars(
                                    url=server_url,
                                    corpus=corpus,
                                    target_tokens=ctx,
                                    model_alias=eng.name,
                                )
                            calibrated_chars = calibrated_chars_cache[cache_key]

                            if users == 1:
                                prompt_1 = make_prompt(corpus, calibrated_chars, f"{round_idx}-1u-{uuid.uuid4().hex[:8]}")
                                r1 = send_chat_completion(
                                    url=server_url,
                                    prompt=prompt_1,
                                    max_tokens=decode_tokens,
                                    stream=True,
                                    model_alias=eng.name,
                                    ignore_eos=True,
                                )
                                actual_pt = r1.get("prompt_tokens", 0)
                                off_ratio = abs(actual_pt - ctx) / float(ctx)
                                if off_ratio > 0.05:
                                    raise RuntimeError(
                                        f"[{eng.name}] Actual prompt tokens ({actual_pt}) is {off_ratio * 100.0:.1f}% off target ({ctx}) > 5% allowed threshold!"
                                    )
                                if r1.get("completion_tokens") != decode_tokens:
                                    print(
                                        f"[warning] [{eng.name}] 1 user completion_tokens {r1.get('completion_tokens')} != max_tokens {decode_tokens}; dropping sample",
                                        file=sys.stderr,
                                    )
                                else:
                                    print(f"  1 user:  decode {r1['decode_tok_s']:.1f} tok/s, prompt {r1['prefill_tok_s']:.1f} tok/s, TTFT {r1['ttft_s']:.3f} s (prompt tokens: {actual_pt})")
                                    engine_results[eng.name]["results"][ctx_k]["single_user"]["decode_tok_s"].append(r1["decode_tok_s"])
                                    engine_results[eng.name]["results"][ctx_k]["single_user"]["prefill_tok_s"].append(r1["prefill_tok_s"])
                                    engine_results[eng.name]["results"][ctx_k]["single_user"]["ttft_s"].append(r1["ttft_s"])
                                    engine_results[eng.name]["results"][ctx_k]["single_user"]["prompt_tokens"].append(actual_pt)

                            elif users == 4:
                                prompts_4 = [
                                    make_prompt(corpus, calibrated_chars, f"{round_idx}-4u-{u}-{uuid.uuid4().hex[:8]}")
                                    for u in range(4)
                                ]
                                r4 = run_concurrent_requests(
                                    url=server_url,
                                    prompts=prompts_4,
                                    max_tokens=decode_tokens,
                                    model_alias=eng.name,
                                    ignore_eos=True,
                                )
                                user_results = r4.get("user_results", [])
                                if len(user_results) != 4 or any(u.get("completion_tokens") != decode_tokens for u in user_results):
                                    print(
                                        f"[warning] [{eng.name}] 4 users: not all users produced max_tokens={decode_tokens}; dropping sample",
                                        file=sys.stderr,
                                    )
                                else:
                                    pts = [u.get("prompt_tokens", 0) for u in user_results]
                                    for pt in pts:
                                        off_ratio = abs(pt - ctx) / float(ctx)
                                        if off_ratio > 0.05:
                                            raise RuntimeError(
                                                f"[{eng.name}] 4 users prompt tokens ({pt}) is {off_ratio * 100.0:.1f}% off target ({ctx}) > 5% allowed threshold!"
                                            )
                                    med_pt = round(statistics.median(pts), 1)
                                    print(f"  4 users: aggregate {r4['aggregate_tok_s']:.1f} tok/s (prompt tokens: {med_pt})")
                                    engine_results[eng.name]["results"][ctx_k]["multi_user_4"]["aggregate_tok_s"].append(r4["aggregate_tok_s"])
                                    engine_results[eng.name]["results"][ctx_k]["multi_user_4"]["decode_aggregate_tok_s"].append(r4["decode_aggregate_tok_s"])
                                    engine_results[eng.name]["results"][ctx_k]["multi_user_4"]["ttft_s"].append(r4["median_ttft_s"])
                                    engine_results[eng.name]["results"][ctx_k]["multi_user_4"]["prompt_tokens"].append(med_pt)

                            elif users == 10:
                                prompts_10 = [
                                    make_prompt(corpus, calibrated_chars, f"{round_idx}-10u-{u}-{uuid.uuid4().hex[:8]}")
                                    for u in range(10)
                                ]
                                r10 = run_concurrent_requests(
                                    url=server_url,
                                    prompts=prompts_10,
                                    max_tokens=decode_tokens,
                                    model_alias=eng.name,
                                    ignore_eos=True,
                                )
                                user_results = r10.get("user_results", [])
                                if len(user_results) != 10 or any(u.get("completion_tokens") != decode_tokens for u in user_results):
                                    print(
                                        f"[warning] [{eng.name}] 10 users: not all users produced max_tokens={decode_tokens}; dropping sample",
                                        file=sys.stderr,
                                    )
                                else:
                                    pts = [u.get("prompt_tokens", 0) for u in user_results]
                                    for pt in pts:
                                        off_ratio = abs(pt - ctx) / float(ctx)
                                        if off_ratio > 0.05:
                                            raise RuntimeError(
                                                f"[{eng.name}] 10 users prompt tokens ({pt}) is {off_ratio * 100.0:.1f}% off target ({ctx}) > 5% allowed threshold!"
                                            )
                                    med_pt = round(statistics.median(pts), 1)
                                    print(f"  10 users: aggregate {r10['aggregate_tok_s']:.1f} tok/s, TTFT {r10['median_ttft_s']:.3f} s (prompt tokens: {med_pt})")
                                    engine_results[eng.name]["results"][ctx_k]["multi_user_10"]["aggregate_tok_s"].append(r10["aggregate_tok_s"])
                                    engine_results[eng.name]["results"][ctx_k]["multi_user_10"]["decode_aggregate_tok_s"].append(r10["decode_aggregate_tok_s"])
                                    engine_results[eng.name]["results"][ctx_k]["multi_user_10"]["ttft_s"].append(r10["median_ttft_s"])
                                    engine_results[eng.name]["results"][ctx_k]["multi_user_10"]["prompt_tokens"].append(med_pt)

                        finally:
                            print(f"[{eng.name}] Stopping server...")
                            eng.stop_server(server_proc)

        # 7. Aggregate statistics and write JSON per engine
        print("\n=== Aggregating Results and Writing JSON ===")
        bandwidth_ceiling = machine_rec["bandwidth_ceiling_bytes_per_s"]

        for eng_name, manifest in engine_results.items():
            if args.dry_run:
                continue

            summary_results: Dict[str, Any] = {}
            for ctx_str, data in manifest["results"].items():
                su_dec = calculate_stats(data["single_user"]["decode_tok_s"])
                su_pref = calculate_stats(data["single_user"]["prefill_tok_s"])
                su_ttft = calculate_stats(data["single_user"]["ttft_s"])
                su_pts = calculate_stats(data["single_user"].get("prompt_tokens", []))

                m4_agg = calculate_stats(data["multi_user_4"]["aggregate_tok_s"])
                m4_ttft = calculate_stats(data["multi_user_4"]["ttft_s"])
                m4_pts = calculate_stats(data["multi_user_4"].get("prompt_tokens", []))

                m10_agg = calculate_stats(data["multi_user_10"]["aggregate_tok_s"])
                m10_ttft = calculate_stats(data["multi_user_10"]["ttft_s"])
                m10_pts = calculate_stats(data["multi_user_10"].get("prompt_tokens", []))

                actual_prompt_tokens = su_pts["median"] if su_pts["median"] > 0 else int(ctx_str)

                # Fraction of ceiling: (bytes_per_token * tok/s) / bandwidth_ceiling
                frac_ceiling = 0.0
                if bandwidth_ceiling > 0 and bytes_per_token > 0:
                    frac_ceiling = (bytes_per_token * su_dec["median"]) / bandwidth_ceiling

                summary_results[ctx_str] = {
                    "context": int(ctx_str),
                    "actual_prompt_tokens": actual_prompt_tokens,
                    "single_user": {
                        "decode_tok_s": su_dec,
                        "prefill_tok_s": su_pref,
                        "ttft_s": su_ttft,
                        "prompt_tokens": su_pts,
                        "bytes_per_token": bytes_per_token,
                        "fraction_of_ceiling": round(frac_ceiling, 4),
                        "percent_of_ceiling": round(frac_ceiling * 100.0, 2),
                        "raw_samples": data["single_user"],
                    },
                    "multi_user_4": {
                        "aggregate_tok_s": m4_agg,
                        "decode_aggregate_tok_s": calculate_stats(data["multi_user_4"]["decode_aggregate_tok_s"]),
                        "ttft_s": m4_ttft,
                        "prompt_tokens": m4_pts,
                        "raw_samples": data["multi_user_4"],
                    },
                    "multi_user_10": {
                        "aggregate_tok_s": m10_agg,
                        "decode_aggregate_tok_s": calculate_stats(data["multi_user_10"]["decode_aggregate_tok_s"]),
                        "first_token_time_s": m10_ttft,
                        "prompt_tokens": m10_pts,
                        "raw_samples": data["multi_user_10"],
                    },
                }

            manifest["summary"] = summary_results


            ts_slug = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")
            out_file = args.output_dir / f"{ts_slug}-{eng_name}-{model_path.stem}.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
            print(f"Saved run set to: {out_file}")


if __name__ == "__main__":
    main()
