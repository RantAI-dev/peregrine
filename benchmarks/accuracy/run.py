#!/usr/bin/env python3
"""Task-accuracy gate harness for Peregrine (roadmap S3.3).

Runs a fixed prompt set against any OpenAI-compatible chat server -- Peregrine's
``peregrine serve``, llama.cpp's ``llama-server``, vLLM -- with greedy decoding, then
scores the replies (GSM8K numeric match, MMLU letter match, HumanEval+ sandboxed
execution) and reports engine-to-engine differences with confidence intervals.

Subcommands::

    fetch      download the pinned datasets into the cache
    run        run the prompt set against one server
    score      score a run's replies
    report     merge scored runs into a Markdown report and gate verdict
    invariance decode prompts alone, batched, and with speculation
    compare    compare invariance phases byte for byte

Quick start::

    python3 benchmarks/accuracy/run.py fetch --cache /var/tmp/pg/work/datasets
    python3 benchmarks/accuracy/run.py run --engine peregrine \
        --url http://127.0.0.1:8299 --model-file /var/tmp/pg/models/acc-gate/m.gguf \
        --name qwen25-peregrine --tasks gsm8k,mmlu,humaneval \
        --limit gsm8k=250,mmlu=500 --out results/today/qwen25-peregrine
    python3 benchmarks/accuracy/run.py score --run results/today/qwen25-peregrine
    python3 benchmarks/accuracy/run.py report --runs results/today --out results/today/report.md
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data  # noqa: E402
import prompts  # noqa: E402
import scorers  # noqa: E402
from client import Client, ServerError, message_text, reasoning_text  # noqa: E402

DEFAULT_CACHE = data.DEFAULT_CACHE
DEFAULT_SEED = 20261109
TASKS = ("gsm8k", "mmlu", "humaneval")
GATE_TASKS = {"gsm8k": "GSM8K", "mmlu": "MMLU", "humaneval": "HumanEval+"}


# --------------------------------------------------------------------------- helpers


def neutral_path(path: str) -> str:
    """A recorded path with no home-directory component."""
    full = os.path.abspath(path)
    for root, label in (("/var/tmp/pg", "<pg>"),):
        if full == root or full.startswith(root + os.sep):
            return label + "/" + os.path.relpath(full, root)
    return "<external>/" + os.path.basename(full)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str) -> str | None:
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def host_info() -> dict:
    cpu = "unknown"
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.lower().startswith("model name") or line.lower().startswith("model\t"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    gpu = None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode == 0 and out.stdout.strip():
            gpu = out.stdout.strip().splitlines()[0]
    except (OSError, subprocess.SubprocessError):
        pass
    return {
        "machine": platform.node(),
        "arch": platform.machine(),
        "cpu": cpu,
        "cores": os.cpu_count(),
        "kernel": platform.release(),
        "gpu": gpu,
        "python": platform.python_version(),
    }


def parse_overrides(text: str | None) -> dict:
    out = {}
    if not text:
        return out
    for part in text.split(","):
        if not part.strip():
            continue
        key, _, value = part.partition("=")
        out[key.strip()] = int(value) if value.strip() else None
    return out


def read_jsonl(path: str) -> list:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- fetch


def cmd_fetch(args) -> int:
    meta = data.build_tasks(args.cache)
    print(json.dumps(meta, indent=2, sort_keys=True))
    return 0


# --------------------------------------------------------------------------- run


def task_rows(task: str, cache: str, limit: int | None, seed: int) -> list:
    rows = data.load_task(cache, task)
    return data.select(rows, limit, seed, stratified=(task == "mmlu"))


def cmd_run(args) -> int:
    cache = args.cache
    limits = parse_overrides(args.limit)
    out = args.out
    os.makedirs(out, exist_ok=True)
    existing = {}
    for task in TASKS:
        path = os.path.join(out, f"{task}.jsonl")
        if os.path.exists(path):
            existing[task] = {row["id"]: row for row in read_jsonl(path)}
        else:
            existing[task] = {}

    client = Client(args.url, args.model_id, timeout=args.timeout, retries=args.retries)
    dataset_manifest = {}
    manifest_path = os.path.join(cache, "dataset_manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as f:
            dataset_manifest = json.load(f)

    task_list = args.tasks.split(",")
    selected = {}
    for task in task_list:
        limit = limits.get(task, limits.get("all"))
        rows = task_rows(task, cache, limit, args.seed)
        selected[task] = rows
        if task == "mmlu":
            shot_cache: dict[str, list] = {}

            def shots_for(subject, shot_cache=shot_cache):
                if subject not in shot_cache:
                    shot_cache[subject] = data.mmlu_shots(cache, subject, args.mmlu_shots)
                return shot_cache[subject]

            prompts_for_task = [prompts.mmlu_messages(row, shots_for(row["subject"])) for row in rows]
        else:
            prompts_for_task = [prompts.build(task, row) for row in rows]

        records = existing.get(task, {})
        todo = []
        for row, (messages, max_tokens) in zip(rows, prompts_for_task):
            if row["id"] in records and records[row["id"]].get("output") is not None:
                continue
            todo.append((row, messages, max_tokens))
        print(f"{task}: {len(rows)} items, {len(todo)} to run", file=sys.stderr)

        def one(job):
            row, messages, max_tokens = job
            started = time.monotonic()
            try:
                result = client.chat(
                    messages,
                    max_tokens,
                    enable_thinking=False,
                    extra={"reasoning_effort": "none"} if args.reasoning_effort_none else None,
                )
                response = result["response"]
                return {
                    "id": row["id"],
                    "task": task,
                    "prompt_sha256": sha256_text(json.dumps(messages, ensure_ascii=False)),
                    "output": message_text(response),
                    "reasoning": reasoning_text(response) or None,
                    "output_sha256": sha256_text(message_text(response)),
                    "finish_reason": (response.get("choices") or [{}])[0].get("finish_reason"),
                    "usage": response.get("usage"),
                    "timings": response.get("timings"),
                    "latency_s": result["latency_s"],
                    "error": None,
                }
            except ServerError as exc:
                return {
                    "id": row["id"],
                    "task": task,
                    "prompt_sha256": sha256_text(json.dumps(messages, ensure_ascii=False)),
                    "output": None,
                    "output_sha256": None,
                    "finish_reason": None,
                    "usage": None,
                    "timings": None,
                    "latency_s": time.monotonic() - started,
                    "error": str(exc)[:2000],
                }

        done = 0
        t0 = time.monotonic()

        def checkpoint():
            # Write atomically so an interrupted run leaves a complete, resumable file.
            written = [records[row["id"]] for row in rows if row["id"] in records]
            path = os.path.join(out, f"{task}.jsonl")
            tmp = path + ".tmp"
            write_jsonl(tmp, written)
            os.replace(tmp, path)

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            for record in pool.map(one, todo):
                records[record["id"]] = record
                done += 1
                if done % 5 == 0:
                    checkpoint()
                if done % 25 == 0 or done == len(todo):
                    elapsed = time.monotonic() - t0
                    rate = done / elapsed if elapsed else 0
                    print(
                        f"  {task}: {done}/{len(todo)} ({rate:.2f} items/s)",
                        file=sys.stderr,
                    )
        written = [records[row["id"]] for row in rows if row["id"] in records]
        if written and all(row.get("error") for row in written):
            raise SystemExit(
                f"{task}: every request failed ({written[0]['error'][:200]}); refusing to write a dead run"
            )
        checkpoint()
        data.write_subset(os.path.join(out, "items", f"{task}.jsonl"), rows)

    manifest = {
        "run_id": os.path.basename(out),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "engine": args.engine,
        "engine_version": args.server_version,
        "model": {
            "name": args.model_name,
            "file": neutral_path(args.model_file) if args.model_file else None,
            "sha256": sha256_file(args.model_file) if args.model_file else None,
            "served_id": args.model_id,
        },
        "server": {
            "url": args.url,
            "threads": args.threads,
            "args": args.server_args,
            "info": client.info(),
        },
        "settings": {
            "temperature": 0.0,
            "greedy": True,
            "seed": args.seed,
            "mmlu_shots": args.mmlu_shots,
            "concurrency": args.concurrency,
            "enable_thinking": False,
            "reasoning_effort": "none" if args.reasoning_effort_none else None,
            "limit": limits,
            "note": args.note,
        },
        "tasks": {
            task: {
                "items": len(selected[task]),
                "ids_sha256": sha256_text(json.dumps([row["id"] for row in selected[task]])),
                "max_tokens": {"gsm8k": prompts.GSM8K_MAX_TOKENS, "mmlu": prompts.MMLU_MAX_TOKENS, "humaneval": prompts.HUMANEVAL_MAX_TOKENS}[task],
            }
            for task in task_list
        },
        "dataset": {
            "sources": dataset_manifest.get("sources"),
            "tasks_sha256": {task: dataset_manifest.get("tasks_sha256", {}).get(task) for task in task_list},
        },
        "host": host_info(),
    }
    # A run may be split across invocations (one task at a time on a slow host):
    # keep the tasks and dataset entries an earlier invocation recorded.
    manifest_path_out = os.path.join(out, "manifest.json")
    if os.path.exists(manifest_path_out):
        try:
            with open(manifest_path_out, encoding="utf-8") as f:
                previous = json.load(f)
            previous["tasks"].update(manifest["tasks"])
            manifest["tasks"] = previous["tasks"]
            previous["dataset"]["tasks_sha256"].update(manifest["dataset"]["tasks_sha256"])
            manifest["dataset"]["tasks_sha256"] = previous["dataset"]["tasks_sha256"]
        except (OSError, KeyError, ValueError):
            pass
    with open(manifest_path_out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"wrote {out}", file=sys.stderr)
    return 0


# --------------------------------------------------------------------------- score


def load_items(run_dir: str, task: str, cache: str, seed: int, limit: int | None) -> dict:
    path = os.path.join(run_dir, "items", f"{task}.jsonl")
    if os.path.exists(path):
        return {row["id"]: row for row in read_jsonl(path)}
    return {row["id"]: row for row in task_rows(task, cache, limit, seed)}


def cmd_score(args) -> int:
    run_dir = args.run
    with open(os.path.join(run_dir, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    seed = manifest["settings"].get("seed", DEFAULT_SEED)
    limits = manifest["settings"].get("limit", {})
    python = args.python
    summary = {}
    tasks = args.tasks.split(",") if args.tasks else list(manifest["tasks"].keys())
    for task in tasks:
        raw_path = os.path.join(run_dir, f"{task}.jsonl")
        if not os.path.exists(raw_path):
            continue
        limit = limits.get(task, limits.get("all"))
        items = load_items(run_dir, task, args.cache, seed, limit)
        scored = []
        for record in read_jsonl(raw_path):
            item = items.get(record["id"])
            if item is None:
                continue
            if record.get("error") or record.get("output") is None:
                detail = {"correct": False, "reason": "server error", "error": record.get("error")}
            else:
                try:
                    detail = scorers.score(
                        task,
                        item,
                        record["output"],
                        python=python,
                        timeout=args.exec_timeout,
                    )
                except Exception as exc:  # a scorer bug must not lose the run
                    detail = {"correct": False, "reason": f"scorer error: {exc!r}"}
            scored.append(
                {
                    "id": record["id"],
                    "task": task,
                    "correct": bool(detail.pop("correct", False)),
                    "detail": detail,
                    "output": record.get("output"),
                    "output_sha256": record.get("output_sha256"),
                    "latency_s": record.get("latency_s"),
                    "usage": record.get("usage"),
                    "error": record.get("error"),
                }
            )
        write_jsonl(os.path.join(run_dir, f"{task}.scored.jsonl"), scored)
        n = len(scored)
        correct = sum(1 for row in scored if row["correct"])
        errors = sum(1 for row in scored if row.get("error"))
        latencies = [row["latency_s"] for row in scored if row.get("latency_s")]
        summary[task] = {
            "n": n,
            "correct": correct,
            "accuracy": correct / n if n else None,
            "errors": errors,
            "mean_latency_s": sum(latencies) / len(latencies) if latencies else None,
        }
        if task == "humaneval":
            durations = [row["detail"].get("duration_s") for row in scored if row["detail"].get("duration_s")]
            if durations:
                summary[task]["mean_exec_s"] = sum(durations) / len(durations)
        print(f"{task}: {correct}/{n} = {summary[task]['accuracy']}", file=sys.stderr)
    summary_path = os.path.join(run_dir, "summary.json")
    merged = {}
    if os.path.exists(summary_path):
        try:
            with open(summary_path, encoding="utf-8") as f:
                merged = json.load(f)
        except (OSError, ValueError):
            merged = {}
    merged.update(summary)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, sort_keys=True)
        f.write("\n")
    return 0


# --------------------------------------------------------------------------- report


def wilson(correct: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = correct / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def difference_ci(c1: int, n1: int, c2: int, n2: int, z: float = 1.96) -> tuple[float, float, float]:
    """Difference p1-p2 and its 95% CI (unpooled, Wald).  Returns (diff, lo, hi) in points."""
    if n1 == 0 or n2 == 0:
        return (0.0, -100.0, 100.0)
    p1, p2 = c1 / n1, c2 / n2
    se = ((p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2) ** 0.5)
    diff = p1 - p2
    return (100 * diff, 100 * (diff - z * se), 100 * (diff + z * se))


def read_scored(run_dir: str, task: str) -> dict:
    path = os.path.join(run_dir, f"{task}.scored.jsonl")
    if not os.path.exists(path):
        return {}
    return {row["id"]: row for row in read_jsonl(path)}


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * tail)


def paired_difference(b: int, c: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Paired accuracy difference (points) and its 95% CI, from discordant counts."""
    if n == 0:
        return (0.0, -100.0, 100.0)
    diff = (b - c) / n
    var = (b + c) / n - diff * diff
    se = (max(var, 0.0) / n) ** 0.5
    return (100 * diff, 100 * (diff - z * se), 100 * (diff + z * se))


def cmd_report(args) -> int:
    run_dirs = args.runs if isinstance(args.runs, list) else [args.runs]
    summaries = {}
    for runs_dir in run_dirs:
        for entry in sorted(os.listdir(runs_dir)):
            path = os.path.join(runs_dir, entry)
            spath = os.path.join(path, "summary.json")
            mpath = os.path.join(path, "manifest.json")
            if not (os.path.exists(spath) and os.path.exists(mpath)):
                continue
            with open(spath, encoding="utf-8") as f:
                summary = json.load(f)
            with open(mpath, encoding="utf-8") as f:
                manifest = json.load(f)
            summaries[entry] = {
                "summary": summary,
                "manifest": manifest,
                "dir": path,
                "scored": {},
            }
            for task in summary:
                summaries[entry]["scored"][task] = read_scored(path, task)

    if not summaries:
        raise SystemExit(f"no scored runs under {run_dirs}")

    def run_key(entry: str) -> str:
        stem, _, suffix = entry.rpartition("-")
        return stem if suffix in ("peregrine", "llama", "vllm") else entry

    keys = {entry: run_key(entry) for entry in summaries}

    def find(engine: str, key: str | None = None) -> list[str]:
        found = [
            name
            for name, info in summaries.items()
            if info["manifest"].get("engine") == engine and (key is None or keys[name] == key)
        ]
        return found

    reference = find(args.reference)
    baseline = find(args.baseline)
    lines = ["# Task-accuracy gate", ""]
    lines.append(
        f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} from "
        f"`{', '.join(os.path.basename(d.rstrip('/')) for d in run_dirs)}`."
    )
    lines.append("")
    lines.append("Greedy decoding, fixed prompts (`prompts.py`), the same item set per task across engines.")
    lines.append("Confidence intervals are Wilson 95%; paired differences use the discordant pairs only.")
    lines.append("")

    header = ["run", "engine", "task", "accuracy", "n", "95% CI", "mean s/item"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for name, info in summaries.items():
        for task, row in sorted(info["summary"].items()):
            lo, hi = wilson(row["correct"], row["n"])
            lines.append(
                "| {run} | {engine} | {task} | {acc:.1f}% | {n} | {lo:.1f}-{hi:.1f}% | {lat} |".format(
                    run=name,
                    engine=info["manifest"].get("engine", "?"),
                    task=GATE_TASKS.get(task, task),
                    acc=100 * (row["accuracy"] or 0),
                    n=row["n"],
                    lo=100 * lo,
                    hi=100 * hi,
                    lat=f"{row['mean_latency_s']:.2f}" if row.get("mean_latency_s") else "-",
                )
            )
    lines.append("")

    def paired(engine_name: str, base_name: str, task: str, ids=None):
        a = summaries[engine_name]["scored"].get(task, {})
        b = summaries[base_name]["scored"].get(task, {})
        common = sorted((set(a) & set(b)) if ids is None else (set(a) & set(b) & set(ids)))
        if not common:
            return None
        b_only = sum(1 for pid in common if a[pid]["correct"] and not b[pid]["correct"])
        c_only = sum(1 for pid in common if b[pid]["correct"] and not a[pid]["correct"])
        diff, lo, hi = paired_difference(b_only, c_only, len(common))
        return {
            "n": len(common),
            "diff": diff,
            "lo": lo,
            "hi": hi,
            "p": mcnemar_exact(b_only, c_only),
            "engine_only": b_only,
            "base_only": c_only,
        }

    lines.append("## Paired differences")
    lines.append("")
    lines.append(
        "`Δ llama.cpp` is the accuracy difference on the items both runs answered, with the 95% CI of the "
        "paired difference and the exact McNemar p-value. `Δ ref` is the same against the full-precision "
        "reference. `n` is the number of paired items."
    )
    lines.append("")
    header = ["run", "task", "n", "Δ ref", "Δ llama.cpp", "paired 95% CI (pp)", "McNemar p", "gate"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    gate_rows = []
    for name, info in summaries.items():
        if info["manifest"].get("engine") in ("vllm", "llama_cpp"):
            continue
        key = keys[name]
        ref_candidates = find(args.reference, key) or reference
        base_candidates = find(args.baseline, key) or baseline
        for task in sorted(info["summary"]):
            row = info["summary"][task]
            if not row["n"]:
                continue
            # Every comparison is made on the same items: the intersection of this
            # run, its llama.cpp counterpart and the BF16 reference run.
            common_ids = set(info["scored"].get(task, {}))
            if base_candidates:
                common_ids &= set(summaries[base_candidates[0]]["scored"].get(task, {}))
            if ref_candidates:
                common_ids &= set(summaries[ref_candidates[0]]["scored"].get(task, {}))
            ref_stats = paired(name, ref_candidates[0], task, common_ids) if ref_candidates else None
            base_stats = paired(name, base_candidates[0], task, common_ids) if base_candidates else None
            base_ref_stats = (
                paired(base_candidates[0], ref_candidates[0], task, common_ids)
                if base_candidates and ref_candidates
                else None
            )
            gate = "n/a"
            if base_stats:
                within = base_stats["lo"] >= -args.llama_tolerance_pp and base_stats["hi"] <= args.llama_tolerance_pp
                not_worse = base_stats["diff"] >= -args.llama_tolerance_pp
                llama_ok = within or (not_worse and base_stats["p"] >= 0.05)
                # Quantisation costs every engine accuracy: judge Peregrine against
                # llama.cpp's own distance from the BF16 reference on the same items,
                # not against BF16 directly.
                ref_ok = (
                    ref_stats is None
                    or base_ref_stats is None
                    or ref_stats["diff"] >= base_ref_stats["diff"] - args.reference_budget_pp
                )
                gate = "pass" if (llama_ok and ref_ok) else "FAIL"
                gate_rows.append((name, task, ref_stats, base_stats, gate))
            lines.append(
                "| {run} | {task} | {n} | {dref} | {dbase} | [{lo:+.1f}, {hi:+.1f}] | {p} | {gate} |".format(
                    run=name,
                    task=GATE_TASKS.get(task, task),
                    n=base_stats["n"] if base_stats else "-",
                    dref=f"{ref_stats['diff']:+.1f}" if ref_stats else "-",
                    dbase=f"{base_stats['diff']:+.1f}" if base_stats else "-",
                    lo=base_stats["lo"] if base_stats else 0,
                    hi=base_stats["hi"] if base_stats else 0,
                    p=f"{base_stats['p']:.3f}" if base_stats else "-",
                    gate=gate,
                )
            )
    lines.append("")

    if gate_rows:
        failed = [row for row in gate_rows if row[4] == "FAIL"]
        lines.append("## Gate verdict")
        lines.append("")
        if failed:
            lines.append(f"**{len(failed)} of {len(gate_rows)} checks fail** the proposed gate:")
            lines.append("")
            for name, task, ref_stats, base_stats, _ in failed:
                if base_stats:
                    lines.append(
                        f"- `{name}` on {GATE_TASKS.get(task, task)}: {base_stats['diff']:+.1f} pp vs llama.cpp "
                        f"(paired 95% CI [{base_stats['lo']:+.1f}, {base_stats['hi']:+.1f}], p={base_stats['p']:.3f})"
                    )
                else:
                    lines.append(f"- `{name}` on {GATE_TASKS.get(task, task)}: more than "
                                 f"{args.reference_budget_pp:.0f} points below the reference")
        else:
            lines.append(
                f"All {len(gate_rows)} checks pass: within ±{args.llama_tolerance_pp:.0f} pp of llama.cpp "
                f"(paired CI or non-significant McNemar) and no more than {args.reference_budget_pp:.0f} points "
                f"below the full-precision reference."
            )
        lines.append("")

    text = "\n".join(lines) + "\n"
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
    print(text)
    with open(os.path.join(runs_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({name: info["summary"] for name, info in summaries.items()}, f, indent=2, sort_keys=True)
        f.write("\n")
    return 0 if not gate_rows or all(row[4] != "FAIL" for row in gate_rows) else 1


# --------------------------------------------------------------------------- selftest


def cmd_selftest(args) -> int:
    """Check the scorers against the datasets' own answers (canonical code must pass)."""
    rows = data.load_task(args.cache, "humaneval")
    failures = []
    for row in rows:
        result = scorers.score_humaneval(row, row["prompt"] + row["canonical_solution"], python=args.python)
        if not result["correct"]:
            failures.append(row["id"])
    print(f"humaneval canonical solutions: {len(rows) - len(failures)}/{len(rows)} pass")
    if failures:
        print("failures:", ", ".join(failures[:10]))
        return 1
    gsm = data.load_task(args.cache, "gsm8k")[:5]
    for row in gsm:
        got = scorers.score_gsm8k(row, f"The answer is #### {row['answer_number']}.")
        if not got["correct"]:
            print("gsm8k selftest failed on", row["id"])
            return 1
    mmlu = data.load_task(args.cache, "mmlu")[:5]
    for row in mmlu:
        got = scorers.score_mmlu(row, f"Answer: {row['answer']}")
        if not got["correct"]:
            print("mmlu selftest failed on", row["id"])
            return 1
    print("gsm8k and mmlu extractors: ok")
    return 0


# --------------------------------------------------------------------------- invariance


INVARIANCE_MIX = {"gsm8k": 0.40, "mmlu": 0.30, "humaneval": 0.30}


def build_invariance_prompts(cache: str, total: int, seed: int) -> list:
    """A deterministic mixed prompt set, built once and reused by every phase."""
    rows = []
    for task, share in INVARIANCE_MIX.items():
        count = int(total * share)
        items = task_rows(task, cache, count, seed)
        if task == "mmlu":
            shot_cache: dict[str, list] = {}

            def shots_for(subject, shot_cache=shot_cache):
                if subject not in shot_cache:
                    shot_cache[subject] = data.mmlu_shots(cache, subject, 5)
                return shot_cache[subject]

            for item in items:
                messages, _ = prompts.mmlu_messages(item, shots_for(item["subject"]))
                rows.append({"id": item["id"], "task": task, "messages": messages})
        else:
            for item in items:
                messages, _ = prompts.build(task, item)
                rows.append({"id": item["id"], "task": task, "messages": messages})
    return rows


def cmd_invariance(args) -> int:
    out = args.out
    os.makedirs(out, exist_ok=True)
    prompts_path = os.path.join(out, "prompts.jsonl")
    if args.prompts_file:
        rows = read_jsonl(args.prompts_file)
    elif os.path.exists(prompts_path):
        rows = read_jsonl(prompts_path)
    else:
        rows = build_invariance_prompts(args.cache, args.prompts, args.seed)
        write_jsonl(prompts_path, rows)
    client = Client(args.url, args.model_id, timeout=args.timeout)
    phase = args.phase
    print(f"{phase}: {len(rows)} prompts", file=sys.stderr)

    def one(row):
        result = client.chat(row["messages"], args.max_tokens, enable_thinking=False)
        text = message_text(result["response"])
        return {
            "id": row["id"],
            "task": row["task"],
            "output": text,
            "output_sha256": sha256_text(text),
            "latency_s": result["latency_s"],
            "usage": result["response"].get("usage"),
        }

    started = time.monotonic()
    records = []
    if phase == "batched":
        group = args.group
        with concurrent.futures.ThreadPoolExecutor(max_workers=group) as pool:
            for i in range(0, len(rows), group):
                chunk = rows[i : i + group]
                records.extend(pool.map(one, chunk))
                if (i // group) % 5 == 0:
                    print(f"  {len(records)}/{len(rows)}", file=sys.stderr)
    else:
        for i, row in enumerate(rows):
            records.append(one(row))
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(rows)}", file=sys.stderr)
    elapsed = time.monotonic() - started
    write_jsonl(os.path.join(out, f"{phase}.jsonl"), records)
    summary = {
        "phase": phase,
        "prompts": len(records),
        "max_tokens": args.max_tokens,
        "group": args.group if phase == "batched" else 1,
        "wall_s": elapsed,
        "items_per_s": len(records) / elapsed if elapsed else None,
        "url": args.url,
        "server": args.server_label,
    }
    with open(os.path.join(out, f"{phase}.summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")
    return 0


def cmd_compare(args) -> int:
    phases = []
    for path in args.phases:
        records_path = path
        if not path.endswith(".jsonl"):
            for candidate in ("alone", "batched", "speculation"):
                probe = os.path.join(path, f"{candidate}.jsonl")
                if os.path.exists(probe):
                    records_path = probe
                    break
            else:
                raise SystemExit(f"no phase jsonl under {path}")
        name = os.path.splitext(os.path.basename(records_path))[0]
        records = {row["id"]: row for row in read_jsonl(records_path)}
        phases.append((name, records_path, records))
    if len(phases) < 2:
        raise SystemExit("need at least two phase directories")
    ids = set(phases[0][2])
    for name, _, records in phases[1:]:
        ids &= set(records)
    mismatches = []
    for pid in sorted(ids):
        hashes = {name: records[pid]["output_sha256"] for name, _, records in phases}
        if len(set(hashes.values())) > 1:
            mismatches.append({"id": pid, "hashes": hashes})
    report = {
        "prompts": len(ids),
        "phases": [name for name, _, _ in phases],
        "identical": len(ids) - len(mismatches),
        "mismatches": len(mismatches),
        "first_mismatches": mismatches[:10],
    }
    text = [
        "# Decoding invariance",
        "",
        f"Compared {len(ids)} prompts across {', '.join(name for name, _, _ in phases)}.",
        "",
        f"- byte-identical: **{report['identical']}** / {len(ids)}",
        f"- mismatches: **{report['mismatches']}**",
        "",
    ]
    if mismatches:
        text.append("| prompt | " + " | ".join(name for name, _, _ in phases) + " |")
        text.append("|" + "---|" * (len(phases) + 1))
        for row in mismatches[:20]:
            text.append(
                "| {id} | {h} |".format(id=row["id"], h=" | ".join(h[:12] for h in row["hashes"].values()))
            )
    out = args.out
    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            f.write("\n".join(text) + "\n")
        with open(os.path.splitext(out)[0] + ".json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, sort_keys=True)
            f.write("\n")
    print("\n".join(text))
    return 0 if not mismatches else 1


# --------------------------------------------------------------------------- cli


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", default=DEFAULT_CACHE, help="dataset cache (default: %(default)s)")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--cache", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("fetch", parents=[common], help="download and normalise the pinned datasets")
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser("run", parents=[common], help="run the prompt set against one server")
    p.add_argument("--engine", required=True, help="label: peregrine, llama_cpp, vllm, ...")
    p.add_argument("--url", required=True)
    p.add_argument("--model-id", default=None, help="served model id (Peregrine accepts any)")
    p.add_argument("--model-file", default=None, help="GGUF/HF path, recorded hashed and path-neutralised")
    p.add_argument("--model-name", default=None)
    p.add_argument("--name", default=None, help="run directory name (default: from model and engine)")
    p.add_argument("--tasks", default="gsm8k,mmlu,humaneval")
    p.add_argument("--limit", default=None, help="per-task item cap, e.g. gsm8k=250,mmlu=500")
    p.add_argument("--mmlu-shots", type=int, default=5)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--timeout", type=float, default=900.0)
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--exec-timeout", type=float, default=10.0, help="HumanEval sandbox seconds")
    p.add_argument("--threads", type=int, default=None, help="engine thread count (recorded)")
    p.add_argument("--server-args", default=None, help="engine command line (recorded)")
    p.add_argument("--server-version", default=None)
    p.add_argument("--note", default=None)
    p.add_argument("--reasoning-effort-none", action="store_true", help="send reasoning_effort=none (Peregrine/llama.cpp)")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("score", help="score a run")
    p.add_argument("--run", required=True)
    p.add_argument("--tasks", default=None, help="comma-separated subset")
    p.add_argument("--python", default=sys.executable, help="interpreter for the HumanEval sandbox")
    p.add_argument("--exec-timeout", type=float, default=10.0)
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("report", help="merge scored runs")
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--reference", default="vllm")
    p.add_argument("--baseline", default="llama_cpp")
    p.add_argument("--llama-tolerance-pp", type=float, default=1.0)
    p.add_argument("--reference-budget-pp", type=float, default=3.0)
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("invariance", help="decode a prompt set in one phase")
    p.add_argument("--url", required=True)
    p.add_argument("--model-id", default=None)
    p.add_argument("--phase", required=True, choices=("alone", "batched", "speculation"))
    p.add_argument("--group", type=int, default=8)
    p.add_argument("--prompts", type=int, default=520)
    p.add_argument("--prompts-file", default=None)
    p.add_argument("--max-tokens", type=int, default=48)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--timeout", type=float, default=900.0)
    p.add_argument("--server-label", default=None)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_invariance)

    p = sub.add_parser("selftest", parents=[common], help="check the scorers on canonical answers")
    p.add_argument("--python", default=sys.executable)
    p.set_defaults(func=cmd_selftest)

    p = sub.add_parser("compare", help="compare invariance phases")
    p.add_argument("--phases", nargs="+", required=True, help="phase directories, each with phase.jsonl")
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_compare)

    args = parser.parse_args(argv)
    if args.command == "run" and args.name:
        args.out = os.path.join(os.path.dirname(args.out.rstrip("/")), args.name)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
