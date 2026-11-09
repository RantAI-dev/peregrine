"""Dataset fetch and task-set construction for the task-accuracy gate (roadmap S3.3).

Every dataset is fetched from a pinned revision and hashed, so a run can be repeated
later even if the upstream repository moves.  The cache lives outside the repository
(``/var/tmp/pg/work/datasets`` by default, ``PG_ACC_CACHE`` overrides it) and holds the
raw downloads plus one normalised JSONL file per task.

Only the fetch step needs third-party packages (``pyarrow`` for MMLU's parquet files);
the runner, scorers and report are standard library only.

Task records (one JSON object per line in ``tasks/<task>.jsonl``):

    gsm8k      {"id", "task", "question", "answer_text", "answer_number"}
    mmlu       {"id", "task", "subject", "question", "choices": [4], "answer": "A".."D"}
    mmlu_dev    same as mmlu; the five per-subject dev examples used for the prompt
    humaneval  {"id", "task", "prompt", "entry_point", "test"}
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import sys
import urllib.request

DEFAULT_CACHE = os.environ.get("PG_ACC_CACHE", "/var/tmp/pg/work/datasets")

# Pinned sources.  GSM8K is the canonical OpenAI JSONL at a fixed commit; MMLU and
# HumanEval+ are Hugging Face datasets at fixed revisions.
SOURCES = {
    "gsm8k": {
        "url": "https://raw.githubusercontent.com/openai/grade-school-math/"
        "3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/test.jsonl",
        "file": "gsm8k-test.jsonl",
        "revision": "3101c7d5072418e28b9008a6636bde82a006892c",
    },
    "mmlu_test": {
        "url": "https://huggingface.co/datasets/cais/mmlu/resolve/"
        "c30699e8356da336a370243923dbaf21066bb9fe/all/test-00000-of-00001.parquet",
        "file": "mmlu-all-test.parquet",
        "revision": "c30699e8356da336a370243923dbaf21066bb9fe",
    },
    "mmlu_dev": {
        "url": "https://huggingface.co/datasets/cais/mmlu/resolve/"
        "c30699e8356da336a370243923dbaf21066bb9fe/all/dev-00000-of-00001.parquet",
        "file": "mmlu-all-dev.parquet",
        "revision": "c30699e8356da336a370243923dbaf21066bb9fe",
    },
    "humaneval": {
        "url": "https://huggingface.co/datasets/evalplus/humanevalplus/resolve/"
        "d32357cf319e50e9c8d8dab5ea876c72b0fd321b/test.jsonl",
        "file": "humanevalplus-test.jsonl",
        "revision": "d32357cf319e50e9c8d8dab5ea876c72b0fd321b",
    },
}

NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _download(url: str, path: str) -> None:
    tmp = path + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "peregrine-accuracy/1"})
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        for block in iter(lambda: r.read(1 << 20), b""):
            f.write(block)
    os.replace(tmp, path)


def fetch(cache: str = DEFAULT_CACHE) -> dict:
    """Download every pinned source into ``cache/raw`` and return the source manifest."""
    raw = os.path.join(cache, "raw")
    os.makedirs(raw, exist_ok=True)
    manifest = {}
    for name, src in SOURCES.items():
        path = os.path.join(raw, src["file"])
        if not os.path.exists(path):
            print(f"fetching {name}: {src['url']}", file=sys.stderr)
            _download(src["url"], path)
        manifest[name] = {
            "url": src["url"],
            "revision": src["revision"],
            "file": os.path.join("raw", src["file"]),
            "sha256": sha256_file(path),
        }
    return manifest


def _read_jsonl(path: str):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _parquet_rows(path: str):
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - message path
        raise SystemExit(
            "MMLU is distributed as parquet: install the fetch dependency with\n"
            "  pip install pyarrow\n"
            "(see benchmarks/accuracy/requirements.txt). The runner itself needs only "
            "the standard library."
        ) from exc
    return pq.read_table(path).to_pylist()


def gsm8k_answer(answer: str) -> str:
    """The numeric answer after the final ``####`` marker, as written in GSM8K."""
    tail = answer.rsplit("####", 1)[-1].strip()
    return tail.replace(",", "")


def normalize_gsm8k(raw_path: str):
    out = []
    for i, row in enumerate(_read_jsonl(raw_path)):
        out.append(
            {
                "id": f"gsm8k-test-{i:04d}",
                "task": "gsm8k",
                "question": row["question"],
                "answer_text": row["answer"],
                "answer_number": gsm8k_answer(row["answer"]),
            }
        )
    return out


def normalize_mmlu(raw_path: str, prefix: str):
    out = []
    for i, row in enumerate(_parquet_rows(raw_path)):
        out.append(
            {
                "id": f"{prefix}-{i:05d}",
                "task": "mmlu",
                "subject": row["subject"],
                "question": row["question"],
                "choices": list(row["choices"]),
                "answer": "ABCD"[int(row["answer"])],
            }
        )
    return out


HUMANEVAL_EXCLUDED = {
    # The plus test for this problem in the pinned revision unpacks the candidate's
    # float return value (``_poly(*candidate(*inp), inp)``), so no implementation can
    # pass it -- the dataset's own canonical solution fails it as well.  Dropping the
    # item keeps pass@1 meaningful; 163 remain.
    "HumanEval/32"
}


def normalize_humaneval(raw_path: str):
    out = []
    for row in _read_jsonl(raw_path):
        if row["task_id"] in HUMANEVAL_EXCLUDED:
            continue
        out.append(
            {
                "id": row["task_id"],
                "task": "humaneval",
                "prompt": row["prompt"],
                "canonical_solution": row["canonical_solution"],
                "entry_point": row["entry_point"],
                "test": row["test"],
            }
        )
    return out


def build_tasks(cache: str = DEFAULT_CACHE) -> dict:
    """Fetch (if needed) and normalise every task set.  Returns the dataset manifest."""
    sources = fetch(cache)
    tasks_dir = os.path.join(cache, "tasks")
    os.makedirs(tasks_dir, exist_ok=True)
    writers = {
        "gsm8k": ("gsm8k.jsonl", normalize_gsm8k(os.path.join(cache, sources["gsm8k"]["file"]))),
        "mmlu": ("mmlu.jsonl", normalize_mmlu(os.path.join(cache, sources["mmlu_test"]["file"]), "mmlu")),
        "mmlu_dev": ("mmlu_dev.jsonl", normalize_mmlu(os.path.join(cache, sources["mmlu_dev"]["file"]), "mmlu-dev")),
        "humaneval": (
            "humaneval.jsonl",
            normalize_humaneval(os.path.join(cache, sources["humaneval"]["file"])),
        ),
    }
    counts = {}
    for name, (filename, rows) in writers.items():
        with open(os.path.join(tasks_dir, filename), "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        counts[name] = len(rows)
    meta = {
        "sources": sources,
        "counts": counts,
        "tasks_sha256": {
            name: sha256_file(os.path.join(tasks_dir, filename))
            for name, (filename, _) in writers.items()
        },
    }
    with open(os.path.join(cache, "dataset_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
        f.write("\n")
    return meta


def load_task(cache: str, name: str) -> list:
    path = os.path.join(cache, "tasks", f"{name}.jsonl")
    if not os.path.exists(path):
        raise SystemExit(f"task file missing: {path}\nRun `run.py fetch` first.")
    return list(_read_jsonl(path))


def select(rows: list, limit: int | None, seed: int, stratified: bool = False) -> list:
    """Deterministic subset.  MMLU is stratified by subject and taken round-robin so a
    small limit still covers many subjects; everything else is a seeded shuffle."""
    if limit is None or limit >= len(rows):
        return list(rows)
    rng = random.Random(seed)
    if not stratified:
        picked = list(rows)
        rng.shuffle(picked)
        return picked[:limit]
    by_subject: dict[str, list] = {}
    for row in rows:
        by_subject.setdefault(row["subject"], []).append(row)
    for subject in by_subject:
        rng.shuffle(by_subject[subject])
    order = sorted(by_subject)
    rng.shuffle(order)
    picked = []
    round_no = 0
    while len(picked) < limit:
        added = False
        for subject in order:
            if round_no < len(by_subject[subject]):
                picked.append(by_subject[subject][round_no])
                added = True
                if len(picked) == limit:
                    break
        if not added:
            break
        round_no += 1
    return picked


def mmlu_shots(cache: str, subject: str, shots: int) -> list:
    """The first ``shots`` dev examples for a subject, in file order (deterministic)."""
    if shots <= 0:
        return []
    rows = [r for r in load_task(cache, "mmlu_dev") if r["subject"] == subject]
    return rows[:shots]


def write_subset(path: str, rows: list) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
