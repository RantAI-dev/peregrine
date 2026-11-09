"""Scorers for the three gate tasks."""

from __future__ import annotations

import re

from sandbox import run_humaneval

NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
LETTER_RE = re.compile(r"[Aa]nswer\s*(?:is|:|=)?\s*\(?([A-D])\b")
LEADING_LETTER_RE = re.compile(r"^\s*\(?([A-D])\b")


def last_number(text: str) -> float | None:
    found = NUMBER_RE.findall(text)
    if not found:
        return None
    try:
        return float(found[-1].replace(",", ""))
    except ValueError:
        return None


def score_gsm8k(item: dict, output: str) -> dict:
    """The last number in the reply against the dataset's ``####`` answer."""
    got = last_number(output)
    try:
        want = float(str(item["answer_number"]).replace(",", ""))
    except ValueError:
        return {"correct": False, "reason": "unparseable reference answer", "got": got, "want": item["answer_number"]}
    if got is None:
        return {"correct": False, "reason": "no number in reply", "got": None, "want": want}
    correct = abs(got - want) <= 1e-4 * max(1.0, abs(want))
    return {"correct": correct, "got": got, "want": want}


def extract_letter(text: str) -> str | None:
    """The option letter a multiple-choice reply picked.

    Chat models answer either with the bare letter or with the choice text
    (``D. demographic momentum``, ``The correct answer is **D**``).  The letter after an
    explicit "answer is / answer:" wins; otherwise the first standalone A-D in the
    reply.  Returns ``None`` when the reply has no letter.
    """
    match = LETTER_RE.search(text)
    if match:
        return match.group(1)
    match = LEADING_LETTER_RE.search(text)
    if match:
        return match.group(1)
    match = re.search(r"\b([A-D])\b", text[:400])
    if match:
        return match.group(1)
    return None


def score_mmlu(item: dict, output: str) -> dict:
    got = extract_letter(output)
    return {"correct": got == item["answer"], "got": got, "want": item["answer"]}


def score_humaneval(item: dict, output: str, *, python: str, timeout: float = 10.0) -> dict:
    code = extract_code_or_empty(output)
    if not code.strip():
        return {"correct": False, "reason": "no code in reply", "got": None, "want": item["entry_point"]}
    run = run_humaneval(code, item["test"], item["entry_point"], python=python, timeout=timeout)
    return {
        "correct": bool(run["passed"]),
        "got": "pass" if run["passed"] else "fail",
        "want": "pass",
        "reason": None if run["passed"] else (run.get("stderr") or run.get("stdout") or "")[-400:],
        "duration_s": run["duration_s"],
    }


def extract_code_or_empty(text: str) -> str:
    from sandbox import extract_code

    return extract_code(text)


def score(task: str, item: dict, output: str, *, python: str | None = None, timeout: float = 10.0) -> dict:
    if task == "gsm8k":
        return score_gsm8k(item, output)
    if task == "mmlu":
        return score_mmlu(item, output)
    if task == "humaneval":
        if not python:
            raise ValueError("humaneval scoring needs the sandbox interpreter")
        return score_humaneval(item, output, python=python, timeout=timeout)
    raise ValueError(f"unknown task {task!r}")
