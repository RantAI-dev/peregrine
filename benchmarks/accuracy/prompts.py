"""Prompt construction for the three gate tasks.

The templates are fixed here, not taken from the server, so every engine sees the
same bytes.  All tasks are asked in a single user turn (no system message): Gemma's
template has no system role, and a user turn is the one shape all four templates share.
"""

from __future__ import annotations

GSM8K_MAX_TOKENS = 512
MMLU_MAX_TOKENS = 24
HUMANEVAL_MAX_TOKENS = 512

GSM8K_INSTRUCTION = (
    "Solve the problem step by step. Then write the final numeric answer on its own "
    "line, as:\n#### <number>"
)

MMLU_INSTRUCTION = "Reply with only the letter of the correct answer (A, B, C or D)."

HUMANEVAL_INSTRUCTION = (
    "Complete the following Python function. Return the complete function, including "
    "its imports and definition, inside one ```python code block and nothing else."
)


def mmlu_block(row: dict) -> str:
    """One multiple-choice question, chat style (no few-shot ``Answer:`` continuation).

    A five-shot prompt made the 1B chat models answer the whole list of questions or
    ignore the format; a single question with a format instruction is what chat models
    are aligned to, and it scored higher for every gate model in the protocol check.
    """
    lines = [f"Question: {row['question'].strip()}"]
    for letter, choice in zip("ABCD", row["choices"]):
        lines.append(f"{letter}. {str(choice).strip()}")
    lines.append("")
    lines.append(MMLU_INSTRUCTION)
    return "\n".join(lines)


def gsm8k_messages(row: dict) -> tuple[list, int]:
    text = f"{GSM8K_INSTRUCTION}\n\nProblem: {row['question'].strip()}"
    return [{"role": "user", "content": text}], GSM8K_MAX_TOKENS


def mmlu_messages(row: dict, shots: list | None = None) -> tuple[list, int]:
    return [{"role": "user", "content": mmlu_block(row)}], MMLU_MAX_TOKENS


def humaneval_messages(row: dict) -> tuple[list, int]:
    text = (
        f"{HUMANEVAL_INSTRUCTION}\n\n```python\n{row['prompt'].rstrip()}\n```"
    )
    return [{"role": "user", "content": text}], HUMANEVAL_MAX_TOKENS


def build(task: str, row: dict, shots_for=None) -> tuple[list, int]:
    """Return ``(messages, max_tokens)`` for one task item."""
    if task == "gsm8k":
        return gsm8k_messages(row)
    if task == "mmlu":
        return mmlu_messages(row)
    if task == "humaneval":
        return humaneval_messages(row)
    raise ValueError(f"unknown task {task!r}")
