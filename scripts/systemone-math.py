#!/usr/bin/env python3
"""Generate the answer-math agreement fixture for `crates/peregrine-core/src/decision.rs`.

The Rust module is a transcription of llama.cpp v0.9.0's
`server_decision_context::format_answer`, `get_temperature` and the confidence helpers. This script
is an independent implementation of the same formulas in Python's `f64`, written from
`docs/reference/systemone.md` §2 rather than from the Rust, and it writes the cases and the answers
it computes to `crates/peregrine-core/tests/fixtures/systemone-math.jsonl`. The Rust test
`crates/peregrine-core/tests/systemone_math.rs` reads the file back and compares every number to
1e-12.

Run it from anywhere:

    python3 scripts/systemone-math.py

Every value is written as the exact decimal of the `f32` the file would carry (scores, temperatures
and biases are `f32` in llama.cpp), so the Rust reader's `f64` is the same number Python computed
with.
"""

from __future__ import annotations

import json
import math
import random
import struct
from pathlib import Path

OUT = (
    Path(__file__).resolve().parents[1]
    / "crates/peregrine-core/tests/fixtures/systemone-math.jsonl"
)

LEV_N_RATINGS = 9


def f32(x: float) -> float:
    """The f32 value the GGUF and llama.cpp would carry, widened to a Python float."""
    return struct.unpack("<f", struct.pack("<f", x))[0]


def n_outputs(decision: str, question: dict) -> int:
    if decision == "lev" and question["type"] == "noul":
        return LEV_N_RATINGS
    return len(question["keys"])


def n_variants(decision: str, question: dict) -> int:
    if decision == "lev" and question["type"] == "choice" and len(question["keys"]) > 1:
        return 2
    return 1


def get_temperature(decision: str, question: dict, temperatures: dict) -> float:
    n = n_outputs(decision, question)
    type_name = question["type"]
    if decision == "lev":
        bucket = "small" if n <= 8 else "mid" if n <= 26 else "large"
    else:
        bucket = "2" if n <= 2 else "3_5" if n <= 5 else "6_10" if n <= 10 else "11"
    for name in (f"{type_name}.{bucket}", type_name):
        if name in temperatures:
            return temperatures[name]
    return 1.0


def probabilities(decision: str, question: dict, scores: list[list[float]],
                  score_bias: dict[int, list[float]], temperature: float) -> list[float]:
    n = n_outputs(decision, question)
    probs = [0.0] * n
    for v, s in enumerate(scores):
        assert len(s) == n
        logits = [float(x) for x in s]
        if question["type"] == "score" and n in score_bias:
            for i in range(n):
                logits[i] += score_bias[n][i]
        logit_max = max(logits)
        p = [0.0] * n
        total = 0.0
        for i in range(n):
            p[i] = math.exp((logits[i] - logit_max) / temperature)
            total += p[i]
        for i in range(n):
            # the second variant is in the reverse order
            probs[i if v == 0 else n - 1 - i] += p[i] / total / len(scores)
    return probs


def confidence_choice(probs: list[float]) -> float:
    if len(probs) < 2:
        return 1.0
    uniform = 1.0 / len(probs)
    return max(0.0, (max(probs) - uniform) / (1.0 - uniform))


def confidence_score(probs: list[float]) -> float:
    if len(probs) < 2:
        return 1.0
    n = len(probs)
    mode = max(range(n), key=lambda i: probs[i])
    dist = 0.0
    dist_uniform = 0.0
    for i in range(n):
        dist += probs[i] * abs(float(i) - float(mode))
        dist_uniform += abs(float(i) - (n - 1) / 2.0) / n
    return max(0.0, 1.0 - dist / dist_uniform)


def confidence_entropy(probs: list[float]) -> float:
    if len(probs) < 2:
        return 1.0
    entropy = 0.0
    for p in probs:
        if p > 0.0:
            entropy -= p * math.log(p)
    return max(0.0, min(1.0, 1.0 - entropy / math.log(float(len(probs)))))


def get_confidence(decision: str, question: dict, probs: list[float]) -> float:
    if decision in ("vllm-decision2", "vllm-d3"):
        return confidence_entropy(probs)
    if decision in ("lfm3-d1", "lfm3-d1-omni"):
        return max(probs)
    if question["type"] == "choice":
        return confidence_choice(probs)
    return confidence_score(probs)


def format_answer(decision: str, question: dict, scores: list[list[float]],
                  temperatures: dict[str, float],
                  score_bias: dict[int, list[float]]) -> dict:
    n = n_outputs(decision, question)
    temperature = get_temperature(decision, question, temperatures)
    probs = probabilities(decision, question, scores, score_bias, temperature)
    kind = question["type"]

    if kind == "noul":
        if decision == "lev":
            expected = 0.0
            for i in range(n):
                expected += probs[i] * i / (n - 1)
            return {"type": "noul", "noul": expected}
        for i, key in enumerate(question["keys"]):
            if key == "true":
                return {"type": "noul", "noul": probs[i]}
        raise AssertionError("a noul question has no true option")

    probabilities_by_key = [
        [question["keys"][i], probs[i]] for i in range(n)
    ]
    if kind == "choice":
        best = max(range(n), key=lambda i: probs[i])
        return {
            "type": "choice",
            "choice": question["keys"][best],
            "probabilities": probabilities_by_key,
            "confidence": get_confidence(decision, question, probs),
        }
    expected = 0.0
    for i in range(n):
        expected += i * probs[i]
    legend = [
        [question["keys"][i], question["descriptions"][i]] for i in range(n)
    ]
    return {
        "type": "score",
        "score": expected,
        "legend": legend,
        "probabilities": probabilities_by_key,
        "confidence": get_confidence(decision, question, probs),
    }


def make_question(rng: random.Random, kind: str, n: int) -> dict:
    if kind == "choice":
        keys = [f"opt{i}" for i in range(n)]
    elif kind == "score":
        keys = [str(i) for i in range(n)]
    else:
        keys = ["false", "true"]
    descriptions = [f"level {i}" for i in range(len(keys))]
    return {"type": kind, "keys": keys, "descriptions": descriptions}


def make_scores(rng: random.Random, n: int, variants: int) -> list[list[float]]:
    style = rng.randrange(4)
    out = []
    for _ in range(variants):
        if style == 0:
            # A dominant option: a confidence near 1.
            s = [f32(rng.uniform(-0.2, 0.2)) for _ in range(n)]
            s[rng.randrange(n)] += f32(rng.uniform(2.0, 8.0))
        elif style == 1:
            # Uniform: confidence 0.
            s = [f32(0.25)] * n
        elif style == 2:
            # A gentle preference.
            s = [f32(rng.uniform(-1.0, 1.0)) for _ in range(n)]
        else:
            s = [f32(rng.gauss(0.0, 1.5)) for _ in range(n)]
        # Every score is written as the f32 the file carries; the dominant style adds in Python's
        # f64 above, so round the whole vector once more.
        out.append([f32(x) for x in s])
    return out


def make_temperatures(rng: random.Random, kind: str, n: int, lev: bool) -> dict[str, str]:
    """The keys a file might carry, as the exact text of their f32 value."""
    out: dict[str, str] = {}
    if lev:
        bucket = "small" if n <= 8 else "mid" if n <= 26 else "large"
    else:
        bucket = "2" if n <= 2 else "3_5" if n <= 5 else "6_10" if n <= 10 else "11"
    style = rng.randrange(5)
    if style in (1, 2):
        out[f"{kind}.{bucket}"] = repr(f32(rng.uniform(0.2, 4.0)))
    if style == 2:
        out[kind] = repr(f32(rng.uniform(0.2, 4.0)))
    if style == 3:
        out[kind] = repr(f32(rng.uniform(0.2, 4.0)))
    if style == 4:
        # A key for a bucket this question is not in: it must be ignored.
        other = "11" if bucket != "11" else "2"
        out[f"{kind}.{other}"] = repr(f32(rng.uniform(0.2, 4.0)))
    return out


def make_bias(rng: random.Random, kind: str, n: int) -> dict[str, list[float]]:
    if kind != "score" or rng.randrange(2) == 0:
        return {}
    return {str(n): [repr(f32(rng.uniform(-1.5, 1.5))) for _ in range(n)]}


def main() -> None:
    rng = random.Random(20261110)
    cases = []

    decision_types = [
        "openjev",
        "kev",
        "nimble",
        "vllm-decision2",
        "vllm-d3",
        "lfm2-d1",
        "granite-switch",
        "h2o-lightning",
        "lfm3-d1",
        "lfm3-d1-omni",
    ]
    plans = [
        # (decision, kind, option counts) — every temperature bucket edge.
        *[("lev", "choice", n) for n in (2, 3, 8, 9, 26, 27)],
        *[("lev", "score", n) for n in (2, 5, 9, 27)],
        *[("lev", "noul", 2) for _ in range(3)],
        *[("openjev", "choice", n) for n in (2, 3, 5, 6, 10, 11)],
        *[("openjev", "score", n) for n in (2, 4, 10)],
        *[("openjev", "noul", 2) for _ in range(3)],
    ]
    for decision, kind, n in plans:
        q = make_question(rng, kind, n)
        variants = n_variants(decision, q)
        temperatures = make_temperatures(rng, kind, n_outputs(decision, q), decision == "lev")
        bias = make_bias(rng, kind, n_outputs(decision, q))
        scores = make_scores(rng, n_outputs(decision, q), variants)
        cases.append(
            {
                "decision": decision,
                "question": q,
                "temperatures": temperatures,
                "score_bias": bias,
                "scores": scores,
            }
        )

    # The alternative confidence formulas, and a few random cases over every type.
    for decision in decision_types:
        for kind, n in [("choice", 3), ("score", 5)]:
            q = make_question(rng, kind, n)
            temperatures = make_temperatures(rng, kind, n, decision == "lev")
            scores = make_scores(rng, n_outputs(decision, q), 1)
            cases.append(
                {
                    "decision": decision,
                    "question": q,
                    "temperatures": temperatures,
                    "score_bias": {},
                    "scores": scores,
                }
            )
    while len(cases) < 40:
        decision = rng.choice(decision_types + ["lev"])
        kind = rng.choice(["choice", "score", "noul"])
        n = rng.choice([2, 3, 5, 6, 8, 10, 11, 26])
        q = make_question(rng, kind, n)
        variants = n_variants(decision, q)
        temperatures = make_temperatures(rng, kind, n_outputs(decision, q), decision == "lev")
        bias = make_bias(rng, kind, n_outputs(decision, q))
        scores = make_scores(rng, n_outputs(decision, q), variants)
        cases.append(
            {
                "decision": decision,
                "question": q,
                "temperatures": temperatures,
                "score_bias": bias,
                "scores": scores,
            }
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for case in cases:
            temperatures = {k: float(v) for k, v in case["temperatures"].items()}
            bias = {int(k): [float(x) for x in v] for k, v in case["score_bias"].items()}
            case["answer"] = format_answer(
                case["decision"], case["question"], case["scores"], temperatures, bias
            )
            case["temperatures"] = temperatures
            case["score_bias"] = {str(k): v for k, v in bias.items()}
            f.write(json.dumps(case, sort_keys=True, separators=(",", ":")) + "\n")
    print(f"wrote {len(cases)} cases to {OUT}")


if __name__ == "__main__":
    main()
