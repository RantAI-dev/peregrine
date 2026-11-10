#!/usr/bin/env python3
"""OpenAI-API compatibility checks for a Peregrine (or llama-server) endpoint.

Only the Python standard library is used. The script either starts
`peregrine serve` on a model itself, or points at an already-running server:

    tests/compat/compat.py --model /path/to/model.gguf
    tests/compat/compat.py --base-url http://127.0.0.1:8201

A check that is not applicable (a model with no reasoning channel, a server
without /v1/completions) is reported as DEVIATION or SKIP; only a broken
contract is a failure. The exit status is 0 when there is no failure.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import http.client
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Optional

PASS, FAIL, SKIP, DEVIATION = "PASS", "FAIL", "SKIP", "DEVIATION"
results: list[tuple[str, str, str]] = []
BASE_URL = ""
MODEL_ID = ""
ENGINE = "peregrine"
TIMEOUT = 180.0
THINKING = False


class Fail(Exception):
    pass


class Skip(Exception):
    pass


class Deviation(Exception):
    """A known, documented difference from the OpenAI specification."""


def record(status: str, name: str, detail: str = "") -> None:
    results.append((status, name, detail))
    print(f"{status:<9} {name}" + (f" — {detail}" if detail else ""), flush=True)


def run(name: str, fn) -> None:
    """Run one check; exceptions become a FAIL and known differences a DEVIATION."""
    try:
        detail = fn() or ""
        record(PASS, name, detail)
    except Skip as e:
        record(SKIP, name, str(e))
    except Deviation as e:
        record(DEVIATION, name, str(e))
    except Fail as e:
        record(FAIL, name, str(e))
    except Exception as e:  # noqa: BLE001 - a check must never abort the suite
        record(FAIL, name, f"{type(e).__name__}: {e}")
        if os.environ.get("COMPAT_VERBOSE"):
            traceback.print_exc()


def need(cond: bool, msg: str) -> None:
    if not cond:
        raise Fail(msg)


@dataclass
class Resp:
    status: int
    headers: dict[str, str]
    body: bytes

    @property
    def text(self) -> str:
        return self.body.decode("utf-8", "replace")

    @property
    def json(self) -> Any:
        try:
            return json.loads(self.body)
        except Exception:  # noqa: BLE001
            return None

    @property
    def content_type(self) -> str:
        return self.headers.get("content-type", "")


def http_request(
    method: str,
    path: str,
    obj: Any = None,
    raw: Optional[str] = None,
    headers: Optional[dict[str, str]] = None,
    timeout: Optional[float] = None,
) -> Resp:
    data: Optional[bytes] = None
    hdrs = dict(headers or {})
    if raw is not None:
        data = raw.encode()
        hdrs.setdefault("content-type", "application/json")
    elif obj is not None:
        data = json.dumps(obj).encode()
        hdrs.setdefault("content-type", "application/json")
    req = urllib.request.Request(BASE_URL + path, data=data, method=method, headers=hdrs)
    try:
        with urllib.request.urlopen(req, timeout=timeout or TIMEOUT) as r:
            return Resp(r.status, dict(r.headers), r.read())
    except urllib.error.HTTPError as e:
        return Resp(e.code, dict(e.headers), e.read())


def post(path: str, obj: Any, **kw) -> Resp:
    return http_request("POST", path, obj=obj, **kw)


def sse_post(path: str, obj: Any, timeout: Optional[float] = None):
    """POST and read an SSE body. Returns (status, chunks, saw_done, all_data_lines)."""
    u = urllib.parse.urlparse(BASE_URL)
    host = u.hostname or "127.0.0.1"
    port = u.port or (443 if u.scheme == "https" else 80)
    body = json.dumps(obj).encode()
    conn = http.client.HTTPConnection(host, port, timeout=timeout or TIMEOUT)
    try:
        conn.request(
            "POST",
            path,
            body=body,
            headers={"content-type": "application/json", "accept": "text/event-stream"},
        )
        r = conn.getresponse()
        if r.status != 200:
            return r.status, [], False, [r.read().decode("utf-8", "replace")]
        chunks: list[Any] = []
        data_lines: list[str] = []
        done = False
        while True:
            line = r.readline()
            if not line:
                break
            text = line.decode("utf-8", "replace").strip()
            if not text or text.startswith(":") or text.startswith("event:"):
                continue
            if not text.startswith("data:"):
                continue
            data = text[5:].strip()
            data_lines.append(data)
            if data == "[DONE]":
                done = True
                break
            chunks.append(json.loads(data))
        return r.status, chunks, done, data_lines
    finally:
        conn.close()


def chat(body: dict, allow_thinking: bool = False, **kw) -> Resp:
    # A thinking model would spend the small token budgets on reasoning; the API
    # mechanics are checked with thinking off. The reasoning check asks for it.
    if THINKING and not allow_thinking:
        body = dict(body, reasoning_effort="none")
    return post("/v1/chat/completions", body, **kw)


def stream_chat(body: dict, allow_thinking: bool = False, timeout: Optional[float] = None):
    if THINKING and not allow_thinking:
        body = dict(body, reasoning_effort="none")
    return sse_post("/v1/chat/completions", body, timeout=timeout)


def message_of(r: Resp) -> dict:
    d = r.json
    need(isinstance(d, dict), f"response is not a JSON object: {r.text[:300]}")
    choices = d.get("choices")
    need(isinstance(choices, list) and choices, f"no choices: {r.text[:300]}")
    return choices[0]


def content_of(r: Resp) -> str:
    m = message_of(r)
    need(isinstance(m.get("message"), dict), f"choice has no message: {m}")
    c = m["message"].get("content")
    need(isinstance(c, str), f"message.content is not a string: {c!r}")
    return c


# --- checks -----------------------------------------------------------------


def check_health() -> str:
    r = http_request("GET", "/health")
    need(r.status == 200, f"status {r.status}: {r.text[:200]}")
    d = r.json
    need(isinstance(d, dict) and d.get("status") == "ok", f"body {r.text[:200]}")
    return "status ok"


def check_models() -> str:
    global MODEL_ID
    r = http_request("GET", "/v1/models")
    need(r.status == 200, f"status {r.status}: {r.text[:200]}")
    d = r.json
    need(isinstance(d, dict), f"not a JSON object: {r.text[:200]}")
    need(d.get("object") == "list", f"object is {d.get('object')!r}, not 'list'")
    data = d.get("data")
    need(isinstance(data, list) and data, "data is not a non-empty list")
    for m in data:
        need(isinstance(m, dict), "model entry is not an object")
        need(isinstance(m.get("id"), str) and m["id"], "model entry has no string id")
        need(m.get("object") == "model", f"model entry object is {m.get('object')!r}")
    MODEL_ID = data[0]["id"]
    return f"{len(data)} model(s), id={MODEL_ID!r}"


def check_chat_nonstream() -> str:
    body = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": "Reply with the single word: pong"}],
        "max_tokens": 8,
        "temperature": 0,
        "seed": 1,
    }
    r = chat(body)
    need(r.status == 200, f"status {r.status}: {r.text[:300]}")
    d = r.json
    need(isinstance(d, dict), "not a JSON object")
    need(isinstance(d.get("id"), str) and d["id"], "no string id")
    need(d.get("object") == "chat.completion", f"object is {d.get('object')!r}")
    need(d.get("model") == MODEL_ID, f"model is {d.get('model')!r}, expected {MODEL_ID!r}")
    choice = message_of(r)
    need(choice.get("index") == 0, f"choice index is {choice.get('index')!r}")
    need(choice.get("message", {}).get("role") == "assistant", "message.role is not assistant")
    need(isinstance(choice.get("message", {}).get("content"), str), "content is not a string")
    need(
        choice.get("finish_reason") in ("stop", "length"),
        f"finish_reason is {choice.get('finish_reason')!r}",
    )
    usage = d.get("usage")
    need(isinstance(usage, dict), "no usage object")
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        need(isinstance(usage.get(key), int), f"usage.{key} is not an integer")
    need(
        usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"],
        f"usage does not add up: {usage}",
    )
    need(usage["completion_tokens"] > 0, "usage.completion_tokens is 0")
    return f"{usage['completion_tokens']} completion tokens, finish={choice['finish_reason']}"


def check_chat_stream() -> str:
    body = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": "Count from one to five."}],
        "max_tokens": 24,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    status, chunks, done, raw = stream_chat(body)
    need(status == 200, f"status {status}: {''.join(raw)[:300]}")
    need(chunks, "no data chunks")
    need(done, "stream ended without data: [DONE]")
    ids = set()
    content, reasoning = "", ""
    last = None
    usage = None
    finish_at = None
    for i, c in enumerate(chunks):
        need(isinstance(c, dict), f"chunk {i} is not an object")
        need(c.get("object") == "chat.completion.chunk", f"chunk {i} object is {c.get('object')!r}")
        need(isinstance(c.get("id"), str), f"chunk {i} has no id")
        need(c.get("model") == MODEL_ID, f"chunk {i} model is {c.get('model')!r}")
        ids.add(c["id"])
        if isinstance(c.get("usage"), dict):
            usage = c["usage"]
        choices = c.get("choices")
        need(isinstance(choices, list), f"chunk {i} has no choices list")
        if not choices:
            # OpenAI sends one final chunk with empty choices (and usage) when
            # stream_options.include_usage is set. Nothing may follow a finish.
            need(finish_at is not None, f"chunk {i} has no choices before any finish_reason")
            need(i > finish_at, f"chunk {i} has no choices in the middle of the stream")
            continue
        ch = choices[0]
        need(isinstance(ch.get("delta"), dict), f"chunk {i} has no delta object")
        delta = ch["delta"]
        if isinstance(delta.get("content"), str):
            content += delta["content"]
        if isinstance(delta.get("reasoning_content"), str):
            reasoning += delta["reasoning_content"]
        if ch.get("finish_reason") is not None:
            need(finish_at is None, f"finish_reason twice (chunks {finish_at} and {i})")
            need(
                ch["finish_reason"] in ("stop", "length"),
                f"finish_reason is {ch['finish_reason']!r}",
            )
            finish_at = i
            last = ch
    need(last is not None, "no chunk carried finish_reason")
    need(len(ids) == 1, f"chunks carry {len(ids)} different ids")
    need(content or reasoning, "the stream carried no content")
    if not isinstance(usage, dict):
        raise Deviation(
            "the stream has no usage, even with stream_options.include_usage "
            "(OpenAI includes it only when asked)"
        )
    need(
        usage.get("total_tokens")
        == usage.get("prompt_tokens", -1) + usage.get("completion_tokens", -1),
        f"usage does not add up: {usage}",
    )
    return f"{len(chunks)} chunks, {len(content)} chars, finish={last['finish_reason']}"


def check_completions() -> str:
    r = post(
        "/v1/completions",
        {"model": MODEL_ID, "prompt": "Hello", "max_tokens": 4, "temperature": 0},
    )
    if r.status in (404, 405):
        raise Deviation(
            f"/v1/completions is not implemented (HTTP {r.status}); OpenAI and llama-server serve it"
        )
    need(r.status == 200, f"status {r.status}: {r.text[:300]}")
    d = r.json
    need(isinstance(d, dict), "not a JSON object")
    need(d.get("object") == "text_completion", f"object is {d.get('object')!r}")
    choices = d.get("choices")
    need(isinstance(choices, list) and choices, "no choices")
    need(isinstance(choices[0].get("text"), str), "choice.text is not a string")
    return f"{len(choices[0]['text'])} chars"


def check_temperature_zero_determinism() -> str:
    body = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": "Name three colours, separated by commas."}],
        "max_tokens": 24,
        "temperature": 0,
        "seed": 123,
    }
    a, b = chat(body), chat(body)
    need(a.status == 200 and b.status == 200, f"status {a.status}/{b.status}")
    ca, cb = content_of(a), content_of(b)
    need(ca == cb, f"temperature 0 differs: {ca!r} vs {cb!r}")
    return f"identical ({len(ca)} chars)"


def check_seed_determinism() -> str:
    body = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": "Tell me a short story about a bird."}],
        "max_tokens": 32,
        "temperature": 0.9,
        "top_p": 0.95,
        "seed": 7,
    }
    a, b, c = chat(body), chat(body), chat(body)
    need(a.status == 200 and b.status == 200 and c.status == 200, f"status {a.status}/{b.status}/{c.status}")
    ca, cb, cc = content_of(a), content_of(b), content_of(c)
    if not ca:
        raise Skip("the model produced no content")
    if ca != cb:
        if cb == cc:
            raise Deviation(
                "the first request differs, later identical requests reproduce "
                "(llama-server's first-slot sampler state)"
            )
        raise Fail(f"same temperature/top_p/seed differs: {ca!r} vs {cb!r}")
    need(cb == cc, f"the third request differs: {cb!r} vs {cc!r}")
    return f"identical across three requests ({len(ca)} chars)"


def check_stop_sequences() -> str:
    base_body = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": "List the first five natural numbers, separated by commas."}],
        "max_tokens": 32,
        "temperature": 0,
        "seed": 3,
    }
    base = chat(base_body)
    need(base.status == 200, f"baseline status {base.status}")
    baseline = content_of(base)
    if len(baseline) < 8:
        raise Skip(f"baseline too short for a stop string ({len(baseline)} chars)")
    stop = baseline[2:6]
    r = chat(dict(base_body, stop=[stop]))
    if r.status == 400:
        raise Deviation(f"stop is refused: {(r.json or {}).get('error', {}).get('message', r.text[:120])}")
    need(r.status == 200, f"status {r.status}: {r.text[:300]}")
    choice = message_of(r)
    text = content_of(r)
    if stop in text or text == baseline:
        raise Deviation(
            f"the stop field is accepted but ignored: {stop!r} still appears / the reply is not truncated"
        )
    need(baseline.startswith(text), f"content is not a prefix of the unstoppped reply: {text!r}")
    need(len(text) < len(baseline), f"stop did not truncate: {text!r}")
    need(choice.get("finish_reason") == "stop", f"finish_reason is {choice.get('finish_reason')!r}")
    return f"stopped at {stop!r}, {len(text)} chars"


def check_max_tokens() -> str:
    body = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": "Count upwards forever."}],
        "max_tokens": 4,
        "temperature": 0,
        "ignore_eos": True,
    }
    r = chat(body)
    need(r.status == 200, f"status {r.status}: {r.text[:300]}")
    d = r.json
    choice = message_of(r)
    need(
        choice.get("finish_reason") == "length",
        f"finish_reason is {choice.get('finish_reason')!r}, expected 'length'",
    )
    usage = d.get("usage", {})
    need(
        usage.get("completion_tokens") == 4,
        f"completion_tokens is {usage.get('completion_tokens')!r}, expected 4",
    )
    return "4 tokens, finish=length"


def check_n_choices() -> str:
    body = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": "Say a colour."}],
        "max_tokens": 6,
        "temperature": 0.7,
        "seed": 11,
        "n": 3,
    }
    r = chat(body)
    if r.status == 400:
        err = (r.json or {}).get("error", {})
        raise Deviation(f"n > 1 is refused: HTTP 400, {err.get('message', r.text[:120])}")
    need(r.status == 200, f"status {r.status}: {r.text[:300]}")
    choices = r.json.get("choices")
    need(isinstance(choices, list) and len(choices) == 3, f"{len(choices or [])} choices, expected 3")
    indexes = sorted(c.get("index") for c in choices)
    need(indexes == [0, 1, 2], f"choice indexes are {indexes}")
    for c in choices:
        need(isinstance(c.get("message", {}).get("content"), str), "a choice has no content")
    # Streaming n > 1 is refused by Peregrine; that is a documented restriction, not a crash.
    status, chunks, done, raw = stream_chat(dict(body, stream=True))
    if status == 400:
        raise Deviation("n > 1 with stream is refused (OpenAI permits it)")
    need(status == 200, f"n > 1 with stream: status {status}")
    return "3 choices; n with stream returns 200"


def check_logprobs() -> str:
    body = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": "Say hello."}],
        "max_tokens": 5,
        "temperature": 0,
        "logprobs": True,
        "top_logprobs": 3,
    }
    r = chat(body)
    if r.status == 400:
        err = (r.json or {}).get("error", {})
        raise Deviation(f"logprobs refused: {err.get('message', r.text[:120])}")
    need(r.status == 200, f"status {r.status}: {r.text[:300]}")
    choice = message_of(r)
    lp = choice.get("logprobs")
    need(isinstance(lp, dict), "choice.logprobs is not an object")
    entries = lp.get("content")
    need(isinstance(entries, list) and entries, "logprobs.content is not a non-empty list")
    for e in entries:
        need(isinstance(e.get("token"), str), f"entry has no token string: {e!r}")
        value = e.get("logprob")
        need(isinstance(value, (int, float)), f"entry has no numeric logprob: {e!r}")
        need(value <= 1e-6, f"logprob {value} is positive")
    usage = r.json.get("usage", {})
    n = usage.get("completion_tokens")
    need(isinstance(n, int) and len(entries) <= n + 1, f"{len(entries)} entries for {n} tokens")
    missing = []
    if any("bytes" not in e for e in entries):
        missing.append("bytes")
    if any("top_logprobs" not in e for e in entries):
        missing.append("top_logprobs")
    if missing:
        raise Deviation(
            f"logprob entries have token+logprob only; OpenAI also returns {', '.join(missing)}"
        )
    return f"{len(entries)} entries"


def check_errors() -> str:
    notes = []

    # Malformed JSON: 400, and the body is an OpenAI error object, not text/plain.
    r = http_request("POST", "/v1/chat/completions", raw='{"messages":')
    need(r.status == 400, f"malformed JSON: status {r.status}")
    if "application/json" not in r.content_type:
        raise Deviation(f"malformed JSON returns {r.content_type or 'no content-type'}, not JSON")
    err = (r.json or {}).get("error")
    need(isinstance(err, dict), f"malformed JSON has no error object: {r.text[:200]}")
    need(isinstance(err.get("message"), str) and err["message"], "error.message is empty")
    need(err.get("type") == "invalid_request_error", f"error.type is {err.get('type')!r}")
    notes.append("bad JSON -> 400 JSON error")

    # Missing messages.
    r = post("/v1/chat/completions", {})
    need(r.status == 400, f"missing messages: status {r.status}")
    err = (r.json or {}).get("error", {})
    need(isinstance(err, dict), f"missing messages has no error object: {r.text[:200]}")
    need("messages" in err.get("message", "").lower(), f"message does not name messages: {err}")
    notes.append("missing messages -> 400")

    # messages not an array.
    r = post("/v1/chat/completions", {"messages": "hello"})
    need(r.status == 400, f"messages as a string: status {r.status}")
    notes.append("messages not an array -> 400")

    # An empty list has no last user message.
    r = post("/v1/chat/completions", {"messages": []})
    need(r.status == 400, f"empty messages: status {r.status}")
    notes.append("empty messages -> 400")

    # The last turn must be the user's.
    r = post(
        "/v1/chat/completions",
        {"messages": [{"role": "assistant", "content": "hi"}]},
    )
    need(r.status == 400, f"assistant last turn: status {r.status}")
    notes.append("last turn not user -> 400")

    # A model id that is not the served one. OpenAI answers 404 model_not_found;
    # Peregrine documents that it accepts any id, so record the difference.
    r = post(
        "/v1/chat/completions",
        {
            "model": "definitely-not-the-served-model",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1,
        },
    )
    if r.status == 200:
        notes.append("unknown model id accepted (documented)")
    else:
        need(r.status == 404, f"unknown model: status {r.status}")
        err = (r.json or {}).get("error", {})
        need(
            err.get("code") == "model_not_found" or "model" in err.get("message", "").lower(),
            f"unknown model error does not name the model: {err}",
        )
        notes.append("unknown model -> 404")

    # A sampling control Peregrine does not implement must be refused by name, never ignored.
    r = post(
        "/v1/chat/completions",
        {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 1, "mirostat": 2},
    )
    if r.status != 400:
        notes.append(f"unknown sampling field mirostat accepted ({r.status})")
    else:
        msg = (r.json or {}).get("error", {}).get("message", "")
        need("mirostat" in msg, f"refusal does not name mirostat: {msg!r}")
        notes.append("unsupported sampler -> 400 by name")

    # n above the documented maximum.
    r = post(
        "/v1/chat/completions",
        {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 1, "n": 99},
    )
    if r.status != 400:
        notes.append(f"n=99 accepted ({r.status})")
    else:
        notes.append("n > 16 -> 400")

    # An unknown path is a 404, not a 200.
    r = http_request("GET", "/v1/does-not-exist")
    need(r.status == 404, f"unknown path: status {r.status}")
    # The wrong method on a known path is 405.
    r = http_request("GET", "/v1/chat/completions")
    need(r.status == 405, f"GET on chat completions: status {r.status}")
    notes.append("unknown path 404, wrong method 405")
    return "; ".join(notes)


def check_concurrency() -> str:
    def one(i: int) -> tuple[int, float]:
        started = time.time()
        r = chat(
            {
                "model": MODEL_ID,
                "messages": [{"role": "user", "content": f"Say the number {i}."}],
                "max_tokens": 8,
                "temperature": 0.3,
                "seed": 100 + i,
            },
            timeout=180,
        )
        return r.status, time.time() - started

    started = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(one, range(8)))
    elapsed = time.time() - started
    bad = [(s, t) for s, t in outcomes if s != 200]
    need(not bad, f"non-200 responses: {bad}")
    need(elapsed < 180, f"8 concurrent requests took {elapsed:.1f}s")
    health = http_request("GET", "/health")
    need(health.status == 200, "the server is unhealthy after the burst")
    slowest = max(t for _, t in outcomes)
    return f"8/8 in {elapsed:.1f}s (slowest {slowest:.1f}s)"


def check_metrics() -> str:
    r = http_request("GET", "/metrics")
    need(r.status == 200, f"status {r.status}")
    text = r.text
    need("llamacpp:requests_processing" in text, "no llama-server-compatible queue gauge")
    return f"{len(text.splitlines())} metric lines"


def check_reasoning(thinking: bool) -> str:
    if not thinking:
        raise Skip("no thinking model supplied (--thinking)")
    prompt = "What is 17 * 23? Think it through, then answer."
    body = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 128,
        "temperature": 0,
        "reasoning_format": "auto",
    }
    r = chat(body, allow_thinking=True)
    need(r.status == 200, f"status {r.status}: {r.text[:300]}")
    msg = message_of(r).get("message", {})
    reasoning = msg.get("reasoning_content")
    content = msg.get("content")
    need(isinstance(reasoning, str), "message.reasoning_content is not a string")
    need(isinstance(content, str), "message.content is not a string")
    if not reasoning:
        raise Skip("the model produced no reasoning tokens")
    for field, value in (("content", content), ("reasoning_content", reasoning)):
        need("<think>" not in value and "</think>" not in value, f"{field} contains a think tag")

    # `none` leaves the analysis in content, still with no control tokens.
    r = chat(dict(body, reasoning_format="none"), allow_thinking=True)
    need(r.status == 200, f"reasoning_format none: status {r.status}")
    msg = message_of(r).get("message", {})
    need(msg.get("reasoning_content", "") == "", "reasoning_format none still fills reasoning_content")
    need(isinstance(msg.get("content"), str) and msg["content"], "reasoning_format none produced no content")
    if "<think>" in msg["content"]:
        raise Deviation(
            "reasoning_format none leaves the <think> tags in content "
            "(llama-server's meaning of none; Peregrine consumes them)"
        )

    # Streaming splits the same way: reasoning deltas before content deltas, no tags.
    status, chunks, done, raw = sse_post("/v1/chat/completions", dict(body, stream=True))
    need(status == 200, f"stream status {status}: {''.join(raw)[:200]}")
    need(done, "reasoning stream ended without [DONE]")
    r_text = "".join(
        c["choices"][0]["delta"].get("reasoning_content", "")
        for c in chunks
        if c.get("choices")
    )
    c_text = "".join(
        c["choices"][0]["delta"].get("content", "")
        for c in chunks
        if c.get("choices")
    )
    need(r_text, "the reasoning stream sent no reasoning_content deltas")
    need("<think>" not in r_text + c_text, "the reasoning stream leaks a think tag")
    return f"{len(reasoning)} reasoning chars; none/auto/stream split"


# --- server process ---------------------------------------------------------


@contextlib.contextmanager
def server(args):
    if args.base_url:
        yield args.base_url
        return
    if not args.model:
        sys.exit("compat: --model is required unless --base-url is given")
    if not os.path.isfile(args.model):
        sys.exit(f"compat: model not found: {args.model}")
    bin_path = args.peregrine_bin
    if not os.path.isfile(bin_path):
        found = shutil.which("peregrine")
        if not found:
            sys.exit(f"compat: peregrine binary not found: {bin_path} (--peregrine-bin)")
        bin_path = found
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    log = tempfile.NamedTemporaryFile("w+", prefix="peregrine-compat-", suffix=".log", delete=False)
    cmd = [
        bin_path,
        "serve",
        "-m",
        args.model,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "-t",
        str(args.threads),
        "-c",
        str(args.ctx),
        "--slots",
        str(args.slots),
    ]
    base = f"http://127.0.0.1:{port}"
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        deadline = time.time() + 180
        while time.time() < deadline:
            if proc.poll() is not None:
                log.flush()
                tail = open(log.name).read()[-4000:]
                sys.exit(f"compat: server exited ({proc.returncode}):\n{tail}")
            try:
                with urllib.request.urlopen(base + "/health", timeout=2) as r:
                    if r.status == 200:
                        break
            except Exception:  # noqa: BLE001
                time.sleep(0.25)
        else:
            sys.exit("compat: server did not become healthy in 180 s")
        print(f"# server: {base} ({' '.join(cmd)})", flush=True)
        yield base
    finally:
        if proc.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        log.close()
        if os.environ.get("COMPAT_VERBOSE"):
            print(open(log.name).read(), file=sys.stderr)
        with contextlib.suppress(OSError):
            os.unlink(log.name)


def main() -> int:
    global BASE_URL, ENGINE, TIMEOUT, THINKING
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", help="an already-running server (instead of starting one)")
    ap.add_argument("--model", help="GGUF model to serve")
    ap.add_argument("--peregrine-bin", default="./target/release/peregrine")
    ap.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 4))
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--slots", type=int, default=4)
    ap.add_argument("--engine", default="peregrine", choices=["peregrine", "llama-server"])
    ap.add_argument("--thinking", action="store_true", help="the served model has a reasoning channel")
    ap.add_argument("--timeout", type=float, default=180.0, help="per-request timeout in seconds")
    ap.add_argument("--json-out", help="write the results as JSON to this path")
    args = ap.parse_args()
    ENGINE = args.engine
    TIMEOUT = args.timeout
    THINKING = args.thinking

    with server(args) as base:
        BASE_URL = base
        run("GET /health", check_health)
        run("GET /v1/models", check_models)
        run("chat completions (non-stream)", check_chat_nonstream)
        run("chat completions (SSE stream)", check_chat_stream)
        run("completions endpoint", check_completions)
        run("temperature 0 determinism", check_temperature_zero_determinism)
        run("temperature/top_p/seed determinism", check_seed_determinism)
        run("stop sequences", check_stop_sequences)
        run("max_tokens and finish_reason=length", check_max_tokens)
        run("n > 1", check_n_choices)
        run("logprobs", check_logprobs)
        run("error shapes", check_errors)
        run("8 concurrent requests", check_concurrency)
        run("metrics", check_metrics)
        run("reasoning split", lambda: check_reasoning(args.thinking))

    counts = {s: sum(1 for r in results if r[0] == s) for s in (PASS, FAIL, SKIP, DEVIATION)}
    print()
    print(
        f"# {ENGINE}: {counts[PASS]} passed, {counts[FAIL]} failed, "
        f"{counts[DEVIATION]} deviations, {counts[SKIP]} skipped",
        flush=True,
    )
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(
                {
                    "engine": ENGINE,
                    "base_url": BASE_URL,
                    "model": MODEL_ID,
                    "counts": counts,
                    "results": [
                        {"status": s, "check": n, "detail": d} for s, n, d in results
                    ],
                },
                f,
                indent=2,
            )
            f.write("\n")
    return 1 if counts[FAIL] else 0


if __name__ == "__main__":
    sys.exit(main())
