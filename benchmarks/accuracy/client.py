"""Minimal OpenAI-compatible chat client (standard library only).

Speaks the subset every engine here implements: ``POST /v1/chat/completions`` with
``stream: false`` and greedy sampling.  The request body is deliberately small --
Peregrine rejects unknown sampling fields instead of ignoring them, so no ``seed``,
``top_k`` or vendor extensions are sent.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request


class ServerError(RuntimeError):
    def __init__(self, message: str, status: int | None = None, body: str = ""):
        super().__init__(message)
        self.status = status
        self.body = body


class Client:
    def __init__(self, url: str, model: str | None = None, timeout: float = 900.0, retries: int = 2):
        self.url = url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.retries = retries

    def _post(self, path: str, body: dict, timeout: float) -> dict:
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            self.url + path,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))

    def chat(
        self,
        messages: list,
        max_tokens: int,
        *,
        stop: list | None = None,
        enable_thinking: bool | None = False,
        extra: dict | None = None,
    ) -> dict:
        """One greedy completion.  Returns the raw response plus a wall-clock time."""
        body = {
            "model": self.model or "default",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": False,
        }
        if stop:
            body["stop"] = stop
        if enable_thinking is not None:
            body["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
        if extra:
            body.update(extra)

        last: Exception | None = None
        for attempt in range(self.retries + 1):
            started = time.monotonic()
            try:
                raw = self._post("/v1/chat/completions", body, self.timeout)
                return {
                    "response": raw,
                    "latency_s": time.monotonic() - started,
                }
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")[:4000]
                last = ServerError(f"HTTP {exc.code}: {detail}", exc.code, detail)
                if exc.code < 500:
                    break
            except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as exc:
                last = exc
            if attempt < self.retries:
                time.sleep(1.5 * (attempt + 1))
        raise ServerError(f"request failed after {self.retries + 1} attempts: {last}")

    def info(self) -> dict:
        """Best-effort server identification: llama.cpp's ``/props`` or Peregrine's metrics."""
        out: dict = {"url": self.url}
        for path, key in (("/props", "props"),):
            try:
                with urllib.request.urlopen(self.url + path, timeout=10) as r:
                    out[key] = json.loads(r.read().decode("utf-8"))
            except Exception:
                pass
        try:
            with urllib.request.urlopen(self.url + "/v1/models", timeout=10) as r:
                out["models"] = json.loads(r.read().decode("utf-8"))
        except Exception:
            pass
        try:
            with urllib.request.urlopen(self.url + "/health", timeout=5) as r:
                out["health"] = r.read().decode("utf-8")[:200]
        except Exception:
            pass
        return out


def message_text(response: dict) -> str:
    try:
        return response["choices"][0]["message"].get("content") or ""
    except (KeyError, IndexError, TypeError):
        return ""


def reasoning_text(response: dict) -> str:
    try:
        return response["choices"][0]["message"].get("reasoning_content") or ""
    except (KeyError, IndexError, TypeError):
        return ""
