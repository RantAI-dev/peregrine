"""Base classes and HTTP helpers for embedding benchmark engine adapters.

Parallel to `base.py` (the generation harness), but for the embedding shape:
a request is a list of texts, the answer is one vector per text.  Adapters only
need to start a server, send a batch and report how many tokens it consumed.

Standard library only; heavy dependencies live in the server scripts under
`engines/servers/`, which run from a venv outside the repository.
"""

from abc import ABC, abstractmethod
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional


def post_json(url: str, payload: Dict[str, Any], timeout: float = 900.0) -> Dict[str, Any]:
    """POST a JSON payload and decode the JSON reply."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get_json(url: str, timeout: float = 5.0) -> Optional[Dict[str, Any]]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            if resp.status == 200:
                return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None
    return None


def wait_ready(url: str, timeout: float = 180.0, paths=("/health", "/v1/models"),
               proc: "subprocess.Popen | None" = None) -> bool:
    """Poll liveness endpoints until the server answers 200 or the timeout expires.

    If `proc` is given, a died process fails the wait immediately instead of
    burning the whole timeout (llama-server aborts on some flag combinations).
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            return False
        for p in paths:
            try:
                with urllib.request.urlopen(url.rstrip("/") + p, timeout=2.0) as r:
                    if r.status == 200:
                        return True
            except Exception:
                pass
        time.sleep(0.25)
    return False


class EmbedAdapter(ABC):
    """One engine, one model file, one process at a time."""

    name = "base"

    def __init__(self, threads: int = 4):
        self.threads = threads
        self.proc: Optional[subprocess.Popen] = None
        self.base_url = ""
        self.port = 0
        self._log_path: Optional[Path] = None
        self.version: Dict[str, Any] = {}

    # -- lifecycle ---------------------------------------------------------
    @abstractmethod
    def start(self, row: Dict[str, Any], model: Dict[str, Any], port: int) -> Dict[str, Any]:
        """Start the engine for (row, model) and block until it is ready.

        `row` is the registry entry (engine/precision/path); `model` is the
        model record (pooling, max_tokens, tokenizer).  Returns the exact
        command (as a list of strings) and any notes for the raw JSON.
        """

    @abstractmethod
    def embed(self, texts: List[str]) -> Dict[str, Any]:
        """Embed a batch of texts; returns {embeddings, prompt_tokens}."""

    def stop(self, timeout: float = 20.0) -> None:
        if self.proc is None:
            return
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self.proc = None

    # -- helpers -----------------------------------------------------------
    def open_log(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._log_path = path
        return open(path, "w", encoding="utf-8")

    def read_log_tail(self, lines: int = 20) -> str:
        if self._log_path is None or not self._log_path.exists():
            return ""
        try:
            data = self._log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            return "\n".join(data[-lines:])
        except Exception:
            return ""

    def get_version_info(self) -> Dict[str, Any]:
        return dict(self.version)
