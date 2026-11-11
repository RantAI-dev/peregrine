"""Peregrine embedding adapter (`peregrine serve` + `/v1/embeddings`).

Written ahead of the endpoint: S4.1 (`origin/s4.1-embeddings`, PR #158) adds
`POST /v1/embeddings` with the OpenAI request/response shape.  Until it merges
this adapter is only exercised from a worktree build of that branch; point
`PEREGRINE_BIN` (or `peregrine_bin`) at the binary and pass
`--peregrine-worktree` to the harness.

The request body is exactly the OpenAI shape, so the same adapter would work
against any engine that speaks it; the point here is that adding Peregrine to
the results is one registry row, not new code.
"""

import os
from pathlib import Path
import subprocess
from typing import Any, Dict, List, Optional

from .embed_base import EmbedAdapter, post_json, wait_ready

DEFAULT_BIN = "target/release/peregrine"


class PeregrineEmbedAdapter(EmbedAdapter):
    """`peregrine serve -m <gguf> --host --port -t N`, then `/v1/embeddings`."""

    name = "peregrine"

    def __init__(
        self,
        threads: int = 4,
        binary: Optional[str] = None,
        worktree: Optional[str] = None,
        slots: int = 40,
    ):
        super().__init__(threads=threads)
        self.worktree = Path(
            worktree
            or os.environ.get("PEREGRINE_WORKTREE")
            or Path(__file__).resolve().parent.parent.parent
        )
        self.binary = Path(
            binary or os.environ.get("PEREGRINE_BIN") or (self.worktree / DEFAULT_BIN)
        )
        self.slots = slots
        self.version = {
            "name": "peregrine",
            "binary": str(self.binary),
            "worktree": str(self.worktree),
        }

    def get_version_info(self) -> Dict[str, Any]:
        info = dict(self.version)
        try:
            info["commit"] = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=str(self.worktree),
                stderr=subprocess.DEVNULL,
            ).decode().strip()
            info["branch"] = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=str(self.worktree),
                stderr=subprocess.DEVNULL,
            ).decode().strip()
        except Exception:
            pass
        try:
            info["server_version"] = subprocess.check_output(
                [str(self.binary), "--version"], stderr=subprocess.STDOUT
            ).decode().strip().splitlines()[0]
        except Exception:
            pass
        return info

    def start(self, row: Dict[str, Any], model: Dict[str, Any], port: int) -> Dict[str, Any]:
        self.slots = int(row.get("slots") or self.slots)
        if not self.binary.exists():
            raise FileNotFoundError(
                f"peregrine binary not found at {self.binary}; build the S4.1 branch "
                "and set PEREGRINE_BIN"
            )
        ctx = int(row.get("_ctx") or model.get("max_tokens") or 512)
        cmd = [
            str(self.binary), "serve",
            "-m", str(Path(row["path"]).expanduser()),
            "--host", "127.0.0.1",
            "--port", str(port),
            "-t", str(self.threads),
            "-c", str(ctx),
            "--slots", str(self.slots),
            "--no-profile",
        ]
        self.port = port
        self.base_url = f"http://127.0.0.1:{port}"
        self.proc = subprocess.Popen(
            cmd,
            cwd=str(self.worktree),
            stdout=self.open_log(Path("/var/tmp") / f"embed-peregrine-{port}.log"),
            stderr=subprocess.STDOUT,
        )
        if not wait_ready(self.base_url, timeout=180.0):
            tail = self.read_log_tail()
            self.stop()
            raise RuntimeError(f"peregrine serve failed to start:\n{tail}")
        return {"cmd": cmd, "threads": self.threads, "slots": self.slots,
                "pooling": model.get("pooling")}

    def embed(self, texts: List[str]) -> Dict[str, Any]:
        reply = post_json(
            self.base_url + "/v1/embeddings",
            {"input": texts, "model": "embed", "encoding_format": "float"},
        )
        rows = sorted(reply["data"], key=lambda d: d.get("index", 0))
        usage = reply.get("usage", {}) or {}
        return {
            "embeddings": [r["embedding"] for r in rows],
            "prompt_tokens": int(usage.get("prompt_tokens", 0)),
        }
