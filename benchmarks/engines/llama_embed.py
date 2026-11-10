"""llama.cpp embedding adapter (`llama-server --embedding`)."""

import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Dict, List, Optional

from .embed_base import EmbedAdapter, post_json, wait_ready

DEFAULT_SERVER = "/var/tmp/llama.cpp/build/bin/llama-server"
PINNED_TAG = "v0.9.0"
PINNED_COMMIT = "cd959ba"


class LlamaCppEmbedAdapter(EmbedAdapter):
    """Pinned llama.cpp, `--embedding`, one server per (model, precision) row."""

    name = "llama_cpp"

    def __init__(
        self,
        threads: int = 4,
        server_bin: Optional[str] = None,
        flash_attn: str = "auto",
    ):
        super().__init__(threads=threads)
        self.server_bin = Path(
            server_bin or os.environ.get("LLAMA_SERVER_BIN") or DEFAULT_SERVER
        )
        self.flash_attn = flash_attn
        self.version = {
            "name": "llama.cpp",
            "version": PINNED_TAG,
            "commit": PINNED_COMMIT,
            "repo": "https://github.com/ggml-org/llama.cpp",
            "binary": str(self.server_bin),
            "flash_attn": flash_attn,
        }

    def start(self, row: Dict[str, Any], model: Dict[str, Any], port: int) -> Dict[str, Any]:
        if not self.server_bin.exists():
            raise FileNotFoundError(f"llama-server not found at {self.server_bin}")
        model_path = Path(row["path"]).expanduser()
        ctx = int(row.get("server_ctx") or model.get("server_ctx") or 70000)
        batch = int(row.get("batch") or 8192)
        ubatch = int(row.get("ubatch") or 2048)
        # An embedding model has no KV cache worth the name, but n_ctx must hold
        # the whole request (batch x sequence length); llama.cpp caps n_batch at
        # n_ubatch for embeddings, so large requests are fed in ubatches.
        cmd = [
            str(self.server_bin),
            "-m", str(model_path),
            "--embedding",
            "--pooling", model.get("pooling", "mean"),
            "-t", str(self.threads),
            "-c", str(ctx),
            "-b", str(batch),
            "-ub", str(ubatch),
            "-np", "1",
            "-fa", self.flash_attn,
            "--host", "127.0.0.1",
            "--port", str(port),
        ]
        self.port = port
        self.base_url = f"http://127.0.0.1:{port}"
        self.proc = subprocess.Popen(
            cmd,
            stdout=self.open_log(Path("/var/tmp") / f"embed-llama-{port}.log"),
            stderr=subprocess.STDOUT,
        )
        if not wait_ready(self.base_url, timeout=180.0):
            tail = self.read_log_tail()
            self.stop()
            raise RuntimeError(f"llama-server failed to start:\n{tail}")
        return {"cmd": cmd, "pooling": model.get("pooling"), "threads": self.threads}

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

    def stop(self, timeout: float = 20.0) -> None:
        super().stop(timeout=timeout)
