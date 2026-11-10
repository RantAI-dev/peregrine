"""fastembed (Python, ONNX Runtime) embedding adapter."""

import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List, Optional

from .embed_base import EmbedAdapter, post_json, wait_ready

DEFAULT_VENV = "/var/tmp/embed-venv"
PINNED_VERSION = "0.9.1"


class FastEmbedAdapter(EmbedAdapter):
    """Starts `servers/fastembed_server.py` from the benchmark venv."""

    name = "fastembed"

    def __init__(self, threads: int = 4, venv: Optional[str] = None):
        super().__init__(threads=threads)
        self.venv = Path(venv or os.environ.get("EMBED_VENV") or DEFAULT_VENV)
        self.python = self.venv / "bin" / "python"
        self.server_script = Path(__file__).resolve().parent / "servers" / "fastembed_server.py"
        self.version = {
            "name": "fastembed",
            "version": PINNED_VERSION,
            "runtime": "onnxruntime",
            "venv": str(self.venv),
            "python": str(self.python),
        }

    def start(self, row: Dict[str, Any], model: Dict[str, Any], port: int) -> Dict[str, Any]:
        model_path = Path(row["path"]).expanduser()
        model_name = row.get("fastembed_name") or model.get("fastembed_name")
        cmd = [
            str(self.python), str(self.server_script),
            "--model-name", model_name,
            "--model-path", str(model_path),
            "--port", str(port),
            "--threads", str(self.threads),
        ]
        self.port = port
        self.base_url = f"http://127.0.0.1:{port}"
        self.proc = subprocess.Popen(
            cmd,
            stdout=self.open_log(Path("/var/tmp") / f"embed-fastembed-{port}.log"),
            stderr=subprocess.STDOUT,
        )
        if not wait_ready(self.base_url, timeout=300.0, paths=("/health",)):
            tail = self.read_log_tail()
            self.stop()
            raise RuntimeError(f"fastembed server failed to start:\n{tail}")
        return {"cmd": cmd, "pooling": model.get("pooling"), "threads": self.threads}

    def embed(self, texts: List[str]) -> Dict[str, Any]:
        reply = post_json(self.base_url + "/embed", {"texts": texts})
        return {
            "embeddings": reply["embeddings"],
            "prompt_tokens": int(reply.get("tokens", 0)),
        }
