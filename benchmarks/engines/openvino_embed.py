"""sentence-transformers with the OpenVINO backend, served over HTTP."""

import os
from pathlib import Path
import subprocess
from typing import Any, Dict, List, Optional

from .embed_base import EmbedAdapter, post_json, wait_ready

DEFAULT_VENV = "/var/tmp/embed-venv"
PINNED_VERSION = "6.2.0"
OPENVINO_VERSION = "2026.4.1"


class OpenVINOAdapter(EmbedAdapter):
    """Starts `servers/openvino_server.py` (sentence-transformers + OpenVINO)."""

    name = "openvino"

    def __init__(self, threads: int = 4, venv: Optional[str] = None):
        super().__init__(threads=threads)
        self.venv = Path(venv or os.environ.get("EMBED_VENV") or DEFAULT_VENV)
        self.python = self.venv / "bin" / "python"
        self.server_script = Path(__file__).resolve().parent / "servers" / "openvino_server.py"
        self.version = {
            "name": "sentence-transformers+openvino",
            "version": PINNED_VERSION,
            "openvino": OPENVINO_VERSION,
            "backend": "openvino",
            "venv": str(self.venv),
            "python": str(self.python),
        }

    def start(self, row: Dict[str, Any], model: Dict[str, Any], port: int) -> Dict[str, Any]:
        model_path = Path(row["path"]).expanduser()
        env = dict(os.environ)
        env.update(
            {
                "OMP_NUM_THREADS": str(self.threads),
                "MKL_NUM_THREADS": str(self.threads),
                "OPENBLAS_NUM_THREADS": str(self.threads),
                "TOKENIZERS_PARALLELISM": "false",
            }
        )
        cmd = [
            str(self.python), str(self.server_script),
            "--model-path", str(model_path),
            "--port", str(port),
            "--threads", str(self.threads),
            "--max-seq-length", str(row.get("max_seq_length")
                                    or model.get("max_tokens") or 512),
        ]
        self.port = port
        self.base_url = f"http://127.0.0.1:{port}"
        self.proc = subprocess.Popen(
            cmd,
            stdout=self.open_log(Path("/var/tmp") / f"embed-openvino-{port}.log"),
            stderr=subprocess.STDOUT,
            env=env,
        )
        if not wait_ready(self.base_url, timeout=600.0, paths=("/health",)):
            tail = self.read_log_tail()
            self.stop()
            raise RuntimeError(f"openvino server failed to start:\n{tail}")
        return {"cmd": cmd, "pooling": model.get("pooling"), "threads": self.threads,
                "ov_config": {"INFERENCE_NUM_THREADS": str(self.threads)}}

    def embed(self, texts: List[str]) -> Dict[str, Any]:
        reply = post_json(self.base_url + "/embed", {"texts": texts})
        return {
            "embeddings": reply["embeddings"],
            "prompt_tokens": int(reply.get("tokens", 0)),
        }
