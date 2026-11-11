"""Plain ONNX Runtime embedding adapter (`ort_server.py`)."""

import os
from pathlib import Path
import subprocess
from typing import Any, Dict, List, Optional

from .embed_base import EmbedAdapter, post_json, wait_ready

DEFAULT_VENV = "/var/tmp/embed-venv"


class OnnxRuntimeAdapter(EmbedAdapter):
    """Starts `servers/ort_server.py` (ONNX Runtime, no fastembed)."""

    name = "onnxruntime"

    def __init__(self, threads: int = 4, venv: Optional[str] = None):
        super().__init__(threads=threads)
        self.venv = Path(venv or os.environ.get("EMBED_VENV") or DEFAULT_VENV)
        self.python = self.venv / "bin" / "python"
        self.server_script = Path(__file__).resolve().parent / "servers" / "ort_server.py"
        self.version = {
            "name": "onnxruntime",
            "backend": "CPUExecutionProvider",
            "venv": str(self.venv),
            "python": str(self.python),
        }

    def get_version_info(self) -> Dict[str, Any]:
        info = dict(self.version)
        try:
            out = subprocess.run(
                [str(self.python), "-c", "import onnxruntime; print(onnxruntime.__version__)"],
                capture_output=True, text=True, timeout=30,
            )
            info["version"] = out.stdout.strip()
        except Exception:
            pass
        return info

    def start(self, row: Dict[str, Any], model: Dict[str, Any], port: int) -> Dict[str, Any]:
        model_path = Path(row["path"]).expanduser()
        env = dict(os.environ)
        env.update({"OMP_NUM_THREADS": str(self.threads), "MKL_NUM_THREADS": str(self.threads)})
        cmd = [
            str(self.python), str(self.server_script),
            "--model-path", str(model_path),
            "--pooling", model.get("pooling", "mean"),
            "--port", str(port),
            "--threads", str(self.threads),
        ]
        self.port = port
        self.base_url = f"http://127.0.0.1:{port}"
        self.proc = subprocess.Popen(
            cmd,
            stdout=self.open_log(Path("/var/tmp") / f"embed-ort-{port}.log"),
            stderr=subprocess.STDOUT,
            env=env,
        )
        if not wait_ready(self.base_url, timeout=300.0, paths=("/health",), proc=self.proc):
            tail = self.read_log_tail()
            self.stop()
            raise RuntimeError(f"onnxruntime server failed to start:\n{tail}")
        return {"cmd": cmd, "pooling": model.get("pooling"), "threads": self.threads,
                "intra_op_num_threads": self.threads}

    def embed(self, texts: List[str]) -> Dict[str, Any]:
        reply = post_json(self.base_url + "/embed", {"texts": texts})
        return {
            "embeddings": reply["embeddings"],
            "prompt_tokens": int(reply.get("tokens", 0)),
        }
