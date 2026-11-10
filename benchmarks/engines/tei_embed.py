"""text-embeddings-inference (TEI) embedding adapter, CPU container."""

import os
from pathlib import Path
import subprocess
from typing import Any, Dict, List, Optional

from .embed_base import EmbedAdapter, post_json, wait_ready

DEFAULT_IMAGE = "ghcr.io/huggingface/text-embeddings-inference:cpu-1.9.4"
PINNED_VERSION = "1.9.4"
CONTAINER_NAME = "pg-embed-tei"


class TEIAdapter(EmbedAdapter):
    """Runs the pinned TEI CPU image with the model directory bind-mounted."""

    name = "tei"

    def __init__(
        self,
        threads: int = 4,
        image: Optional[str] = None,
        runtime: Optional[str] = None,
    ):
        super().__init__(threads=threads)
        self.image = image or os.environ.get("TEI_IMAGE") or DEFAULT_IMAGE
        self.runtime = runtime or os.environ.get("TEI_RUNTIME") or "podman"
        self.version = {
            "name": "text-embeddings-inference",
            "version": PINNED_VERSION,
            "image": self.image,
            "runtime": self.runtime,
            "backend": "candle (CPU image; no ONNX Runtime backend in published CPU images)",
        }

    def start(self, row: Dict[str, Any], model: Dict[str, Any], port: int) -> Dict[str, Any]:
        model_dir = Path(row["path"]).expanduser().resolve()
        max_tokens = int(model.get("server_ctx") or 70000)
        cmd = [
            self.runtime, "run", "--rm", "--name", CONTAINER_NAME,
            "--network", "host",
            "-e", f"MKL_NUM_THREADS={self.threads}",
            "-e", f"OMP_NUM_THREADS={self.threads}",
            "-e", f"RAYON_NUM_THREADS={self.threads}",
            "-v", f"{model_dir}:/data:ro,Z",
            self.image,
            "--model-id", "/data",
            "--hostname", "127.0.0.1",
            "--port", str(port),
            "--pooling", model.get("pooling", "mean"),
            "--dtype", "float32",
            "--tokenization-workers", "1",
            "--max-client-batch-size", "128",
            "--max-batch-tokens", str(max_tokens),
            "--max-batch-requests", "128",
        ]
        self.port = port
        self.base_url = f"http://127.0.0.1:{port}"
        self.proc = subprocess.Popen(
            cmd,
            stdout=self.open_log(Path("/var/tmp") / f"embed-tei-{port}.log"),
            stderr=subprocess.STDOUT,
        )
        if not wait_ready(self.base_url, timeout=300.0, paths=("/health",)):
            tail = self.read_log_tail(40)
            self.stop()
            raise RuntimeError(f"TEI failed to start:\n{tail}")
        return {"cmd": cmd, "pooling": model.get("pooling"), "threads": self.threads,
                "backend": "candle", "dtype": "float32"}

    def embed(self, texts: List[str]) -> Dict[str, Any]:
        reply = post_json(self.base_url + "/embed", {"inputs": texts, "truncate": False})
        return {
            "embeddings": reply,
            "prompt_tokens": 0,  # TEI does not report usage; the harness uses tokenizer counts
        }

    def stop(self, timeout: float = 30.0) -> None:
        if self.proc is not None:
            subprocess.run(
                [self.runtime, "rm", "-f", CONTAINER_NAME],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        super().stop(timeout=timeout)
