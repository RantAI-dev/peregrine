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
        parallel: int = 32,
    ):
        super().__init__(threads=threads)
        self.server_bin = Path(
            server_bin or os.environ.get("LLAMA_SERVER_BIN") or DEFAULT_SERVER
        )
        self.flash_attn = flash_attn
        self.parallel = parallel
        self.version = {
            "name": "llama.cpp",
            "version": PINNED_TAG,
            "commit": PINNED_COMMIT,
            "repo": "https://github.com/ggml-org/llama.cpp",
            "binary": str(self.server_bin),
            "flash_attn": flash_attn,
            "parallel_slots": parallel,
        }

    def start(self, row: Dict[str, Any], model: Dict[str, Any], port: int) -> Dict[str, Any]:
        if not self.server_bin.exists():
            raise FileNotFoundError(f"llama-server not found at {self.server_bin}")
        self.parallel = int(row.get("parallel") or self.parallel)
        model_path = Path(row["path"]).expanduser()
        # llama-server caps n_ctx_slot at the model's training context, and an
        # embedding request is processed as one task per slot (inputs are fed
        # serially, see the results README), so one slot only ever holds one
        # text: ctx = max_tokens per slot x parallel slots.
        ctx_slot = int(model.get("max_tokens") or 512) + 8
        ctx = int(row.get("server_ctx") or ctx_slot * self.parallel)
        # n_batch/n_ubatch must stay inside the total context, otherwise the
        # pooling graph aborts in graph_reserve (ggml_mul_mat assertion).
        batch = int(row.get("batch") or min(4096, max(1024, ctx_slot)))
        ubatch = int(row.get("ubatch") or min(batch, 1024))
        cmd = [
            str(self.server_bin),
            "-m", str(model_path),
            "--embedding",
            "--pooling", model.get("pooling", "mean"),
            "-t", str(self.threads),
            "-c", str(ctx),
            "-b", str(batch),
            "-ub", str(ubatch),
            "-np", str(self.parallel),
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
        ok = wait_ready(self.base_url, timeout=180.0, proc=self.proc)
        if not ok and self.parallel > 1:
            # v0.9.0 aborts in build_pooling ("GGML_ASSERT(ggml_can_mul_mat)")
            # for some embedding GGUFs as soon as n_slots > 1.  Retry with one
            # slot so the row is still measured; the concurrency cells then
            # queue behind that slot and the JSON says so.
            tail = self.read_log_tail(6)
            self.stop()
            self.version["slots_fallback"] = (
                "llama-server aborted with -np {} (build_pooling assertion); "
                "restarted with -np 1".format(self.parallel)
            )
            self.version["slots_fallback_log"] = tail
            self.parallel = 1
            cmd = [a for a in cmd]
            i = cmd.index("-np")
            cmd[i + 1] = "1"
            i = cmd.index("-c")
            cmd[i + 1] = str(ctx_slot)
            self.proc = subprocess.Popen(
                cmd,
                stdout=self.open_log(Path("/var/tmp") / f"embed-llama-{port}.log"),
                stderr=subprocess.STDOUT,
            )
            ok = wait_ready(self.base_url, timeout=180.0, proc=self.proc)
        if not ok:
            tail = self.read_log_tail()
            self.stop()
            raise RuntimeError(f"llama-server failed to start:\n{tail}")
        return {"cmd": cmd, "pooling": model.get("pooling"), "threads": self.threads,
                "parallel": self.parallel, "ctx": ctx, "batch": batch, "ubatch": ubatch}

    def get_version_info(self) -> Dict[str, Any]:
        info = dict(self.version)
        try:
            out = subprocess.run(
                [str(self.server_bin), "--version"], capture_output=True, text=True, timeout=20
            )
            lines = (out.stdout + out.stderr).strip().splitlines()
            info["server_version"] = lines[0].strip() if lines else ""
        except Exception:
            pass
        return info

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
