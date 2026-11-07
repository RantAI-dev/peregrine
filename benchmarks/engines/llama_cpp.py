"""llama.cpp engine benchmark adapter."""

import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Dict, List, Optional

from .base import EngineAdapter, wait_for_server


class LlamaCppAdapter(EngineAdapter):
    """Adapter for pinned llama.cpp builds."""

    # Pinned to the reference release tag and commit from docs/targets.md (Section 1 & 4)
    DEFAULT_PINNED_TAG = "v0.9.0"
    DEFAULT_PINNED_COMMIT = "cd959ba"
    GIT_REPO_URL = "https://github.com/ggml-org/llama.cpp.git"

    def __init__(
        self,
        pinned_tag: str = DEFAULT_PINNED_TAG,
        pinned_commit: str = DEFAULT_PINNED_COMMIT,
        custom_bin_dir: Optional[Path] = None,
    ):
        super().__init__(name="llama.cpp", pinned_version=f"{pinned_tag} ({pinned_commit})")
        self.pinned_tag = pinned_tag
        self.pinned_commit = pinned_commit
        self.custom_bin_dir = custom_bin_dir
        self.server_bin: Optional[Path] = None
        self.bench_bin: Optional[Path] = None

    def build_or_locate(self, cache_dir: str, dry_run: bool = False) -> str:
        """Locate existing binaries or clone and build at the pinned version via CMake."""
        # 1. Check environment variable override
        env_server = os.environ.get("LLAMA_SERVER_BIN")
        env_bench = os.environ.get("LLAMA_BENCH_BIN")
        if env_server and Path(env_server).exists():
            self.server_bin = Path(env_server)
            self.bench_bin = Path(env_bench) if env_bench else None
            return str(self.server_bin)

        # 2. Check custom_bin_dir
        if self.custom_bin_dir:
            cand_server = self.custom_bin_dir / "llama-server"
            cand_bench = self.custom_bin_dir / "llama-bench"
            if cand_server.exists():
                self.server_bin = cand_server
                self.bench_bin = cand_bench if cand_bench.exists() else None
                return str(self.server_bin)

        # 3. Check system PATH
        sys_server = shutil.which("llama-server")
        if sys_server:
            self.server_bin = Path(sys_server)
            sys_bench = shutil.which("llama-bench")
            self.bench_bin = Path(sys_bench) if sys_bench else None
            return str(self.server_bin)

        # 4. Build in cache directory
        target_dir = Path(cache_dir).expanduser() / f"llama.cpp-{self.pinned_tag}"
        build_dir = target_dir / "build"
        bin_server = build_dir / "bin" / "llama-server"
        bin_bench = build_dir / "bin" / "llama-bench"

        if bin_server.exists():
            self.server_bin = bin_server
            self.bench_bin = bin_bench if bin_bench.exists() else None
            return str(self.server_bin)

        clone_cmd = [
            "git", "clone", "--depth", "1", "--branch", self.pinned_tag,
            self.GIT_REPO_URL, str(target_dir),
        ]
        cmake_cfg_cmd = [
            "cmake", "-B", str(build_dir),
            "-DGGML_AVX2=ON", "-DGGML_AVX=ON", "-DGGML_FMA=ON",
            "-DCMAKE_BUILD_TYPE=Release",
            str(target_dir),
        ]
        n_cores = os.cpu_count() or 4
        cmake_build_cmd = [
            "cmake", "--build", str(build_dir),
            "--config", "Release",
            "-j", str(n_cores),
            "--target", "llama-server", "llama-bench",
        ]

        if dry_run:
            print(f"[dry-run] {' '.join(clone_cmd)}")
            print(f"[dry-run] {' '.join(cmake_cfg_cmd)}")
            print(f"[dry-run] {' '.join(cmake_build_cmd)}")
            self.server_bin = bin_server
            self.bench_bin = bin_bench
            return str(bin_server)

        target_dir.parent.mkdir(parents=True, exist_ok=True)
        if not target_dir.exists():
            subprocess.run(clone_cmd, check=True)

        subprocess.run(cmake_cfg_cmd, check=True)
        subprocess.run(cmake_build_cmd, check=True)

        self.server_bin = bin_server
        self.bench_bin = bin_bench
        return str(self.server_bin)

    def get_version_info(self) -> Dict[str, Any]:
        """Return engine metadata for run sets."""
        return {
            "name": self.name,
            "version": self.pinned_tag,
            "commit": self.pinned_commit,
            "repo": self.GIT_REPO_URL,
            "build_type": "CMake Release AVX2/FMA CPU",
            "server_binary": str(self.server_bin) if self.server_bin else "pending",
        }

    def supports_kv_unified(self) -> bool:
        """Check if the llama-server binary supports the --kv-unified flag."""
        if not self.server_bin or not Path(self.server_bin).exists():
            return False
        try:
            res = subprocess.run(
                [str(self.server_bin), "--help"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=5.0,
            )
            help_text = res.stdout + res.stderr
            return "--kv-unified" in help_text
        except Exception:
            return False

    def start_server(
        self,
        model_path: str,
        port: int = 8201,
        threads: int = 8,
        ctx: int = 4096,
        slots: int = 16,
        extra_args: Optional[List[str]] = None,
    ) -> subprocess.Popen:
        """Start `llama-server` in a background subprocess."""
        if not self.server_bin or not Path(self.server_bin).exists():
            raise FileNotFoundError(
                f"llama-server binary not found at {self.server_bin}. Run build_or_locate first."
            )

        # In llama.cpp, -c specifies total context divided among -np slots (or unified KV pool across all slots).
        # In both cases, the total context allocated must be ctx * slots so each parallel slot can hold up to ctx tokens.
        kv_args = ["-c", str(ctx * slots), "-np", str(slots)]
        if self.supports_kv_unified():
            kv_args.append("--kv-unified")

        cmd = [
            str(self.server_bin),
            "-m", str(model_path),
            "--host", "127.0.0.1",
            "--port", str(port),
            "-t", str(threads),
        ] + kv_args
        if extra_args:
            cmd.extend(extra_args)

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        url = f"http://127.0.0.1:{port}"
        if not wait_for_server(url, timeout=30.0):
            err = ""
            if proc.stderr:
                try:
                    err = proc.stderr.read().decode("utf-8", errors="replace")
                except Exception:
                    pass
            self.stop_server(proc)
            raise RuntimeError(f"llama-server failed to start on {url}: {err}")

        return proc

    def run_cli_bench(
        self,
        model_path: str,
        prompt_tokens: int,
        decode_tokens: int,
        threads: int = 8,
        extra_args: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run `llama-bench -o json` for single-user measurements."""
        if not self.bench_bin or not self.bench_bin.exists():
            raise FileNotFoundError(
                f"llama-bench binary not found at {self.bench_bin}. Run build_or_locate first."
            )

        cmd = [
            str(self.bench_bin),
            "-m", str(model_path),
            "-p", str(prompt_tokens),
            "-n", str(decode_tokens),
            "-t", str(threads),
            "-o", "json",
        ]
        if extra_args:
            cmd.extend(extra_args)

        res = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        data = json.loads(res.stdout.decode("utf-8"))
        # Parse llama-bench JSON output array
        prefill_tok_s = 0.0
        decode_tok_s = 0.0
        for entry in data:
            if entry.get("n_prompt", 0) > 0 and entry.get("n_gen", 0) == 0:
                prefill_tok_s = float(entry.get("speed_prompt", 0.0))
            elif entry.get("n_gen", 0) > 0:
                decode_tok_s = float(entry.get("speed_gen", 0.0))

        return {
            "decode_tok_s": decode_tok_s,
            "prefill_tok_s": prefill_tok_s,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": decode_tokens,
        }
