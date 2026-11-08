"""PrismML's llama.cpp fork engine benchmark adapter.

The upstream llama.cpp release pinned for the other baselines cannot load the
Bonsai 2 ternary formats (PQ2_0 is ggml type 142, which upstream rejects), so
the only other engine that runs those files is PrismML's fork. This adapter
mirrors :mod:`engines.llama_cpp`: it fetches a pinned commit, builds
``llama-server`` and ``llama-bench`` with CMake Release, and drives the same
OpenAI-compatible HTTP interface.

The commit is the one ``docs/reference/bonsai2_spec.md`` was written against.
If that commit can no longer be fetched, the adapter falls back to the head of
the ``prism`` branch and records the SHA it actually built.
"""

import json
import os
from pathlib import Path
import subprocess
from typing import Any, Dict, List, Optional

from .llama_cpp import LlamaCppAdapter


class PrismLlamaCppAdapter(LlamaCppAdapter):
    """Adapter for PrismML's llama.cpp fork (branch ``prism``)."""

    DEFAULT_PINNED_TAG = "prism"
    DEFAULT_PINNED_COMMIT = "62c418ccd7aefee37997e94c2d9fac1eed7fc89b"
    DEFAULT_PINNED_COMMIT_SHORT = DEFAULT_PINNED_COMMIT[:9]
    GIT_REPO_URL = "https://github.com/PrismML-Eng/llama.cpp.git"

    ENV_SERVER_BIN = "PRISM_LLAMA_SERVER_BIN"
    ENV_BENCH_BIN = "PRISM_LLAMA_BENCH_BIN"

    def __init__(
        self,
        pinned_tag: str = DEFAULT_PINNED_TAG,
        pinned_commit: str = DEFAULT_PINNED_COMMIT,
        custom_bin_dir: Optional[Path] = None,
        flash_attn: Optional[str] = None,
    ):
        super().__init__(
            pinned_tag=pinned_tag,
            pinned_commit=pinned_commit,
            custom_bin_dir=custom_bin_dir,
        )
        self.name = "prism-llama.cpp"
        self.pinned_version = f"{pinned_tag} ({pinned_commit[:9]})"
        self.resolved_commit: Optional[str] = None
        self.commit_fallback = False
        self.repo_dir: Optional[Path] = None
        self.build_flags = "CMake Release, native CPU (AVX-512 on this host)"
        if flash_attn not in (None, "auto", "on", "off"):
            raise ValueError(f"flash_attn must be auto, on or off, not {flash_attn!r}")
        # None leaves the fork's own default (`-fa auto`); the value is recorded in
        # every run set so a reader can tell which setting produced the numbers.
        self.flash_attn = flash_attn

    def _flash_attn_args(self, extra_args: Optional[List[str]]) -> List[str]:
        """Return extra_args with this adapter's explicit -fa setting in front."""
        args = list(extra_args or [])
        if self.flash_attn and not any(a in ("-fa", "--flash-attn") for a in args):
            args = ["-fa", self.flash_attn] + args
        return args

    def start_server(
        self,
        model_path: str,
        port: int = 8201,
        threads: int = 8,
        ctx: int = 4096,
        slots: int = 16,
        extra_args: Optional[List[str]] = None,
    ) -> subprocess.Popen:
        """Start `llama-server`, with the adapter's flash-attention setting applied."""
        return super().start_server(
            model_path=model_path,
            port=port,
            threads=threads,
            ctx=ctx,
            slots=slots,
            extra_args=self._flash_attn_args(extra_args),
        )

    def _git(self, args: List[str], cwd: Path) -> Optional[str]:
        try:
            out = subprocess.check_output(["git"] + args, cwd=str(cwd), stderr=subprocess.DEVNULL)
            return out.decode("utf-8").strip()
        except Exception:
            return None

    def _record_resolved_commit(self) -> None:
        if self.repo_dir and (self.repo_dir / ".git").exists():
            sha = self._git(["rev-parse", "HEAD"], self.repo_dir)
            if sha:
                self.resolved_commit = sha
                self.pinned_version = f"{self.pinned_tag} ({sha[:9]})"

    def build_or_locate(self, cache_dir: str, dry_run: bool = False) -> str:
        """Locate existing binaries or fetch and build the pinned commit.

        Unlike llama.cpp's adapter, the pin is a commit rather than a tag, so the
        clone fetches that commit directly (shallow). Binaries built here land in
        the caller's cache directory, never in the repository.
        """
        # 1. Environment variable override
        env_server = os.environ.get(self.ENV_SERVER_BIN)
        if env_server and Path(env_server).exists():
            self.server_bin = Path(env_server)
            env_bench = os.environ.get(self.ENV_BENCH_BIN)
            self.bench_bin = Path(env_bench) if env_bench else None
            self.build_flags = "prebuilt binary from the environment"
            return str(self.server_bin)

        # 2. Explicit binary directory
        if self.custom_bin_dir:
            cand_server = self.custom_bin_dir / "llama-server"
            cand_bench = self.custom_bin_dir / "llama-bench"
            if cand_server.exists():
                self.server_bin = cand_server
                self.bench_bin = cand_bench if cand_bench.exists() else None
                self.build_flags = "prebuilt binary from a custom directory"
                return str(self.server_bin)

        # 3. Build in the cache directory (never inside the checkout)
        target_dir = Path(cache_dir).expanduser() / f"llama.cpp-prism-{self.pinned_commit[:9]}"
        build_dir = target_dir / "build"
        bin_server = build_dir / "bin" / "llama-server"
        bin_bench = build_dir / "bin" / "llama-bench"
        self.repo_dir = target_dir

        if bin_server.exists():
            self.server_bin = bin_server
            self.bench_bin = bin_bench if bin_bench.exists() else None
            self._record_resolved_commit()
            return str(self.server_bin)

        n_cores = os.cpu_count() or 4
        cmake_cfg_cmd = [
            "cmake", "-B", str(build_dir),
            "-DGGML_AVX2=ON", "-DGGML_AVX=ON", "-DGGML_FMA=ON",
            "-DCMAKE_BUILD_TYPE=Release",
            str(target_dir),
        ]
        cmake_build_cmd = [
            "cmake", "--build", str(build_dir),
            "--config", "Release",
            "-j", str(n_cores),
            "--target", "llama-server", "llama-bench",
        ]

        if dry_run:
            print(f"[dry-run] git init {target_dir}")
            print(f"[dry-run] git -C {target_dir} remote add origin {self.GIT_REPO_URL}")
            print(f"[dry-run] git -C {target_dir} fetch --depth 1 origin {self.pinned_commit}")
            print(f"[dry-run] {' '.join(cmake_cfg_cmd)}")
            print(f"[dry-run] {' '.join(cmake_build_cmd)}")
            self.server_bin = bin_server
            self.bench_bin = bin_bench
            return str(bin_server)

        target_dir.parent.mkdir(parents=True, exist_ok=True)
        if not (target_dir / ".git").exists():
            target_dir.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "init", "-q", str(target_dir)], check=True)
            subprocess.run(
                ["git", "remote", "add", "origin", self.GIT_REPO_URL],
                cwd=str(target_dir), check=True,
            )
            # Try the pinned commit first; fall back to the branch head if the
            # commit is no longer reachable.
            fetched = subprocess.run(
                ["git", "fetch", "--depth", "1", "origin", self.pinned_commit],
                cwd=str(target_dir),
            ).returncode == 0
            if fetched:
                subprocess.run(
                    ["git", "checkout", "-q", "FETCH_HEAD"],
                    cwd=str(target_dir), check=True,
                )
            else:
                print(
                    f"[warning] pinned commit {self.pinned_commit} is not fetchable; "
                    f"falling back to the head of {self.pinned_tag}",
                    flush=True,
                )
                subprocess.run(
                    ["git", "fetch", "--depth", "1", "origin", self.pinned_tag],
                    cwd=str(target_dir), check=True,
                )
                subprocess.run(
                    ["git", "checkout", "-q", "FETCH_HEAD"],
                    cwd=str(target_dir), check=True,
                )
                self.commit_fallback = True

        self._record_resolved_commit()
        if self.commit_fallback:
            print(
                f"[warning] built {self.resolved_commit} (branch head), not the pin "
                f"{self.pinned_commit}",
                flush=True,
            )

        subprocess.run(cmake_cfg_cmd, check=True)
        subprocess.run(cmake_build_cmd, check=True)

        self.server_bin = bin_server
        self.bench_bin = bin_bench
        return str(self.server_bin)

    def get_version_info(self) -> Dict[str, Any]:
        """Return engine metadata for run sets."""
        return {
            "name": self.name,
            "version": self.pinned_version,
            "commit": self.resolved_commit or self.pinned_commit,
            "pinned_commit": self.pinned_commit,
            "commit_fallback": self.commit_fallback,
            "repo": self.GIT_REPO_URL,
            "build_type": self.build_flags,
            "flash_attn": self.flash_attn or "default",
            "server_binary": str(self.server_bin) if self.server_bin else "pending",
        }

    def run_cli_bench(
        self,
        model_path: str,
        prompt_tokens: int,
        decode_tokens: int,
        threads: int = 8,
        extra_args: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run `llama-bench -o json` for single-user measurements.

        Newer llama-bench JSON reports one entry per test with ``avg_ts``;
        older releases report ``speed_prompt`` / ``speed_gen``. Both are read.
        """
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
        cmd.extend(self._flash_attn_args(extra_args))

        res = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        data = json.loads(res.stdout.decode("utf-8"))
        prefill_tok_s = 0.0
        decode_tok_s = 0.0
        for entry in data:
            n_prompt = int(entry.get("n_prompt", 0) or 0)
            n_gen = int(entry.get("n_gen", 0) or 0)
            if n_prompt > 0 and n_gen == 0:
                prefill_tok_s = float(entry.get("avg_ts", entry.get("speed_prompt", 0.0)))
            elif n_gen > 0:
                decode_tok_s = float(entry.get("avg_ts", entry.get("speed_gen", 0.0)))

        return {
            "decode_tok_s": decode_tok_s,
            "prefill_tok_s": prefill_tok_s,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": decode_tokens,
        }
