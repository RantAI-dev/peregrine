"""ik_llama.cpp engine benchmark adapter.

ik_llama.cpp is the CPU-focused fork of llama.cpp named as a competitor in
``docs/targets.md`` Section 2 (Claim 2). It shares llama.cpp's OpenAI-compatible
server and most of its flags, but it has its own quantised GEMM kernels (the
``IQK`` path, gated by ``HAVE_FANCY_SIMD``), its own CPU flash attention, and a
run-time row-repacking option (``-rtr``) that turns the stock k-quant tensors of
a Q4_K_M file into the ``_R4``/``_R8`` interleaved layouts at load time.

This adapter therefore does three things differently from
:mod:`engines.llama_cpp`:

1. It pins a commit rather than a release tag (the fork has no comparable
   release), and builds it with the AVX-512 flags its ``docs/build.md`` requires
   for the fast path. A plain ``Release`` build silently falls back to AVX2.
2. It starts the server with ``-rtr`` (run-time repack) and the fork's own flash
   attention default (on), which are its documented CPU performance settings.
3. It reads llama-bench's newer JSON fields (``avg_ts``), which is what this
   fork emits, instead of upstream's ``speed_prompt`` / ``speed_gen``.

The same GGUF files as the other engines are used; ``-rtr`` repacks them in
memory, it does not re-quantise or rewrite them.
"""

import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Dict, List, Optional

from .llama_cpp import LlamaCppAdapter


class IkLlamaCppAdapter(LlamaCppAdapter):
    """Adapter for the pinned ik_llama.cpp fork."""

    DEFAULT_PINNED_TAG = "master"
    # Head of master on 2026-11-11, 08:54:37 +0200 (the build used for the
    # 2026-11-11 desktop run). Update together with a re-measurement.
    DEFAULT_PINNED_COMMIT = "479074ccac2ac94f06b6d24a9ce26f0f9e9aa6ae"
    DEFAULT_PINNED_COMMIT_SHORT = DEFAULT_PINNED_COMMIT[:9]
    GIT_REPO_URL = "https://github.com/ikawrakow/ik_llama.cpp.git"

    ENV_SERVER_BIN = "IK_LLAMA_SERVER_BIN"
    ENV_BENCH_BIN = "IK_LLAMA_BENCH_BIN"

    # Documented CPU performance settings for this fork.
    # -rtr: repack k-quants to their interleaved _R4/_R8 kernels at load time
    #       (docs/parameters.md, "--run-time-repack").
    # -fa on: the fork's own default (docs/parameters.md), its CPU flash
    #       attention kernels; the harness records whichever value produced a row.
    DEFAULT_REPACK = True
    DEFAULT_FLASH_ATTN = "on"

    # docs/build.md, "CPU build flags for AVX-512": without these the IQK
    # quantised kernels are not compiled in and the build silently uses AVX2.
    AVX512_CMAKE_FLAGS = [
        "-DGGML_NATIVE=ON",
        "-DGGML_AVX512=ON",
        "-DGGML_AVX512_VBMI=ON",
        "-DGGML_AVX512_VNNI=ON",
        "-DGGML_AVX512_BF16=ON",
    ]

    def __init__(
        self,
        pinned_tag: str = DEFAULT_PINNED_TAG,
        pinned_commit: str = DEFAULT_PINNED_COMMIT,
        custom_bin_dir: Optional[Path] = None,
        flash_attn: str = DEFAULT_FLASH_ATTN,
        repack: bool = DEFAULT_REPACK,
    ):
        super().__init__(
            pinned_tag=pinned_tag,
            pinned_commit=pinned_commit,
            custom_bin_dir=custom_bin_dir,
            flash_attn=flash_attn,
        )
        self.name = "ik_llama.cpp"
        self.pinned_version = f"{pinned_tag} ({pinned_commit[:9]})"
        self.repack = repack
        self.resolved_commit: Optional[str] = None
        self.commit_fallback = False
        self.repo_dir: Optional[Path] = None
        self.build_flags = (
            "CMake Release, GGML_NATIVE + AVX-512 VNNI/VBMI/BF16 "
            "(HAVE_FANCY_SIMD IQK kernels)"
        )

    def _git(self, args: List[str], cwd: Path) -> Optional[str]:
        try:
            out = subprocess.check_output(
                ["git"] + args, cwd=str(cwd), stderr=subprocess.DEVNULL
            )
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
        """Locate existing binaries or fetch and build the pinned commit."""
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
        target_dir = (
            Path(cache_dir).expanduser()
            / f"ik_llama.cpp-{self.pinned_commit[:9]}"
        )
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
            "-DCMAKE_BUILD_TYPE=Release",
        ] + self.AVX512_CMAKE_FLAGS + [str(target_dir)]
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
            "flash_attn": self.flash_attn,
            "run_time_repack": self.repack,
            "server_binary": str(self.server_bin) if self.server_bin else "pending",
        }

    def start_server(
        self,
        model_path: str,
        port: int = 8201,
        threads: int = 8,
        ctx: int = 4096,
        slots: int = 16,
        extra_args: Optional[List[str]] = None,
    ) -> subprocess.Popen:
        """Start `llama-server` with the fork's CPU performance settings."""
        if not self.server_bin or not Path(self.server_bin).exists():
            raise FileNotFoundError(
                f"llama-server binary not found at {self.server_bin}. "
                "Run build_or_locate first."
            )

        extra = list(extra_args or [])
        # This fork has no --kv-unified: -c is the total context pool, divided
        # evenly among -np slots, exactly as in upstream llama.cpp.
        cmd = [
            str(self.server_bin),
            "-m", str(model_path),
            "--host", "127.0.0.1",
            "--port", str(port),
            "-t", str(threads),
            "-c", str(ctx * slots),
            "-np", str(slots),
        ]
        if self.repack and "-rtr" not in extra and "--run-time-repack" not in extra:
            cmd.append("-rtr")
        if "-fa" not in extra and "--flash-attn" not in extra:
            cmd.extend(["-fa", self.flash_attn])
        cmd.extend(extra)

        url = f"http://127.0.0.1:{port}"
        from .base import wait_for_server

        # Unlike upstream llama.cpp, this fork's server does not set
        # SO_REUSEADDR, so binding the same port immediately after the previous
        # phase's server exits can fail with "couldn't bind to server socket"
        # while the old socket's connections are still in TIME_WAIT. Retry a few
        # times; this is a server-start detail, not a measurement change.
        last_err = ""
        for attempt in range(5):
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
            if wait_for_server(url, timeout=60.0):
                return proc
            last_err = ""
            if proc.stderr:
                try:
                    last_err = proc.stderr.read().decode("utf-8", errors="replace")
                except Exception:
                    pass
            self.stop_server(proc)
            if "bind" not in last_err.lower() or attempt == 4:
                break
            time.sleep(2.0)
        raise RuntimeError(
            f"ik_llama.cpp llama-server failed to start on {url}: {last_err}"
        )

    def stop_server(self, proc: subprocess.Popen, timeout: float = 10.0) -> None:
        """Stop the whole process group.

        The fork's server keeps worker threads busy in a spin-wait and does not
        always leave on SIGTERM within the harness timeout, so kill the group it
        was started in (start_new_session=True) after a short grace period.
        """
        if proc.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(proc.pid), 15)
        except Exception:
            proc.terminate()
        try:
            proc.wait(timeout=timeout)
            return
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(os.getpgid(proc.pid), 9)
        except Exception:
            proc.kill()
        proc.wait()

    def run_cli_bench(
        self,
        model_path: str,
        prompt_tokens: int,
        decode_tokens: int,
        threads: int = 8,
        extra_args: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run `llama-bench -o json`; this fork reports ``avg_ts``."""
        if not self.bench_bin or not self.bench_bin.exists():
            raise FileNotFoundError(
                f"llama-bench binary not found at {self.bench_bin}. "
                "Run build_or_locate first."
            )

        extra = list(extra_args or [])
        cmd = [
            str(self.bench_bin),
            "-m", str(model_path),
            "-p", str(prompt_tokens),
            "-n", str(decode_tokens),
            "-t", str(threads),
            "-o", "json",
        ]
        if self.repack and "-rtr" not in extra and "--run-time-repack" not in extra:
            cmd.append("-rtr")
        if "-fa" not in extra and "--flash-attn" not in extra:
            cmd.extend(["-fa", self.flash_attn])
        cmd.extend(extra)

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
            speed = float(
                entry.get("avg_ts")
                or entry.get("speed_prompt")
                or entry.get("speed_gen")
                or 0.0
            )
            if n_prompt > 0 and n_gen == 0:
                prefill_tok_s = speed
            elif n_gen > 0:
                decode_tok_s = speed

        return {
            "decode_tok_s": decode_tok_s,
            "prefill_tok_s": prefill_tok_s,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": decode_tokens,
        }
