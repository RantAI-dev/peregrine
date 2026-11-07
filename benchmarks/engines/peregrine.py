"""Peregrine inference engine benchmark adapter."""

import json
import os
from pathlib import Path
import subprocess
from typing import Any, Dict, List, Optional

from .base import EngineAdapter, wait_for_server


class PeregrineAdapter(EngineAdapter):
    """Adapter for the Peregrine CPU inference engine."""

    def __init__(self, workspace_root: Optional[Path] = None):
        super().__init__(name="peregrine", pinned_version="head")
        self.workspace_root = workspace_root or Path(__file__).resolve().parent.parent.parent
        self.bin_path = self.workspace_root / "target" / "release" / "peregrine"
        self.probe_bin = self.workspace_root / "target" / "release" / "examples" / "probe"
        self.synth_bin = self.workspace_root / "target" / "release" / "examples" / "write_synthetic"

    def build_or_locate(self, cache_dir: str, dry_run: bool = False) -> str:
        """Build the release binary for peregrine and supporting bench tools."""
        cmd_peregrine = ["cargo", "build", "--release", "-p", "peregrine"]
        cmd_bench = [
            "cargo", "build", "--release", "-p", "peregrine-bench",
            "--example", "probe", "--example", "write_synthetic",
        ]

        if dry_run:
            print(f"[dry-run] {' '.join(cmd_peregrine)}")
            print(f"[dry-run] {' '.join(cmd_bench)}")
            return str(self.bin_path)

        subprocess.run(cmd_peregrine, cwd=str(self.workspace_root), check=True)
        subprocess.run(cmd_bench, cwd=str(self.workspace_root), check=True)
        return str(self.bin_path)

    def get_version_info(self) -> Dict[str, Any]:
        """Collect git commit, rustc version, and build profile."""
        commit = "unknown"
        try:
            out = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=str(self.workspace_root)
            ).decode().strip()
            commit = out
        except Exception:
            pass

        rustc_ver = "unknown"
        try:
            out = subprocess.check_output(["rustc", "-V"]).decode().strip()
            rustc_ver = out
        except Exception:
            pass

        return {
            "name": self.name,
            "version": "0.1.0",
            "commit": commit,
            "rustc": rustc_ver,
            "profile": "release",
            "build_command": "cargo build --release -p peregrine",
        }

    def probe_machine(self, threads: int = 8) -> Dict[str, Any]:
        """Probe hardware ceilings (streaming bandwidth and FMA compute) via Peregrine telemetry."""
        if not self.probe_bin.exists():
            subprocess.run(
                ["cargo", "build", "--release", "-p", "peregrine-bench", "--example", "probe"],
                cwd=str(self.workspace_root),
                check=True,
            )

        cmd = [str(self.probe_bin), str(threads)]
        out = subprocess.check_output(cmd, cwd=str(self.workspace_root)).decode("utf-8")
        data = json.loads(out)
        return data

    def write_synthetic_model(self, out_dir: Path, ctx: int = 512) -> Path:
        """Generate a tiny synthetic GGUF model for smoke testing without real model files."""
        if not self.synth_bin.exists():
            subprocess.run(
                ["cargo", "build", "--release", "-p", "peregrine-bench", "--example", "write_synthetic"],
                cwd=str(self.workspace_root),
                check=True,
            )

        cmd = [str(self.synth_bin), str(out_dir), str(ctx)]
        out = subprocess.check_output(cmd, cwd=str(self.workspace_root)).decode("utf-8").strip()
        return Path(out)

    def start_server(
        self,
        model_path: str,
        port: int = 8201,
        threads: int = 8,
        ctx: int = 4096,
        slots: int = 16,
        extra_args: Optional[List[str]] = None,
    ) -> subprocess.Popen:
        """Start `peregrine serve` in a background subprocess."""
        cmd = [
            str(self.bin_path),
            "serve",
            "-m",
            str(model_path),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "-t",
            str(threads),
            "-c",
            str(ctx),
            "--slots",
            str(slots),
            "--no-profile",
        ]
        if extra_args:
            cmd.extend(extra_args)

        proc = subprocess.Popen(
            cmd,
            cwd=str(self.workspace_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        url = f"http://127.0.0.1:{port}"
        if not wait_for_server(url, timeout=30.0):
            # If server failed, capture stderr and raise error
            err = ""
            if proc.stderr:
                try:
                    err = proc.stderr.read().decode("utf-8", errors="replace")
                except Exception:
                    pass
            self.stop_server(proc)
            raise RuntimeError(f"Peregrine server failed to start on {url}: {err}")

        return proc

    def run_cli_bench(
        self,
        model_path: str,
        prompt_tokens: int,
        decode_tokens: int,
        threads: int = 8,
        extra_args: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run `peregrine bench --json` for single-user throughput and bytes/token measurement."""
        cmd = [
            str(self.bin_path),
            "bench",
            "-m",
            str(model_path),
            "--prompt-tokens",
            str(prompt_tokens),
            "--tokens",
            str(decode_tokens),
            "--json",
            "--no-profile",
        ]
        if extra_args:
            cmd.extend(extra_args)

        env = os.environ.copy()
        if threads > 0:
            env["RAYON_NUM_THREADS"] = str(threads)

        res = subprocess.run(
            cmd,
            cwd=str(self.workspace_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            check=True,
        )
        data = json.loads(res.stdout.decode("utf-8"))
        return {
            "decode_tok_s": float(data.get("decode_tok_s", 0.0)),
            "prefill_tok_s": float(data.get("prefill_tok_s", 0.0)),
            "bytes_per_token": float(data.get("bytes_per_token", 0.0)),
            "decode_s": float(data.get("decode_s", 0.0)),
            "prefill_s": float(data.get("prefill_s", 0.0)),
            "prompt_tokens": int(data.get("prompt_tokens", prompt_tokens)),
            "completion_tokens": int(data.get("tokens", decode_tokens)),
            "options": data.get("options", {}),
        }
