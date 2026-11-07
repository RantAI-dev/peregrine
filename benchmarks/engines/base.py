"""Base classes and utilities for benchmark engine adapters."""

from abc import ABC, abstractmethod
import json
import os
import subprocess
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional


class EngineAdapter(ABC):
    """Abstract base class for benchmark engine adapters."""

    def __init__(self, name: str, pinned_version: str):
        self.name = name
        self.pinned_version = pinned_version

    @abstractmethod
    def build_or_locate(self, cache_dir: str, dry_run: bool = False) -> str:
        """Ensure the engine binary exists, building or locating it.

        Returns path or description of the binary.
        """
        pass

    @abstractmethod
    def get_version_info(self) -> Dict[str, Any]:
        """Return dict describing engine name, version tag/commit, and build details."""
        pass

    @abstractmethod
    def start_server(
        self,
        model_path: str,
        port: int,
        threads: int,
        ctx: int,
        slots: int,
        extra_args: Optional[List[str]] = None,
    ) -> subprocess.Popen:
        """Start the OpenAI-compatible HTTP server as a subprocess."""
        pass

    def stop_server(self, proc: subprocess.Popen, timeout: float = 10.0) -> None:
        """Cleanly terminate a running server process."""
        if proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

    @abstractmethod
    def run_cli_bench(
        self,
        model_path: str,
        prompt_tokens: int,
        decode_tokens: int,
        threads: int,
        extra_args: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run single-user CLI benchmark if supported by the engine."""
        pass


def wait_for_server(url: str, timeout: float = 60.0) -> bool:
    """Poll server health endpoint until ready or timeout expires."""
    start = time.time()
    health_urls = [f"{url.rstrip('/')}/health", f"{url.rstrip('/')}/v1/models"]
    while time.time() - start < timeout:
        for u in health_urls:
            try:
                req = urllib.request.Request(u, method="GET")
                with urllib.request.urlopen(req, timeout=1.0) as resp:
                    if resp.status == 200:
                        return True
            except Exception:
                pass
        time.sleep(0.1)
    return False


def send_chat_completion(
    url: str,
    prompt: str,
    max_tokens: int,
    temperature: float = 0.0,
    stream: bool = True,
    model_alias: str = "peregrine",
    timeout: float = 120.0,
) -> Dict[str, Any]:
    """Send an OpenAI-compatible chat completion request and measure timings.

    Returns dict with keys:
      ttft_s: time-to-first-token in seconds
      total_s: total wall time in seconds
      decode_s: time spent generating tokens after first token
      prompt_tokens: prompt tokens reported by usage or timings
      completion_tokens: tokens generated
      prefill_tok_s: prefill throughput (tok/s)
      decode_tok_s: decode throughput (tok/s)
      timings: raw timings object from server if available
    """
    endpoint = f"{url.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": model_alias,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t0 = time.time()
    t_first: Optional[float] = None
    tokens_emitted = 0
    final_timings: Dict[str, Any] = {}
    usage: Dict[str, Any] = {}

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if stream:
                # Parse Server-Sent Events (SSE) stream
                for raw_line in resp:
                    line = raw_line.decode("utf-8").strip()
                    if not line.startswith("data:"):
                        continue
                    chunk_str = line[len("data:"):].strip()
                    if chunk_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(chunk_str)
                        choices = chunk.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            if "content" in delta and delta["content"]:
                                if t_first is None:
                                    t_first = time.time()
                                tokens_emitted += 1
                        if "timings" in chunk and chunk["timings"]:
                            final_timings = chunk["timings"]
                        if "usage" in chunk and chunk["usage"]:
                            usage = chunk["usage"]
                    except Exception:
                        continue
            else:
                body = resp.read().decode("utf-8")
                t_first = time.time()
                try:
                    res_json = json.loads(body)
                    usage = res_json.get("usage", {})
                    final_timings = res_json.get("timings", {})
                    tokens_emitted = usage.get("completion_tokens", max_tokens)
                except Exception:
                    pass
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Server returned HTTP {e.code}: {err_body}") from e

    t_end = time.time()
    total_s = t_end - t0
    ttft_s = (t_first - t0) if t_first is not None else total_s
    decode_s = max(0.0, t_end - (t_first if t_first is not None else t0))

    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", tokens_emitted)

    # Server-reported timings take precedence if present
    prefill_tok_s = 0.0
    decode_tok_s = 0.0
    if final_timings:
        p_ms = final_timings.get("prompt_ms", 0.0)
        p_n = final_timings.get("prompt_n", prompt_tokens)
        if p_ms > 0:
            prefill_tok_s = (p_n / (p_ms / 1000.0))
        elif final_timings.get("prompt_per_second"):
            prefill_tok_s = float(final_timings["prompt_per_second"])

        d_ms = final_timings.get("predicted_ms", 0.0)
        d_n = final_timings.get("predicted_n", completion_tokens)
        if d_ms > 0:
            decode_tok_s = (d_n / (d_ms / 1000.0))
        elif final_timings.get("predicted_per_second"):
            decode_tok_s = float(final_timings["predicted_per_second"])

    # Fallback to client-observed metrics
    if prefill_tok_s <= 0.0 and ttft_s > 0.0 and prompt_tokens > 0:
        prefill_tok_s = prompt_tokens / ttft_s
    if decode_tok_s <= 0.0 and decode_s > 0.0 and completion_tokens > 0:
        decode_tok_s = completion_tokens / decode_s

    return {
        "ttft_s": ttft_s,
        "total_s": total_s,
        "decode_s": decode_s,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "prefill_tok_s": prefill_tok_s,
        "decode_tok_s": decode_tok_s,
        "timings": final_timings,
    }


def run_concurrent_requests(
    url: str,
    prompts: List[str],
    max_tokens: int,
    temperature: float = 0.0,
    model_alias: str = "peregrine",
    timeout: float = 120.0,
) -> Dict[str, Any]:
    """Execute concurrent user completions synchronized by a thread barrier.

    Calculates aggregate throughput (total tokens / overall elapsed wall time)
    and collects per-user metrics (TTFT, decode tok/s).
    """
    n_users = len(prompts)
    results: List[Optional[Dict[str, Any]]] = [None] * n_users
    errors: List[Optional[Exception]] = [None] * n_users
    barrier = threading.Barrier(n_users)

    t_start_all: List[float] = [0.0]
    t_end_all: List[float] = [0.0]

    def worker(idx: int, prompt_text: str):
        try:
            barrier.wait()
            if idx == 0:
                t_start_all[0] = time.time()
            res = send_chat_completion(
                url=url,
                prompt=prompt_text,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
                model_alias=model_alias,
                timeout=timeout,
            )
            results[idx] = res
        except Exception as e:
            errors[idx] = e

    threads = [
        threading.Thread(target=worker, args=(i, p))
        for i, p in enumerate(prompts)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    t_end_all[0] = time.time()

    valid_results = [r for r in results if r is not None]
    if not valid_results:
        raise RuntimeError(f"All {n_users} concurrent requests failed: {errors}")

    wall_s = max(1e-5, t_end_all[0] - t_start_all[0])
    total_tokens = sum(r["completion_tokens"] for r in valid_results)
    aggregate_tok_s = total_tokens / wall_s

    ttfts = [r["ttft_s"] for r in valid_results]
    ttfts.sort()
    median_ttft = ttfts[len(ttfts) // 2] if ttfts else 0.0

    return {
        "users": n_users,
        "successful_users": len(valid_results),
        "wall_s": wall_s,
        "total_tokens": total_tokens,
        "aggregate_tok_s": aggregate_tok_s,
        "median_ttft_s": median_ttft,
        "min_ttft_s": min(ttfts) if ttfts else 0.0,
        "max_ttft_s": max(ttfts) if ttfts else 0.0,
        "user_results": valid_results,
    }
