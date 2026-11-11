#!/usr/bin/env python3
"""Decision benchmark: `POST /v1/systemone` questions per second.

Drives one engine at a time against a native decision model (System One):
one state, `--questions` noul questions in one request, medians of `--runs`
requests.  The states come from `decide_texts.py` (token-calibrated with the
model's own tokenizer, within 2%).  Standard library only; each engine is a
subprocess this script starts and stops by PID.

The fair llama.cpp setting is the default: `-np 11` (one slot per question of
the largest request plus a spare) and `-c 16384` (so a 1 000-token state fits
a slot).  llama.cpp groups a request's questions in `server_decision_group_tasks`
and forwards their common prefix once per group; fewer slots than questions
splits the request into groups and evaluates the state once per group, which
makes llama.cpp look far slower than it is.  `--llama-slots 1` reproduces the
historical, unfair setting for comparison.

  python benchmarks/decide.py --engine peregrine --binary target/release/peregrine \
      --model ~/models/tinyopenjev-for-testing-Q8_0.gguf \
      --states /var/tmp/embed-texts-s49k/decide-states.json \
      --output-dir benchmarks/results/bluefin-2026-11-11-s4.9-standings/decide/fair
"""

import argparse
import datetime
import json
import os
from pathlib import Path
import platform
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request

PEREGRINE = "peregrine"
LLAMA = "llama_cpp"

# Result files never carry this machine's home path.
HOME = os.path.expanduser("~")


def clean_path(text):
    return text.replace(HOME, "~") if HOME not in ("", "/") else text

# Ten noul questions, in request order.  The answers are probabilities, so the
# wording only has to give the model something to decide; the timing is what
# this harness measures.
QUESTIONS = [
    "Is the customer angry?",
    "Does the message ask for a refund?",
    "Is the message about a delivery?",
    "Does the sender mention a technical problem?",
    "Should this be escalated to a human?",
    "Is the tone formal?",
    "Does the complaint concern a charge?",
    "Does the message mention an order number?",
    "Is the customer satisfied?",
    "Does the message ask a question?",
]


def post(url, body, timeout=600.0):
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def wait_ready(url, proc, timeout=180.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(url, timeout=2.0) as r:
                if r.status < 500:
                    return True
        except Exception:
            pass
        time.sleep(0.1)
    return False


def free_port(start):
    port = start
    while port < start + 50:
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
        port += 1
    raise RuntimeError("no free port")


class Server:
    def __init__(self, engine, args):
        self.engine = engine
        self.args = args
        self.port = free_port(args.port)
        self.base = f"http://127.0.0.1:{self.port}"
        self.log_path = Path("/var/tmp") / f"decide-{engine}-{self.port}.log"
        self.log = open(self.log_path, "wb")
        env = dict(os.environ)
        if engine == PEREGRINE and args.profile:
            env["PEREGRINE_PROF2"] = "1"
        if engine == PEREGRINE:
            cmd = [
                str(Path(args.binary).expanduser()),
                "serve",
                "-m", str(Path(args.model).expanduser()),
                "--host", "127.0.0.1",
                "--port", str(self.port),
                "-t", str(args.threads),
                "-c", str(args.ctx),
                "--slots", str(args.slots),
                "--no-profile",
            ]
        else:
            cmd = [
                str(Path(args.server_bin).expanduser()),
                "-m", str(Path(args.model).expanduser()),
                "-t", str(args.threads),
                "-c", str(args.llama_ctx),
                "-np", str(args.llama_slots),
                "-b", str(args.llama_batch),
                "-ub", str(args.llama_ubatch),
                "--host", "127.0.0.1",
                "--port", str(self.port),
            ]
            if args.llama_kv_unified:
                cmd.append("-kvu")
            if args.llama_cache_ram is not None:
                cmd += ["--cache-ram", str(args.llama_cache_ram)]
            if args.llama_flash_attn:
                cmd += ["-fa", args.llama_flash_attn]
        self.cmd = cmd
        self.proc = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT, env=env)
        if not wait_ready(self.base + "/v1/models", self.proc):
            self.stop()
            tail = self.log_path.read_text(errors="replace")[-4000:]
            raise RuntimeError(f"{engine} failed to start:\n{tail}")

    def request(self, state, n_questions):
        questions = {
            f"q{i}": {"type": "noul", "instructions": QUESTIONS[i]}
            for i in range(n_questions)
        }
        return post(self.base + "/v1/systemone", {"state": state, "questions": questions})

    def profile_lines(self, offset):
        self.log.flush()
        with open(self.log_path, "rb") as f:
            f.seek(offset)
            data = f.read()
        return [ln for ln in data.decode("utf-8", "replace").splitlines()
                if ln.startswith("prof:")], offset + len(data)

    def stop(self):
        try:
            self.proc.terminate()
            self.proc.wait(timeout=20)
        except Exception:
            try:
                self.proc.kill()
                self.proc.wait(timeout=10)
            except Exception:
                pass
        try:
            self.log.close()
        except Exception:
            pass


def pct(vals, p):
    s = sorted(vals)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def main():
    ap = argparse.ArgumentParser(description="Decision (/v1/systemone) benchmark")
    ap.add_argument("--engine", choices=[PEREGRINE, LLAMA], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--states", type=Path, required=True)
    ap.add_argument("--states-lengths", default="64,300,1000")
    ap.add_argument("--questions", default="1,3,10")
    ap.add_argument("--runs", type=int, default=21)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--ctx", type=int, default=8192,
                    help="Peregrine context (per slot)")
    ap.add_argument("--llama-ctx", type=int, default=16384,
                    help="llama.cpp -c: the KV pool; with -np 11 it must hold "
                         "a 1000-token state in every slot")
    ap.add_argument("--slots", type=int, default=4,
                    help="Peregrine server slots")
    ap.add_argument("--llama-batch", type=int, default=2048)
    ap.add_argument("--llama-ubatch", type=int, default=512)
    ap.add_argument("--llama-cache-ram", type=int, default=None,
                    help="llama.cpp --cache-ram MiB (0 keeps the decision-model default)")
    ap.add_argument("--llama-flash-attn", default=None,
                    help="llama.cpp -fa value (auto, on, off)")
    ap.add_argument("--llama-kv-unified", action="store_true",
                    help="llama.cpp -kvu (one KV pool shared across slots)")
    ap.add_argument("--llama-slots", type=int, default=11,
                    help="llama.cpp -np: at least the largest question count, "
                         "so one request's questions share their prefix")
    ap.add_argument("--port", type=int, default=8351)
    ap.add_argument("--binary", default="target/release/peregrine")
    ap.add_argument("--server-bin", default="/var/tmp/llama.cpp/build/bin/llama-server")
    ap.add_argument("--profile", action="store_true",
                    help="start Peregrine with PEREGRINE_PROF2=1 and keep its prof lines")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--round", type=int, default=1)
    args = ap.parse_args()

    states = json.loads(args.states.read_text(encoding="utf-8"))["states"]
    lengths = [int(x) for x in args.states_lengths.split(",") if x]
    qcounts = [int(x) for x in args.questions.split(",") if x]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    server = Server(args.engine, args)
    cells = {}
    log_offset = 0
    try:
        for L in lengths:
            state = states[str(L)]["text"]
            for q in qcounts:
                # Warm-up, not timed: the template compile, the cache allocation,
                # and the first pass over the weights.
                server.request(state, q)
                walls = []
                tokens = 0
                answers = 0
                for _ in range(args.runs):
                    t0 = time.perf_counter()
                    reply = server.request(state, q)
                    walls.append(time.perf_counter() - t0)
                    tokens = int(reply.get("usage", {}).get("input_tokens", 0))
                    answers = len(reply.get("answers", {}))
                if answers != q:
                    raise RuntimeError(f"{args.engine}: {q} questions asked, {answers} answered")
                med = statistics.median(walls)
                key = f"S{L}_q{q}"
                cell = {
                    "state_tokens": states[str(L)]["tokens"],
                    "questions": q,
                    "wall_ms_median": round(med * 1e3, 3),
                    "wall_ms_min": round(min(walls) * 1e3, 3),
                    "wall_ms_p50": round(pct(walls, 0.50) * 1e3, 3),
                    "wall_ms_p90": round(pct(walls, 0.90) * 1e3, 3),
                    "questions_per_s": round(q / med, 3),
                    "per_question_ms": round(med * 1e3 / q, 3),
                    "runs": [round(w * 1e3, 3) for w in walls],
                    "input_tokens": tokens,
                }
                if args.profile and args.engine == PEREGRINE:
                    lines, log_offset = server.profile_lines(log_offset)
                    cell["profile"] = lines
                cells[key] = cell
                print(f"  {key:12s} {cell['questions_per_s']:8.2f} q/s  "
                      f"{cell['wall_ms_median']:8.1f} ms  {cell['per_question_ms']:7.2f} ms/q",
                      flush=True)
    finally:
        server.stop()

    record = {
        "benchmark": "peregrine-s4.9-decisions",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "round": args.round,
        "engine": args.engine,
        "model": clean_path(str(Path(args.model).expanduser())),
        "machine": {
            "hostname": platform.node(),
            "cpu": platform.processor(),
            "threads": args.threads,
            "kernel": platform.release(),
        },
        "settings": {
            "ctx": args.ctx,
            "llama_ctx": args.llama_ctx,
            "slots": args.slots,
            "llama_slots": args.llama_slots,
            "llama_batch": args.llama_batch,
            "llama_ubatch": args.llama_ubatch,
            "llama_kv_unified": args.llama_kv_unified,
            "llama_cache_ram": args.llama_cache_ram,
            "llama_flash_attn": args.llama_flash_attn,
            "runs": args.runs,
            "questions": qcounts,
            "states": lengths,
            "cmd": [clean_path(part) for part in server.cmd],
        },
        "cells": cells,
    }
    name = f"decide-{args.engine}-r{args.round}.json"
    (args.output_dir / name).write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(f"wrote {args.output_dir / name}")


if __name__ == "__main__":
    main()
