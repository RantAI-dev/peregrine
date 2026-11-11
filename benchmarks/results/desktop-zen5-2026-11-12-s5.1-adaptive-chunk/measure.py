#!/usr/bin/env python3
"""Single-user prefill sweep for the adaptive-chunk change (S5.1 PR 1).

Points are interleaved (one round = every point once). Each point is one
`peregrine bench --json` invocation; the row records the prompt speed, the
prefill seconds (the time to first token, plus one row) and the decode speed.

    python3 measure.py <out.json> --model <gguf> --prompts 512,4096 \
        --chunks 16,64,128 --rounds 3 --threads 12 [--isa avx2]
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time

ROOT = "/home/uaa/peregrine"
PEREGRINE = os.path.join(ROOT, "target/release/peregrine")


def run(cmd, env):
    e = dict(os.environ)
    e.update(env)
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, env=e, timeout=560)
    if p.returncode != 0:
        print(p.stdout[-2000:], file=sys.stderr)
        print(p.stderr[-2000:], file=sys.stderr)
        raise SystemExit(1)
    return json.loads(p.stdout), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", default="512,4096")
    ap.add_argument("--chunks", default="16,64,128")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--tokens", type=int, default=16)
    ap.add_argument("--isa", default=None)
    args = ap.parse_args()

    prompts = [int(x) for x in args.prompts.split(",")]
    chunks = [int(x) for x in args.chunks.split(",")]
    rows = []
    for rnd in range(args.rounds):
        for prompt in prompts:
            for chunk in chunks:
                env = {}
                if chunk > 0:
                    env["PEREGRINE_PREFILL_CHUNK"] = str(chunk)
                if args.isa:
                    env["PEREGRINE_ISA"] = args.isa
                cmd = [PEREGRINE, "bench", "-m", args.model,
                       "--prompt-tokens", str(prompt),
                       "--tokens", str(args.tokens),
                       "--threads", str(args.threads),
                       "--no-profile", "--json"]
                j, dt = run(cmd, env)
                rows.append({
                    "round": rnd,
                    "model": os.path.basename(args.model),
                    "isa": args.isa or "native",
                    "threads": args.threads,
                    "prompt_tokens": prompt,
                    "chunk": chunk,
                    "tokens": args.tokens,
                    "prefill_tok_s": j["prefill_tok_s"],
                    "prefill_s": j["prefill_s"],
                    "decode_tok_s": j.get("decode_tok_s"),
                    "wall_s": round(dt, 2),
                    "load": open("/proc/loadavg").read().split()[:3],
                })
                print(f"r{rnd} p{prompt} c{chunk} {j['prefill_tok_s']:.1f} tok/s "
                      f"({j['prefill_s']:.2f} s)", file=sys.stderr, flush=True)
    med = {}
    for r in rows:
        key = f"chunk{r['chunk']}-p{r['prompt_tokens']}"
        med.setdefault(key, []).append((r["prefill_tok_s"], r["prefill_s"]))
    summary = {
        k: {
            "prefill_tok_s": round(statistics.median(v[0] for v in vs), 1),
            "prefill_s": round(statistics.median(v[1] for v in vs), 3),
            "n": len(vs),
            "spread_tok_s": round((max(x[0] for x in vs) - min(x[0] for x in vs)) / 2, 1),
        }
        for k, vs in sorted(med.items())
    }
    out = {
        "host": "AMD Ryzen AI 9 HX 370, WSL2",
        "isa": args.isa or "native",
        "threads": args.threads,
        "rounds": args.rounds,
        "rows": rows,
        "summary": summary,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
