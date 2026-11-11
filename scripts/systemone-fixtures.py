#!/usr/bin/env python3
"""Generate System One prompt fixtures from llama.cpp itself.

Each case in the cases file is sent as one `/v1/systemone` request with a single question to a
`llama-server` built from llama.cpp v0.9.0 (commit cd959ba) with `scripts/systemone-dump.patch`
applied. The patch does not touch rendering: at the point where the server tokenises the prompt
it has just rendered, it appends the prompt and its token ids to the file named by the
environment variable SYSTEMONE_DUMP. A fixture line holds the request, that prompt, its token
ids and the server's answer:

    {"state": ..., "question": {...}, "prompt": "...", "tokens": [...], "answer": {...}}

`crates/peregrine-core/tests/systemone_render.rs` renders every line and compares the prompt
byte for byte. The token ids and the answers are for tests that run with the model file.

Usage (standard library only):

    git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp && git checkout cd959ba
    git apply <peregrine>/scripts/systemone-dump.patch
    cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --target llama-server -j
    python3 scripts/systemone-fixtures.py --server llama.cpp/build/bin/llama-server \
        --model tinyopenjev-for-testing-Q8_0.gguf \
        --cases scripts/systemone-cases.json \
        --out crates/peregrine-core/tests/fixtures/systemone-openjev.jsonl
"""

import argparse
import json
import os
import subprocess
import tempfile
import time
import urllib.request


def post(port, body):
    data = json.dumps(body, ensure_ascii=False).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/systemone", data=data,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--port", type=int, default=8401)
    a = ap.parse_args()

    cases = json.load(open(a.cases, encoding="utf-8"))
    dump = tempfile.NamedTemporaryFile(prefix="systemone-dump-", suffix=".jsonl", delete=False)
    dump.close()
    env = dict(os.environ, SYSTEMONE_DUMP=dump.name)
    server = subprocess.Popen(
        [a.server, "-m", a.model, "--port", str(a.port), "-t", "4", "-np", "1"],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(120):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{a.port}/health", timeout=2) as r:
                    if r.status == 200:
                        break
            except OSError:
                time.sleep(0.5)
        else:
            raise SystemExit("llama-server did not come up")

        lines = []
        for case in cases:
            before = sum(1 for _ in open(dump.name, encoding="utf-8"))
            res = post(a.port, {"state": case["state"], "questions": {"q": case["question"]}})
            dumped = [json.loads(l) for l in open(dump.name, encoding="utf-8")][before:]
            if len(dumped) != 1:
                raise SystemExit(f"expected one prompt, got {len(dumped)}: {case}")
            lines.append({
                "state": case["state"],
                "question": case["question"],
                "prompt": dumped[0]["prompt"],
                "tokens": dumped[0]["tokens"],
                "answer": res["answers"]["q"],
            })
    finally:
        server.terminate()
        server.wait(timeout=30)
        os.unlink(dump.name)

    with open(a.out, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    print(f"{len(lines)} cases written to {a.out}")


if __name__ == "__main__":
    main()
