#!/usr/bin/env python3
"""Render the tool-calling cases with llama.cpp and write the fixtures (S6.2).

For every case in `scripts/tool-cases.json` this posts `{"messages", "tools"}` to
llama-server's `/apply-template` and records the prompt it builds. Peregrine's
renderer must produce the same bytes (`crates/peregrine-core/tests/tools_render.rs`).

    python3 scripts/tool-fixtures.py \
        --server /path/to/llama.cpp/build/bin/llama-server \
        --model /path/to/any-small-chat-model.gguf \
        [--template crates/peregrine-core/src/tools/qwen3.jinja] \
        --out crates/peregrine-core/tests/fixtures/tools-qwen25.jsonl

`--template` renders with that template file in place of the model's own
(`--chat-template-file`), so a template can be exercised without its model: the
template, not the weights, decides the prompt.

The request body is stored as the exact text that was sent, so the test parses
the same bytes llama.cpp parsed (number spellings and key order included).
"""

import argparse
import json
import pathlib
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def post(port, path, body):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=body.encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, r.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--template")
    ap.add_argument("--cases", default="scripts/tool-cases.json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    spec = json.loads(pathlib.Path(a.cases).read_text(encoding="utf-8"))
    port = free_port()
    cmd = [a.server, "-m", a.model, "--jinja", "--port", str(port), "-c", "1024", "-t", "2"]
    if a.template:
        cmd += ["--chat-template-file", a.template]
    srv = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(240):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                    if r.status == 200:
                        break
            except (urllib.error.URLError, ConnectionError, OSError):
                pass
            if srv.poll() is not None:
                sys.exit("llama-server exited before it was ready")
            time.sleep(0.5)
        else:
            sys.exit("llama-server did not become ready")

        lines = []
        for case in spec["cases"]:
            req = {"messages": case["messages"]}
            if case.get("tools"):
                req["tools"] = [spec["tools"][t] for t in case["tools"]]
            body = json.dumps(req, ensure_ascii=False)
            status, text = post(port, "/apply-template", body)
            rec = {"name": case["name"], "request": body}
            if status == 200:
                rec["prompt"] = json.loads(text)["prompt"]
            else:
                # A request llama.cpp refuses is recorded as refused: Peregrine must
                # refuse it too, not invent a rendering.
                rec["error"] = json.loads(text).get("error", {}).get("message", text)
            lines.append(json.dumps(rec, ensure_ascii=False))
        pathlib.Path(a.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"{len(lines)} cases -> {a.out}")
    finally:
        srv.terminate()
        try:
            srv.wait(timeout=10)
        except subprocess.TimeoutExpired:
            srv.kill()


if __name__ == "__main__":
    main()
