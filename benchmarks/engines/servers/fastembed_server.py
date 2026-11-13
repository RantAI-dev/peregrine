#!/usr/bin/env python3
"""Tiny embedding server around `fastembed` (ONNX Runtime), for benchmarks.

Runs from the benchmark venv under /var/tmp, never from the repository.
One inference at a time (a lock), batch = the texts of one request.
POST /embed {"texts": [...]} -> {"embeddings": [[...]], "tokens": n}
"""

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from fastembed import TextEmbedding
from tokenizers import Tokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--model-path", required=True, help="dir with model.onnx + tokenizer.json")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--cache-dir", default="/var/tmp/embed-venv/fastembed-cache")
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()

    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer.from_file(str(Path(args.model_path) / "tokenizer.json"))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    model = TextEmbedding(
        model_name=args.model_name,
        specific_model_path=args.model_path,
        threads=args.threads,
        cache_dir=args.cache_dir,
        providers=["CPUExecutionProvider"],
    )
    # One inference at a time: a single-threaded server queues requests in the
    # socket backlog, which is how these engines are actually embedded.  A
    # threaded server with a lock makes ten clients contend on the GIL and
    # collapses aggregate throughput, which measures Python, not ONNX Runtime.
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        # Headers and body are separate small writes; without TCP_NODELAY a
        # 40 ms delayed-ACK stalls every response (~100 ms per request with
        # several connections open).  That would measure the kernel, not ONNX.
        disable_nagle_algorithm = True

        def log_message(self, fmt, *a):
            pass

        def _send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/health", "/v1/models"):
                self._send(200, {"status": "ok"})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path not in ("/embed", "/v1/embeddings"):
                self._send(404, {"error": "not found"})
                return
            n = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(n).decode())
            texts = payload.get("texts")
            if texts is None:
                texts = payload.get("input")
            if isinstance(texts, str):
                texts = [texts]
            enc = tokenizer.encode_batch(texts, add_special_tokens=True)
            tokens = sum(len(e.ids) for e in enc)
            with lock:
                vecs = [list(map(float, v)) for v in model.embed(
                    texts, batch_size=max(1, min(args.batch_size, len(texts))))]
            self._send(200, {"embeddings": vecs, "tokens": tokens})

    class Server(HTTPServer):
        # Must be a class attribute: the listen backlog is applied in
        # server_activate() during __init__.  With the default of 5 a ten-client
        # cell drops SYNs, and every dropped connect costs a 1 s retransmit.
        request_queue_size = 128

    srv = Server(("127.0.0.1", args.port), Handler)
    srv.serve_forever()


if __name__ == "__main__":
    main()
