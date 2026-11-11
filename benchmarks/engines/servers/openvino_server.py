#!/usr/bin/env python3
"""Tiny embedding server around sentence-transformers' OpenVINO backend.

Runs from the benchmark venv under /var/tmp, never from the repository.
One inference at a time (a lock), batch = the texts of one request.
POST /embed {"texts": [...]} -> {"embeddings": [[...]], "tokens": n}
"""

import argparse
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from tokenizers import Tokenizer

# The OpenVINO export writes hundreds of megabytes of intermediates; /tmp is a
# small tmpfs on this host, so put them on the real disk.
os.environ.setdefault("TMPDIR", "/var/tmp/ov-tmp")
Path("/var/tmp/ov-tmp").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True, help="dir with config/tokenizer (+ OV IR or safetensors)")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer

    tokenizer = Tokenizer.from_file(str(Path(args.model_path) / "tokenizer.json"))
    tokenizer.no_truncation()
    tokenizer.no_padding()

    ov_config = {
        "INFERENCE_NUM_THREADS": str(args.threads),
        "PERFORMANCE_HINT": "LATENCY",
        "PERFORMANCE_HINT_NUM_REQUESTS": "1",
    }
    model = SentenceTransformer(
        args.model_path,
        device=args.device,
        backend="openvino",
        model_kwargs={"ov_config": ov_config},
        trust_remote_code=False,
    )
    model.max_seq_length = 512
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
                arr = model.encode(
                    texts,
                    batch_size=max(1, len(texts)),
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
            self._send(200, {"embeddings": arr.tolist(), "tokens": tokens})

    class Server(HTTPServer):
        # Must be a class attribute: the listen backlog is applied in
        # server_activate() during __init__.  With the default of 5 a ten-client
        # cell drops SYNs, and every dropped connect costs a 1 s retransmit.
        request_queue_size = 128

    srv = Server(("127.0.0.1", args.port), Handler)
    srv.serve_forever()


if __name__ == "__main__":
    main()
