#!/usr/bin/env python3
"""Tiny embedding server around ONNX Runtime alone, for benchmarks.

Not fastembed: this loads `model.onnx` directly, pads each request's texts to
the longest one, runs `last_hidden_state` and pools it (mean or cls) itself.
Runs from the benchmark venv under /var/tmp, never from the repository.
One inference at a time (a lock), batch = the texts of one request.
POST /embed {"texts": [...]} -> {"embeddings": [[...]], "tokens": n}
"""

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True, help="dir with model.onnx + tokenizer.json")
    ap.add_argument("--pooling", default="mean", choices=["mean", "cls"])
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--onnx-file", default="model.onnx")
    args = ap.parse_args()

    model_path = Path(args.model_path)
    tokenizer = Tokenizer.from_file(str(model_path / "tokenizer.json"))
    tokenizer.no_truncation()
    tokenizer.no_padding()

    so = ort.SessionOptions()
    so.intra_op_num_threads = args.threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(
        str(model_path / args.onnx_file), sess_options=so,
        providers=["CPUExecutionProvider"],
    )
    input_names = {i.name for i in sess.get_inputs()}

    lock = threading.Lock()

    def embed(texts):
        enc = tokenizer.encode_batch(texts, add_special_tokens=True)
        width = max(len(e.ids) for e in enc)
        n = len(enc)
        ids = np.zeros((n, width), dtype=np.int64)
        mask = np.zeros((n, width), dtype=np.int64)
        types = np.zeros((n, width), dtype=np.int64)
        for i, e in enumerate(enc):
            k = len(e.ids)
            ids[i, :k] = e.ids
            mask[i, :k] = e.attention_mask if e.attention_mask is not None else [1] * k
            if e.type_ids is not None:
                types[i, :k] = e.type_ids
        feeds = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in input_names:
            feeds["token_type_ids"] = types
        hidden = sess.run(None, feeds)[0].astype(np.float64)
        if args.pooling == "cls":
            vecs = hidden[:, 0, :]
        else:
            m = mask.astype(np.float64)[:, :, None]
            vecs = (hidden * m).sum(axis=1) / m.sum(axis=1)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs, sum(len(e.ids) for e in enc)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
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
            with lock:
                vecs, tokens = embed(texts)
            self._send(200, {"embeddings": [list(map(float, v)) for v in vecs],
                             "tokens": tokens})

    class Server(HTTPServer):
        request_queue_size = 128

    srv = Server(("127.0.0.1", args.port), Handler)
    srv.serve_forever()


if __name__ == "__main__":
    main()
