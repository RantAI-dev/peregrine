#!/usr/bin/env python3
"""Compute fp32 reference vectors from an ONNX model's sentence_embedding output.

For models whose PyTorch weights are licence-gated (EmbeddingGemma), the ONNX
export is the available fp32 reference.  The graph carries its own pooling and
normalisation head, so the vectors are directly comparable with what the GGUF
produces.  Writes the same JSON shape as `embed_ref.py`, with the label as the
backend, so it slots into `embed.py --mode accuracy --ref-label <label>`.

  python embed_ref_onnx.py --onnx DIR/model.onnx --tokenizer DIR/tokenizer.json \
      --texts texts.json --out ref/model-onnx-fp32.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--texts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="onnx-fp32")
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    data = json.loads(Path(args.texts).read_text(encoding="utf-8"))
    lengths = list(data["lengths"].keys())
    texts = []
    for L in lengths:
        texts.extend(t["text"] for t in data["lengths"][L])

    tok = Tokenizer.from_file(args.tokenizer)
    tok.enable_truncation(max_length=args.max_length)
    tok.enable_padding()
    enc = tok.encode_batch(texts)
    ids = np.array([e.ids for e in enc], dtype=np.int64)
    mask = np.array([e.attention_mask for e in enc], dtype=np.int64)

    so = ort.SessionOptions()
    so.intra_op_num_threads = args.threads
    sess = ort.InferenceSession(args.onnx, sess_options=so, providers=["CPUExecutionProvider"])
    out = sess.run(["sentence_embedding"], {"input_ids": ids, "attention_mask": mask})[0]
    vecs = out.astype(np.float64)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "backend": args.label,
        "model_dir": str(Path(args.onnx).resolve()),
        "dim": int(vecs.shape[1]),
        "count": len(texts),
        "lengths": lengths,
        "norms": {"min": float(np.linalg.norm(vecs, axis=1).min()),
                  "max": float(np.linalg.norm(vecs, axis=1).max()),
                  "mean": float(np.linalg.norm(vecs, axis=1).mean())},
        "vectors": vecs.round(6).tolist(),
    }), encoding="utf-8")
    print(f"wrote {args.out}: {len(texts)} vectors, dim {vecs.shape[1]}")


if __name__ == "__main__":
    main()
