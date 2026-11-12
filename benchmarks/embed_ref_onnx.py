#!/usr/bin/env python3
"""Compute fp32 reference vectors from an fp32 ONNX export.

For models whose PyTorch weights cannot be loaded on this host (EmbeddingGemma is
licence-gated; nomic-bert and jina-bert-v2 need remote code that does not load on
this transformers), the fp32 ONNX export is the available full-precision
reference.  Two export shapes are handled:

  sentence_embedding   the graph carries its own pooling and normalisation head
                       (the EmbeddingGemma and jina exports)
  last_hidden_state    the graph returns per-token rows; this script pools them
                       (mean or cls) exactly as the benchmark servers do

Writes the same JSON shape as `embed_ref.py`, with the label as the backend, so
it slots into `embed.py --mode accuracy --ref-label <label>`.

  python embed_ref_onnx.py --onnx DIR/model.onnx --tokenizer DIR/tokenizer.json \
      --texts texts.json --out ref/model-onnx-fp32.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


def pool(rows: np.ndarray, mask: np.ndarray, how: str) -> np.ndarray:
    """Pool `rows` (batch, seq, dim) with `mask` (batch, seq), as the servers do."""
    if how == "cls":
        return rows[:, 0, :]
    m = mask.astype(np.float64)[:, :, None]
    return (rows * m).sum(axis=1) / m.sum(axis=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--texts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="onnx-fp32")
    ap.add_argument("--output-name", default="sentence_embedding",
                    help="the graph output to read (sentence_embedding or last_hidden_state)")
    ap.add_argument("--pooling", default="none", choices=["none", "mean", "cls"],
                    help="pool last_hidden_state rows; `none` means the output is already pooled")
    ap.add_argument("--max-length", type=int, default=0,
                    help="truncate to this length; 0 keeps every token")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    data = json.loads(Path(args.texts).read_text(encoding="utf-8"))
    lengths = list(data["lengths"].keys())
    texts = []
    for L in lengths:
        texts.extend(t["text"] for t in data["lengths"][L])

    tok = Tokenizer.from_file(args.tokenizer)
    if args.max_length:
        tok.enable_truncation(max_length=args.max_length)
    tok.enable_padding()

    so = ort.SessionOptions()
    so.intra_op_num_threads = args.threads
    sess = ort.InferenceSession(args.onnx, sess_options=so, providers=["CPUExecutionProvider"])
    names = {i.name for i in sess.get_inputs()}

    vecs = []
    for start in range(0, len(texts), args.batch_size):
        enc = tok.encode_batch(texts[start:start + args.batch_size])
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
        feeds = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in names:
            feeds["token_type_ids"] = np.zeros_like(ids)
        out = sess.run([args.output_name], feeds)[0]
        if args.output_name == "last_hidden_state":
            if args.pooling == "none":
                raise SystemExit("a last_hidden_state export needs --pooling mean or cls")
            out = pool(out, mask, args.pooling)
        vecs.append(out.astype(np.float64))
    vecs = np.concatenate(vecs, axis=0)
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
