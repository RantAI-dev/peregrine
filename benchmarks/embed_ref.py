#!/usr/bin/env python3
"""Compute the fp32 sentence-transformers (PyTorch) reference vectors.

Runs from the benchmark venv. The vectors are written once per model and reused
as the accuracy column's ground truth for every engine and precision.

  python embed_ref.py --model-dir DIR --texts texts.json --out ref.json
"""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--texts", required=True, help="texts JSON from embed_texts.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--lengths", default="", help="optional subset, e.g. 16,128")
    ap.add_argument("--label", default="torch-fp32")
    args = ap.parse_args()

    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(args.threads)
    data = json.loads(Path(args.texts).read_text(encoding="utf-8"))
    lengths = [x for x in args.lengths.split(",") if x] or list(data["lengths"].keys())
    texts = []
    for L in lengths:
        texts.extend(t["text"] for t in data["lengths"][L])

    model = SentenceTransformer(
        args.model_dir,
        device="cpu",
        model_kwargs={"dtype": torch.float32, "torch_dtype": torch.float32},
    )
    vecs = model.encode(
        texts,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    dim = int(vecs.shape[1])
    norms = np.linalg.norm(vecs, axis=1)
    out = {
        "backend": args.label,
        "model_dir": str(Path(args.model_dir).resolve()),
        "dim": dim,
        "count": len(texts),
        "lengths": lengths,
        "norms": {"min": float(norms.min()), "max": float(norms.max()),
                  "mean": float(norms.mean())},
        "vectors": vecs.round(6).tolist(),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out), encoding="utf-8")
    print(f"wrote {args.out}: {len(texts)} vectors, dim {dim}, "
          f"norm {out['norms']['min']:.4f}..{out['norms']['max']:.4f}")


if __name__ == "__main__":
    main()
