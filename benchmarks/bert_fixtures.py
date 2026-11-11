#!/usr/bin/env python3
"""Turn the bert family's committed fixtures into the accuracy harness's inputs.

`crates/peregrine-models/tests/fixtures/bert/` holds 200 mixed-length texts and,
per model, the fp32 PyTorch (sentence-transformers) reference as little-endian
f32 vectors.  `benchmarks/embed.py --mode accuracy` wants a texts JSON and a
reference JSON, so this writes both from the fixtures (no model is run):

  <texts>/<model>-acc.json                 one length key, "fixture"
  <texts>/ref/<model>-torch-fp32.json      the same vectors as a JSON list

  python benchmarks/bert_fixtures.py --fixtures DIR --texts-dir DIR
"""

import argparse
import json
from pathlib import Path

import numpy as np

MODELS = {
    "minilm": ("minilm_torch.f32", 384),
    "bge-small": ("bge_small_torch.f32", 384),
    "bge-m3": ("bge_m3_torch.f32", 1024),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", required=True,
                    help="crates/peregrine-models/tests/fixtures/bert")
    ap.add_argument("--texts-dir", required=True)
    args = ap.parse_args()

    fixtures = Path(args.fixtures)
    out = Path(args.texts_dir)
    (out / "ref").mkdir(parents=True, exist_ok=True)
    texts = [ln.rstrip("\n") for ln in
             (fixtures / "texts.txt").read_text(encoding="utf-8").splitlines()]
    texts = [t for t in texts if t]
    for model, (fname, dim) in MODELS.items():
        vectors = np.fromfile(fixtures / fname, dtype="<f4")
        assert vectors.size == len(texts) * dim, (model, vectors.size, len(texts), dim)
        vectors = vectors.reshape(len(texts), dim).astype(np.float64)
        norms = np.linalg.norm(vectors, axis=1)
        vectors /= norms[:, None]
        acc = {
            "tokenizer": model,
            "per_length": len(texts),
            "max_tokens": 512,
            "lengths": {"fixture": [{"text": t, "tokens": 0} for t in texts]},
        }
        (out / f"{model}-acc.json").write_text(json.dumps(acc), encoding="utf-8")
        ref = {
            "backend": "torch-fp32",
            "label": "torch-fp32",
            "model_dir": f"crates/peregrine-models/tests/fixtures/bert/{fname}",
            "dim": dim,
            "count": len(texts),
            "lengths": ["fixture"],
            "norms": {"min": float(norms.min()), "max": float(norms.max()),
                      "mean": float(norms.mean())},
            "vectors": vectors.round(6).tolist(),
        }
        (out / "ref" / f"{model}-torch-fp32.json").write_text(
            json.dumps(ref), encoding="utf-8")
        print(f"{model}: {len(texts)} texts, dim {dim}, wrote {out}/{model}-acc.json")


if __name__ == "__main__":
    main()
