#!/usr/bin/env python3
"""Turn the ignored real-file tests' output into the proof run's accuracy JSON.

`cargo test --release -p peregrine-models --test embeddings -- --ignored
--nocapture` prints, per model, the cosine of Peregrine's vectors to the fp32
reference and to llama.cpp's vectors (and, for the models with no PyTorch
reference here, to llama.cpp's F16 vectors). This script reads that captured
output and writes `raw/accuracy-real-file.json`, the shape
`benchmarks/proof_report.py` expects.

  python benchmarks/real_file_accuracy.py --log tests/real-file-tests.log \
      --out raw/accuracy-real-file.json
"""

import argparse
import json
from pathlib import Path
import re

# The model file names the test prints, and the row they belong to.
FILES = {
    "all-MiniLM-L6-v2-ggml-model-f16.gguf": ("minilm", "f16"),
    "all-MiniLM-L6-v2-Q8_0.gguf": ("minilm", "q8_0"),
    "bge-small-en-v1.5-f32.gguf": ("bge-small", "f32"),
    "bge-small-en-v1.5-q8_0.gguf": ("bge-small", "q8_0"),
    "bge-m3-Q8_0.gguf": ("bge-m3", "q8_0"),
}

NUM = r"([0-9.]+)"


def check(name, mean, minimum):
    return {"name": name, "cosine_mean": float(mean), "cosine_min": float(minimum)}


def parse(lines):
    records = {}

    def rec(model, engine, precision, reference):
        return records.setdefault(
            (model, engine, precision),
            {"model": model, "engine": engine, "precision": precision,
             "reference": reference, "cosine_mean": None, "cosine_min": None,
             "checks": []},
        )

    for line in lines:
        line = line.strip()
        # `cargo test` prints "test <name> ... " before the test's own output.
        line = re.sub(r"^test \S+ \.\.\. ?", "", line)
        # bert family: "<file>: vs fp32 reference min X mean Y (llama.cpp min A mean B)"
        m = re.match(rf"^(\S+\.gguf): vs fp32 reference min {NUM} mean {NUM}", line)
        if m and m.group(1) in FILES:
            model, prec = FILES[m.group(1)]
            r = rec(model, "peregrine", prec, "fp32 PyTorch")
            r["checks"].append(check("peregrine vs fp32 PyTorch", m.group(3), m.group(2)))
            r["cosine_mean"], r["cosine_min"] = float(m.group(3)), float(m.group(2))
            mm = re.search(rf"llama\.cpp min {NUM} mean {NUM}", line)
            if mm:
                r["checks"].append(check("llama.cpp vs fp32 PyTorch",
                                         mm.group(2), mm.group(1)))
            continue
        # bert family: "<file>: 200 texts, 384-d, vs llama.cpp min X mean Y"
        m = re.match(rf"^(\S+\.gguf): .* vs llama\.cpp min {NUM} mean {NUM}", line)
        if m and m.group(1) in FILES:
            model, prec = FILES[m.group(1)]
            r = rec(model, "peregrine", prec, "fp32 PyTorch")
            r["checks"].append(check("peregrine vs llama.cpp", m.group(3), m.group(2)))
            continue
        # EmbeddingGemma: "<d>-d: vs llama.cpp Q8_0 min X mean Y; vs the F16 reference ..."
        m = re.match(
            rf"^(768|256)-d: vs llama\.cpp Q8_0 min {NUM} mean {NUM}; "
            rf"vs the F16 reference Peregrine min {NUM} mean {NUM}, "
            rf"llama\.cpp min {NUM} mean {NUM}", line)
        if m:
            dims = m.group(1)
            r = rec("embeddinggemma", "peregrine", "q8_0", "llama.cpp F16")
            r["checks"].append(check(f"peregrine vs llama.cpp Q8_0 ({dims}-d)",
                                     m.group(3), m.group(2)))
            r["checks"].append(check(f"peregrine vs llama.cpp F16 ({dims}-d)",
                                     m.group(5), m.group(4)))
            r["checks"].append(check(f"llama.cpp Q8_0 vs llama.cpp F16 ({dims}-d)",
                                     m.group(7), m.group(6)))
            if dims == "768":
                r["cosine_mean"], r["cosine_min"] = float(m.group(5)), float(m.group(4))
            continue
        # Nemotron F16: "F16 2048-d vs llama.cpp F16: min X mean Y"
        m = re.match(rf"^F16 (2048|1024|512)-d vs llama\.cpp F16: min {NUM} mean {NUM}", line)
        if m:
            r = rec("nemotron-3.5-embed-1b", "peregrine", "f16", "llama.cpp F16")
            r["checks"].append(check(f"peregrine F16 vs llama.cpp F16 ({m.group(1)}-d)",
                                     m.group(3), m.group(2)))
            if m.group(1) == "2048":
                r["cosine_mean"], r["cosine_min"] = float(m.group(3)), float(m.group(2))
            continue
        # Nemotron Q8_0: "Q8_0 <d>-d: vs llama.cpp Q8_0 min X mean Y; vs the F16 reference ..."
        m = re.match(
            rf"^Q8_0 (2048|1024|512)-d: vs llama\.cpp Q8_0 min {NUM} mean {NUM}; "
            rf"vs the F16 reference Peregrine min {NUM} mean {NUM}, "
            rf"llama\.cpp min {NUM} mean {NUM}", line)
        if m:
            dims = m.group(1)
            r = rec("nemotron-3.5-embed-1b", "peregrine", "q8_0", "llama.cpp F16")
            r["checks"].append(check(f"peregrine Q8_0 vs llama.cpp Q8_0 ({dims}-d)",
                                     m.group(3), m.group(2)))
            r["checks"].append(check(f"peregrine Q8_0 vs llama.cpp F16 ({dims}-d)",
                                     m.group(5), m.group(4)))
            r["checks"].append(check(f"llama.cpp Q8_0 vs llama.cpp F16 ({dims}-d)",
                                     m.group(7), m.group(6)))
            if dims == "2048":
                r["cosine_mean"], r["cosine_min"] = float(m.group(5)), float(m.group(4))
            continue
        # Qwen3 prints no cosine: it asserts bit-identical vectors.
        if "Qwen3-Embedding-0.6B-Q8_0.gguf" in line:
            r = rec("qwen3-embed-0.6b", "peregrine", "q8_0", "reference forward")
            r["checks"].append({
                "name": "bit-identical to the reference forward and to llama.cpp ids; "
                        "no cosine printed",
                "cosine_mean": None, "cosine_min": None,
            })
        if "by text: 64 ids match llama.cpp" in line:
            r = rec("nemotron-3.5-embed-1b", "peregrine", "q8_0", "llama.cpp F16")
            r["checks"].append({
                "name": "64 ids match llama.cpp; 8 vectors bit-identical by text and ids",
                "cosine_mean": None, "cosine_min": None,
            })
    return list(records.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    lines = args.log.read_text(encoding="utf-8", errors="replace").splitlines()
    records = parse(lines)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"source": str(args.log), "records": records}, indent=1),
                        encoding="utf-8")
    for r in records:
        print(f"{r['model']} {r['engine']} {r['precision']}: "
              f"{len(r['checks'])} checks, headline {r['cosine_mean']}")


if __name__ == "__main__":
    main()
