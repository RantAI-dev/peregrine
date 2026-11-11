# The `bert` family's real-file fixtures (S4.6a)

200 mixed-length English texts (16–428 tokens), and, per reference file, llama.cpp v0.9.0
(`cd959ba`)'s token ids and L2-normalised vectors and the fp32 PyTorch reference's vectors.
`crates/peregrine-models/tests/embeddings.rs` runs them when the model files are on the
machine (the test is `#[ignore]`d: CI has no GGUF files) and enforces the S4.6 bound.

## Provenance

| Fixture | Model file (SHA-256) |
|---|---|
| `minilm.ids`, `minilm_f16_llama.f32`, `minilm_q8_0_llama.f32`, `minilm_torch.f32` | `all-MiniLM-L6-v2-ggml-model-f16.gguf` (`second-state/All-MiniLM-L6-v2-Embedding-GGUF`, `797b70c4edf85907fe0a49eb85811256f65fa0f7bf52166b147fd16be2be4662`) and `all-MiniLM-L6-v2-Q8_0.gguf` (same repo, `263215c3cadd6e16740741a7624ab4cbb6c8e777688bd5331ecfbf5681c2f8ed`) |
| `bge_small.ids`, `bge_small_f32_llama.f32`, `bge_small_q8_0_llama.f32`, `bge_small_torch.f32` | `bge-small-en-v1.5-f32.gguf` (`CompendiumLabs/bge-small-en-v1.5-gguf`, `bf40c42ad7d89382e9ba7376d5c4b73f6b556cb541fab37aaa1da9c320149b65`) and `bge-small-en-v1.5-q8_0.gguf` (same repo, `ec38e8da142596baa913124ae50550de284b6916bf59577ef2f0cb9660c2f514`) |
| `bge_m3.ids`, `bge_m3_q8_0_llama.f32`, `bge_m3_torch.f32` | `bge-m3-Q8_0.gguf` (`gpustack/bge-m3-GGUF`, `950f4a8e5e19477a6d3c26d2f162233c20002c601f75e4b002e3239997821167`) |

The llama.cpp vectors are `/embedding` responses (`llama-server --embedding --pooling` the
file's own, `-ub 1024` so a 515-token XLM-R text fits), already normalised by
`common_embd_normalize(…, 2)`. The token ids are its `/tokenize` with `add_special: true`
(`[CLS] … [SEP]`, `<s> … </s>`). The torch vectors are `sentence-transformers` on the original
checkpoints (`sentence-transformers/all-MiniLM-L6-v2`, `BAAI/bge-small-en-v1.5`, `BAAI/bge-m3`), CPU torch,
`normalize_embeddings=True`, with `max_seq_length = 512` — the models' own default of 256
truncates the long texts and is not the comparison.

`*.f32` is little-endian f32, one vector after another, `texts.txt`'s order.
