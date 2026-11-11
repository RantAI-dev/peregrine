# llama.cpp reference vectors for `gemma-embedding`

The cross-engine half of the S4.4 gate (`docs/design/s4-embeddings-decisions.md` §5 item 2):
200 texts and the vectors llama.cpp v0.9.0 (`cd959ba`) produces for them, so the ignored
real-model test can compare Peregrine's EmbeddingGemma 300M against the reference engine
without a llama.cpp build on the test machine.

## Files

| File | What it is |
|---|---|
| `texts.txt` | 200 texts, one per line, in the order of the vector rows |
| `llama_q8_0.f32` | 200 × 768 f32, little-endian, L2-normalised: llama.cpp's vectors for `embeddinggemma-300M-Q8_0.gguf` |
| `llama_f16.f32` | 200 × 768 f32, little-endian, L2-normalised: llama.cpp's vectors for the F16 reference below |

Width 256 is the design's Matryoshka truncation: take the first 256 components and normalise
again. The fixtures store the full 768-d vectors, so both widths are checked from one file.

## Texts

Generated with the S4.2 harness's deterministic generator
(`benchmarks/embed_texts.py` on `s4.2-embed-baselines`) and EmbeddingGemma's own tokenizer:
lengths 16/64/128/256/512, 40 texts each, interleaved (`--mixed`), capped at 451 tokens
(`--max-tokens 451`). Token counts run 16 to 451; the text itself is what is stored.

## Reference file (F16, with the dense head)

The uploads with an F32/F16 weight file (`unsloth/embeddinggemma-300m-GGUF`,
`second-state/embeddinggemma-300m-GGUF`, `gaianet/embeddinggemma-300m-GGUF`) were converted
without `--sentence-transformers-dense-modules`, so they carry 314 tensors and no
`dense_2`/`dense_3`. The full-precision reference is therefore converted from the ungated
original mirror `unsloth/embeddinggemma-300m` (commit `bfa3c84`) with llama.cpp `cd959ba`:

```sh
python3 convert_hf_to_gguf.py /path/to/unsloth-embeddinggemma-300m \
    --outtype f16 --sentence-transformers-dense-modules \
    --outfile embeddinggemma-300m-f16.gguf
```

That file is 621,867,232 bytes, SHA-256
`eeccb769249a2bac5e23b3e408c6ea17d016d6d8c273c3b5bdb23b7fdd955d44`, and carries 316
tensors including `dense_2` and `dense_3`. Its weights agree with the Q8_0 upload
(`ggml-org/embeddinggemma-300M-GGUF`, SHA-256
`b5ce9d77a3fc4b3b39ccb5643c36777911cc4eb46a66962eadfa3f5f60490d63`) to cosine 0.99997 on
every tensor sampled: it is the same model, unquantised.

Both vector files were collected from `llama-server --embedding --pooling mean -c 2048
-b 2048 -ub 2048 -t 8`, one text per request, on the H0 host (Xeon E5-2680 v4, AVX2).
