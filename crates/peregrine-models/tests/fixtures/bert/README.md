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

## The rerank pairs (S4.6b)

`rerank/` holds one fixed set of (query, document) pairs and, per reranker file, llama.cpp's
own pair encoding and its `/v1/rerank` scores. `crates/peregrine-models/tests/rerank_real.rs`
runs them when the GGUF files are on the machine (the test is `#[ignore]`d: CI has none).

| Fixture | Contents |
|---|---|
| `rerank/texts.tsv` | 320 pairs: 40 queries of eight documents, backslash-escaped, `query<TAB>document` |
| `rerank/v2_m3.ids` | llama.cpp's `format_prompt_rerank` ids for each pair (a list per line) |
| `rerank/v2_m3_q8_0_llama.f32` | llama.cpp's score per pair on the Q8_0 file, little-endian f32 |
| `rerank/v2_m3_fp16_llama.f32` | the same pairs on the FP16 conversion |
| `rerank/base.ids`, `rerank/base_q8_0_llama.f32` | the second reranker's ids and scores; one pair is past its 512-token table and is stored as NaN |

The queries are five each in English, Chinese, French, German, Spanish, Russian, Arabic and
Japanese. A query's eight documents are the right answer and a wrong one in the query's language,
four medium English passages, and two English passages from `texts.txt` — so a set mixes scripts
and lengths from 3 to 428 tokens.

| File | Repo | SHA-256 |
|---|---|---|
| `bge-reranker-v2-m3-Q8_0.gguf` | `gpustack/bge-reranker-v2-m3-GGUF` | `a43c7c9b11a4c1517e5bf95151960e1621d1b72f7a493364b01e386cf1aaa1d3` |
| `bge-reranker-v2-m3-FP16.gguf` (reference only) | same | `5df93be121c09c43432102ad2b9569d369ccb85c209ca7583e8ccd28f0e41b88` |
| `bge-reranker-base-q8_0.gguf` | `xinming0111/bge-reranker-base-Q8_0-GGUF` | `4bf3f9269c55954a5ad4d9e0e7fb05b448cb1053818170f497242ff760fd46bb` |

The scores come from `llama-server --reranking` at v0.9.0 (`cd959ba`), one request per query
(one per pair for the 512-token file), and the ids from `scripts/tokenizer-pair-oracle.cpp`,
which calls llama.cpp's `format_prompt_rerank` rather than re-implementing it. Regenerate with:

```text
python3 scripts/rerank-fixtures.py --pairs /tmp/pairoracle \
    --base-url http://127.0.0.1:18111 --model bge-reranker-v2-m3-Q8_0.gguf \
    --out crates/peregrine-models/tests/fixtures/bert/rerank \
    --ids v2_m3.ids --scores v2_m3_q8_0_llama.f32
```

The head of a `bert` reranker has no published WordPiece file: `bge-reranker-v2-m3` and
`bge-reranker-base` are both XLM-R (unigram). WordPiece pair encoding is pinned by the pair files
of `crates/peregrine-core/tests/fixtures/tokenize-golden-wpm-*-pair.tsv` and by the synthetic
WordPiece head model instead.
