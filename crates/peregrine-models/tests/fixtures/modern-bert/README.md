# The `modern-bert` family's real-file fixtures (S4.7)

200 mixed-length English texts (16–6976 tokens; five are longer than the 128-token
local window, so both mask rules run), llama.cpp v0.9.0 (`cd959ba`)'s token ids and
L2-normalised vectors for four files, and the F16 vectors that are the Q8_0 cases'
full-precision reference. `crates/peregrine-models/tests/embeddings.rs` runs them
when the model files are on the machine (the test is `#[ignore]`d: CI has no GGUF
files) and enforces the S4 bound.

## Provenance

| Fixture | Model file (SHA-256) |
|---|---|
| `ids.txt` | llama.cpp `/tokenize` with `add_special: true` (`<bos> … <eos>`). Identical for both models: they share the ModernBERT 50 368-token vocabulary and `modern-bert` pre-tokenizer, checked on all 200 texts. |
| `gte_f16_llama.f32` | `gte-modernbert-base-F16.gguf` (`keisuke-miyako/gte-modernbert-base-gguf`, `2cda419ce09e87b1dd5294177a1c7d563e9536700e33652a9bb9e9c503c4e437`) |
| `gte_q8_0_llama.f32` | `gte-modernbert-base-Q8_0.gguf` (same repo, `d3c637bb219c4900ef56f4fe2fde2cf0f03bed0de5476e0141e049e68bc93301`) |
| `granite_f16_llama.f32` | `granite-embedding-small-english-r2.f16.gguf` (`mradermacher/granite-embedding-small-english-r2-GGUF`, `6f5f7c70199658f54547b35174b50eef635ccc6d5b8e171dc635c201af4b2806`) |
| `granite_q8_0_llama.f32` | `granite-embedding-small-english-r2.Q8_0.gguf` (same repo, `d4b41f5d7db712806722103a1c6aba2f0fe99f77740501d6f313c8240641f145`) |

The llama.cpp vectors are `/v1/embeddings` responses from
`llama-server --embeddings -c 8192 -b 8192 -ub 8192 -t 4`, already normalised by
`common_embd_normalize(..., 2)`; the text's own vector is read one request at a
time. The gte F16 file is the Q8_0 cases' full-precision reference (llama.cpp on
the F16 conversion), as the S4 gate states. `*.f32` is little-endian f32, one
768-d (gte) or 384-d (Granite) vector after another, `texts.txt`'s order.

Commands, from this directory (the servers run one at a time):

```sh
# ids
curl -s http://127.0.0.1:PORT/tokenize -H 'content-type: application/json' \
  -d '({"content": "<text>", "add_special": true})'
# vectors
curl -s http://127.0.0.1:PORT/v1/embeddings -H 'content-type: application/json' \
  -d '({"input": "<text>", "encoding_format": "float"})'
```

The 195 short texts are the first 195 lines of
`crates/peregrine-models/tests/fixtures/bert/texts.txt`; the five long ones are
concatenations of those lines trimmed to 390, 793, 1199, 1998 and 6976 tokens.

## The rerank pairs (S4.7b)

`rerank/` holds 100 (query, document) pairs built from `../bert/texts.txt`, llama.cpp's own
`format_prompt_rerank` ids and its `/v1/rerank` scores on gte-reranker-modernbert-base F16
and Q8_0; its README lists the files, the SHAs and the commands.
`crates/peregrine-models/tests/rerank_real.rs` runs them when the GGUF files are on the
machine (the test is `#[ignore]`d: CI has none).
