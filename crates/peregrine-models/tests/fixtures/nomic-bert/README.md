# The `nomic-bert` family's real-file fixtures (S4.8a)

200 texts with `search_document: ` in front, llama.cpp v0.9.0 (`cd959ba`)'s token
ids and L2-normalised vectors for nomic-embed-text v1.5 in F16 and Q8_0, and the F16
vectors that are the Q8_0 case's full-precision reference.
`crates/peregrine-models/tests/embeddings.rs` runs them when the model files are on
the machine (the test is `#[ignore]`d: CI has no GGUF files) and enforces the S4
bound.

## Provenance

| Fixture | Model file (SHA-256) |
|---|---|
| `ids.txt` | llama.cpp `/tokenize` with `add_special: true` (`[CLS] … [SEP]`). Identical for both files: they share the 30 522-token WordPiece vocabulary and WordPiece pre-tokenizer, checked on all 200 texts. |
| `f16_llama.f32` | `nomic-embed-text-v1.5.f16.gguf` (`nomic-ai/nomic-embed-text-v1.5-GGUF`, `f7af6f66802f4df86eda10fe9bbcfc75c39562bed48ef6ace719a251cf1c2fdb`) |
| `q8_0_llama.f32` | `nomic-embed-text-v1.5.Q8_0.gguf` (same repo, `3e24342164b3d94991ba9692fdc0dd08e3fd7362e0aacc396a9a5c54a544c3b7`) |

The llama.cpp vectors are `/v1/embeddings` responses from
`llama-server --embeddings -c 2048 -b 2048 -ub 2048 -t 4` (the model's own
context is 2048 tokens), already normalised by
`common_embd_normalize(..., 2)`; each text is one request, read one at a time. The
F16 file is the Q8_0 case's full-precision reference (llama.cpp on the F16
conversion), as the S4 gate states. `*.f32` is little-endian f32, one 768-d vector
after another, in `texts.txt`'s order.

The 195 short texts are the first 195 lines of
`crates/peregrine-models/tests/fixtures/bert/texts.txt`; the five long ones are
concatenations of those lines trimmed to 390, 793, 1199, 1998 and 2036 tokens
(the last encodes to 2040 ids, under the model's 2048-token context). Every text is
prefixed with `search_document: `, which the client writes — the model does not.

## Commands

```sh
# ids
curl -s http://127.0.0.1:PORT/tokenize -H 'content-type: application/json' \
  -d '({"content": "<text>", "add_special": true})'
# vectors
curl -s http://127.0.0.1:PORT/v1/embeddings -H 'content-type: application/json' \
  -d '({"input": "<text>", "encoding_format": "float"})'
```

## Measured

| File | ids | vs llama.cpp min / mean | vs the F16 reference min / mean |
|---|---|---|---|
| F16 | identical | 0.9999995 / 0.9999998 | 0.9999995 / 0.9999998 |
| Q8_0 | identical | 0.9997719 / 0.9998660 | 0.9978348 / 0.9982994 (llama.cpp's own Q8_0: 0.9978020 / 0.9982981) |

`docs/reference/nomic-bert.md` has the reading of llama.cpp's graph and the speed.
