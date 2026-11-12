# The `nomic-bert-moe` family's real-file fixtures (S4.8b)

200 texts with `search_document: ` in front, llama.cpp v0.9.0 (`cd959ba`)'s token
ids and L2-normalised vectors for nomic-embed-text-v2-moe in F16 and Q8_0, and the
F16 vectors that are the Q8_0 case's full-precision reference.
`crates/peregrine-models/tests/embeddings.rs` runs them when the model files are on
the machine (the test is `#[ignore]`d: CI has no GGUF files) and reports the S4
bound.

## Provenance

| Fixture | Model file (SHA-256) |
|---|---|
| `ids.txt` | llama.cpp `/tokenize` with `add_special: true` (`<s> … </s>`). Identical for both files: they share the 250 048-token XLM-R unigram vocabulary, checked on all 200 texts. |
| `f16_llama.f32` | `nomic-embed-text-v2-moe.f16.gguf` (`nomic-ai/nomic-embed-text-v2-moe-GGUF`, `a5db3381f2e514d3490a3a31fe70eb1a65e95016c85c6c2c23223b810806594f`, 957 680 480 bytes) |
| `q8_0_llama.f32` | `nomic-embed-text-v2-moe.Q8_0.gguf` (same repo, `06e7a7e594a26985523c18383aba4aad39fe6e14f08ffc6ab5b554e1ccdc3cff`, 512 225 120 bytes) |

The llama.cpp vectors are `/v1/embeddings` responses from
`llama-server --embedding --pooling mean -c 512 -b 512 -ub 512 -t 4` (the model's
own context is 512 tokens), already normalised by
`common_embd_normalize(..., 2)`; each text is one request, read one at a time. The
F16 file is the Q8_0 case's full-precision reference (llama.cpp on the F16
conversion), as the S4 gate states. `*.f32` is little-endian f32, one 768-d vector
after another, in `texts.txt`'s order.

The 200 texts are 17 to 511 tokens. 156 are the first 156 lines of
`crates/peregrine-models/tests/fixtures/bert/texts.txt` (trimmed only if they
exceeded 508 tokens); 24 are multilingual sentences (French, Spanish, German,
Italian, Portuguese, Polish, Dutch, Turkish, Japanese, Vietnamese, Russian,
Indonesian, Arabic, Czech, Romanian, Swedish, Greek, Ukrainian, Chinese, Hindi,
Korean, Thai, Hebrew, Persian); and 20 are concatenations of the remaining `bert`
lines trimmed to 470, 481, 492, 501, 505, 507 and then 509-511 ids, so several texts
run the whole 512-token context. Every text is prefixed with `search_document: `,
which the client writes — the model does not.

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
| F16 | identical | 0.9997738 / 0.9999977 | 0.9997738 / 0.9999977 |
| Q8_0 | identical | 0.9987744 / 0.9997209 | 0.9989152 / 0.9996520 (llama.cpp's own Q8_0: 0.9990825 / 0.9994075) |

`docs/reference/nomic-bert-moe.md` has the reading of llama.cpp's graph, why the
F16 minimum is one text at the last MoE layer's router boundary, and the speed.
