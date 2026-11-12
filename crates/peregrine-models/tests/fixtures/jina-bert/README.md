# The `jina-bert-v2` family's real-file fixtures (S4.8c)

200 mixed-length English texts (16–8180 tokens; the five long ones run to the
model's 8192-token context, so ALiBi's bias spans the whole range), llama.cpp
v0.9.0 (`cd959ba`)'s token ids and L2-normalised vectors for two files, and the
F16 vectors that are the Q8_0 case's full-precision reference.
`crates/peregrine-models/tests/embeddings.rs` runs them when the model files are on
the machine (the test is `#[ignore]`d: CI has no GGUF files) and enforces the S4
bound.

## Provenance

| Fixture | Model file (SHA-256) |
|---|---|
| `ids.txt` | llama.cpp `/tokenize` with `add_special: true` (`[CLS] … [SEP]`). One vocabulary, checked on all 200 texts. |
| `jina_f16_llama.f32` | `jina-embeddings-v2-base-en-FP16.gguf` (`gpustack/jina-embeddings-v2-base-en-GGUF`, `8d29fa0c9b806ae5c165eaf1d8053820760c5d4a3b4f57dd203c426c4680c5f0`) |
| `jina_q8_0_llama.f32` | `jina-embeddings-v2-base-en-Q8_0.gguf` (same repo, `123b13ec105de8e0d2becd7dca4f4c46cef3a535f6de9ff1b6ade3fbbf6f5184`) |

The llama.cpp vectors are `/v1/embeddings` responses from
`llama-server --embeddings -c 8192 -b 8192 -ub 8192 -t 4`, already normalised by
`common_embd_normalize(..., 2)`; the text's own vector is read one request at a
time. The F16 file is the Q8_0 case's full-precision reference (llama.cpp on the
F16 conversion), as the S4 gate states. `*.f32` is little-endian f32, one 768-d
vector after another, `texts.txt`'s order.

`texts.txt`'s first 195 lines are the first 195 lines of
`crates/peregrine-models/tests/fixtures/bert/texts.txt`; the five long ones are
concatenations of those lines trimmed (by re-tokenising) to 512, 1024, 2048, 4096
and 8180 tokens.

Commands, from this directory (the servers run one at a time):

```sh
# ids
curl -s http://127.0.0.1:PORT/tokenize -H 'content-type: application/json' \
  -d '{"content": "<text>", "add_special": true}'
# vectors
curl -s http://127.0.0.1:PORT/v1/embeddings -H 'content-type: application/json' \
  -d '{"input": "<text>", "encoding_format": "float"}'
```

## The measured numbers

F16: ids identical on all 200 texts; cosine against llama.cpp's F16 min
0.9999999 / mean 1.0000000. Q8_0: ids identical; against the F16 reference min
0.9993389 / mean 0.9996382 (llama.cpp's own Q8_0: min 0.9992368 / mean 0.9996404),
and against llama.cpp's Q8_0 min 0.9998662 / mean 0.9999650. The test asserts
min 0.9992 / mean 0.9996 for the Q8_0 distance to the reference, and the S4
gate's clause that Peregrine's mean is no more than 1e-4 below llama.cpp's own
(`docs/reference/jina-bert.md`).
