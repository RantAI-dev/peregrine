# llama.cpp reference vectors for Nemotron 3.5 Embed 1B (`mistral3`)

The cross-engine half of the S4.14b gate (`docs/design/s4-embeddings-decisions.md` §5 item 2):
64 texts and the vectors **llama.cpp v0.9.0 (`cd959ba`)** produces for them, so the ignored
real-model test can compare Peregrine against the reference engine without a llama.cpp build
on the test machine.

## Files

| File | What it is |
|---|---|
| `texts.txt` | 64 texts, one per line, in the order of the vector rows |
| `tokens.txt` | the 64 token id lists llama.cpp's `/tokenize` returned (`add_special` and `parse_special`), one text per line, space-separated |
| `llama_f16.f32` | 64 × 2048 f32, little-endian, L2-normalised: llama.cpp's vectors for the F16 upload |
| `llama_q8_0.f32` | 64 × 2048 f32, little-endian, L2-normalised: llama.cpp's vectors for the Q8_0 upload |

Widths 1024 and 512 are the design's Matryoshka truncations: take the first 1024 (or 512)
components and normalise again. The fixtures store the full 2048-d vectors, so every width is
checked from one file.

## Texts

Mixed length, 7 to 1520 tokens (one 8-token text, half a dozen short ones, a spread of 22 to
279, and four around 1350–1520). Half start with `query: `, half with `passage: `, as the model
card asks. Ten texts are non-English: Chinese, Japanese, Russian, Arabic, Hindi, French,
Spanish, German, Korean and Portuguese. The long texts repeat one English paragraph, so their
vectors measure the model's long-context path rather than new content.

## The files

| File | Bytes | SHA-256 |
|---|---|---|
| `Nemotron-3.5-Embed-1B-F16.gguf` | 2289844320 | `e1346f416e2f945911bac7a80f449eb2218b54921f526fc15d7b86218b7211b5` |
| `Nemotron-3.5-Embed-1B-Q8_0.gguf` | 1220296800 | `27c8e82c48828017c29f16fa87eb8958b19d3e35d8f25d73bf51b61efb2ec298` |
| `Nemotron-3.5-Embed-1B-Q4_K_M.gguf` | 749355104 | `3a2d43e650f4b92bd248e07557f3817dd0ba6dce98c014cc7e46be2f03ce2127` |

The three are the same model at three precisions; the F16 upload is the reference (the
unquantised weights).

## How the vectors were made

`llama-server` v0.9.0, one model at a time, one text per request:

```sh
llama-server -m Nemotron-3.5-Embed-1B-<tag>.gguf --embeddings --port 18093 \
    -c 2048 -b 2048 -ub 2048 -t 4 -np 1 --no-webui
```

For each text: `POST /tokenize` with `{"content": <text>, "add_special": true,
"parse_special": true}` for the ids, then `POST /v1/embeddings` with
`{"input": [<text>], "encoding_format": "float"}`. The server normalises; the vectors are
stored as returned. The ids are also what the test feeds Peregrine, so the test needs no
tokenizer.

The upload's `tokenizer.ggml.pre` is `pixtral` (Mistral's Tekken pre-tokenizer, roadmap S4.14a).
Until that lands in the build under test, the measurement was run on byte-identical copies of
the three files with only that metadata string changed to `default` (same length, so the rest of
the file and every weight byte is unchanged). The fixture's ids and vectors come from the
unmodified uploads.

## Measured accuracy

Peregrine, 4 threads, on the 64 texts, cosine of L2-normalised vectors; the F16 reference is
llama.cpp on the F16 upload. Run from this PR's test
(`crates/peregrine-models/tests/embeddings.rs`, `--ignored`).

F16 against llama.cpp's F16:

| Width | min | mean |
|---|---|---|
| 2048 | 0.9999998 | 0.99999994 |
| 1024 | 0.9999996 | 0.99999992 |
| 512 | 0.9999992 | 0.99999989 |

Q8_0, three-way (llama.cpp's Q8_0 and Peregrine's Q8_0 against the F16 reference, and the two
engines against each other):

| Width | llama.cpp Q8_0 vs F16 (min / mean) | Peregrine Q8_0 vs F16 (min / mean) | Peregrine vs llama.cpp Q8_0 (min / mean) |
|---|---|---|---|
| 2048 | 0.999762 / 0.9998549 | 0.999775 / 0.9998594 | 0.999944 / 0.9999777 |
| 1024 | 0.999680 / 0.9998384 | 0.999686 / 0.9998428 | 0.999944 / 0.9999774 |
| 512 | 0.999507 / 0.9998062 | 0.999519 / 0.9998107 | 0.999940 / 0.9999766 |

Q4_K_M, measured once with llama.cpp's Q4_K_M vectors (not committed):

| Width | llama.cpp Q4_K_M vs F16 (min / mean) | Peregrine Q4_K_M vs F16 (min / mean) | Peregrine vs llama.cpp Q4_K_M (min / mean) |
|---|---|---|---|
| 2048 | 0.9870541 / 0.99242838 | 0.9866695 / 0.99241421 | 0.9996871 / 0.99986461 |
| 1024 | 0.9850565 / 0.99178078 | 0.9847100 / 0.99177184 | 0.9996796 / 0.99986132 |
| 512 | 0.9783365 / 0.99120549 | 0.9786488 / 0.99119499 | 0.9996678 / 0.99985628 |

The test's bounds are the design's: F16 min ≥ 0.9999; Q8_0 cross-engine min ≥ 0.9995 and
mean ≥ 0.9999; and Peregrine Q8_0 no further from the F16 reference than llama.cpp's Q8_0 is
(1e-4 mean slack, 1e-3 min gross-break slack).
