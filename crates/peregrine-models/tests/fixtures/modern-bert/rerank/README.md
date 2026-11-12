# The `modern-bert` rerank fixtures (S4.7b)

100 (query, document) pairs built from `../bert/texts.txt`, and llama.cpp v0.9.0
(`cd959ba`)'s own pair encoding and `/v1/rerank` scores for
`gte-reranker-modernbert-base`. `crates/peregrine-models/tests/rerank_real.rs` runs them
when the two GGUF files are on the machine (the test is `#[ignore]`d: CI has none).

| Fixture | Contents |
|---|---|
| `rerank/texts.tsv` | 100 pairs: 10 queries × 10 documents, backslash-escaped, `query<TAB>document` |
| `rerank/ids.txt` | llama.cpp's `format_prompt_rerank` ids for each pair (a list per line) |
| `rerank/q8_0_llama.f32` | llama.cpp's score per pair on the Q8_0 file, little-endian f32 |
| `rerank/f16_llama.f32` | the same pairs on the F16 conversion (the full-precision reference) |

The ten queries are lines 0–45 tokens long, spread through `texts.txt`; the ten
documents are at increasing lengths (18 to 400 tokens) so 51 of the 100 pairs are past
the 128-token local window and the sliding-window layers do banded work. Pair lengths run
36 to 445 tokens (`query + document + 3`: `[CLS] q [SEP] d [SEP]`).

## Provenance

| File | Repo | SHA-256 |
|---|---|---|
| `gte-reranker-modernbert-base-Q8_0.gguf` | `keisuke-miyako/gte-reranker-modernbert-base-gguf-q8_0` | `4fb51743840cd40e0cc32587241b3c61d9a90c698ff045729cf7c979a7a10d70` |
| `gte-reranker-modernbert-base-F16.gguf` | `keisuke-miyako/gte-reranker-modernbert-base-gguf-f16` | `0d4b9cd1354abb85dceb9b616cde88c2cb525ce67b795fa5523e6f42fe91f624` |

Both files declare `general.architecture = modern-bert`, 22 blocks of 768, the window 128
with pattern 3, `classifier.output_labels = ["LABEL_0"]`,
`tokenizer.ggml.add_eos_token = true` and `tokenizer.ggml.add_sep_token = false`. Neither
declares `classifier.pooling_type` or `classifier.activation`, so the head pools the mean
and activates with the erf GELU — this family's defaults. The id set is the ModernBERT
50 368-token GPT-2 vocabulary, with `[CLS]` (50281) as BOS and `[SEP]` (50282) as both EOS
and SEP.

The ids come from `scripts/tokenizer-pair-oracle.cpp`, which calls llama.cpp's
`format_prompt_rerank` rather than a re-implementation:

```sh
g++ -O2 -std=c++17 scripts/tokenizer-pair-oracle.cpp -o /tmp/pairoracle \
    -I /var/tmp/lc-s43/tools/server -I /var/tmp/lc-s43/tools/mtmd \
    -I /var/tmp/lc-s43/common -I /var/tmp/lc-s43/include -I /var/tmp/lc-s43/ggml/include \
    -L /var/tmp/lc-s43/build/bin -lllama-server-impl -lllama-common -lllama -lggml \
    -lggml-base -lmtmd -Wl,-rpath,/var/tmp/lc-s43/build/bin
/tmp/pairoracle gte-reranker-modernbert-base-Q8_0.gguf texts.tsv > ids.txt
```

The scores come from `llama-server --reranking -m <file> -c 8192 -b 8192 -ub 8192 -t 4`,
one request per query with its ten documents (`POST /v1/rerank`), the results put back in
document order:

```sh
curl -s http://127.0.0.1:8199/v1/rerank -H 'content-type: application/json' \
  -d '{"query": "<query>", "documents": ["<document>", ...]}'
```

`llama.cpp`'s `usage.prompt_tokens` for each request equals the fixture's own pair lengths
(`q + d + 3` for every pair), which is the check that the pair is `[CLS] q [SEP] d [SEP]`
with no extra separator: this file sets `add_sep_token = false`.

`texts.txt` is English only and its lines are synthetic sentences, so the documents are a
length ladder rather than a relevance ladder: what the test compares is the ids, the
scores and the order, not what the scores mean.
