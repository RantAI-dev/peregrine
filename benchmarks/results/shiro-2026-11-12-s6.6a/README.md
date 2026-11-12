# S6.6a: constrained decoding — measurements

Host: `shiro`, Intel Xeon E5-2680 v4, 8 vCPUs, AVX2, Linux x86-64. A **shared** VM: the
load average during these runs was 2.5–6.4 on 8 vCPUs (recorded in `host.txt` and in the
rows below). Decode rows are the median of five, interleaved so a slow spell on the host
does not land on one configuration. The reference engine is llama.cpp v0.9.0
(`/var/tmp/lc-s43/build/bin/llama-server`, commit `cd959bae9`), 4 threads.

```
uname -a: Linux shiro 6.8.0-147-generic ... x86_64
CPU:      Intel(R) Xeon(R) CPU E5-2680 v4 @ 2.40GHz, 8 vCPUs
models:   ~/models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf
          /var/tmp/s66a-models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
```

## (a) decode throughput, mask time, vocabulary preparation

`constrain_bench` (`crates/peregrine-bench/examples/constrain_bench.rs`) runs a fixed prompt
at temperature 0 through one `Engine`. One warm-up request per configuration builds the
vocabulary's tries and the masks; it also records the first request's "cold" mask samples.
The measured rounds then interleave the four configurations. Raw output:
`qwen-decode-mask.txt`, `llama-decode-mask.txt`.

### Qwen 2.5 0.5B Instruct Q4_K_M (load 2.5–4.0)

| configuration | decode tok/s (median of 5) | vs unconstrained |
|---|---:|---:|
| unconstrained | 40.69 | 1.000 |
| `json_object` | 41.86 | 1.029 |
| medium schema (10 properties, nested, enum, array) | 41.13 | 1.011 |
| large schema (60 properties + recursive `$ref`) | 40.30 | **0.990** |

| mask | median | p99 | worst |
|---|---:|---:|---:|
| `json_object`, per token (175 calls) | 149.3 µs | 0.55 ms | 0.57 ms |
| medium, per token (760 calls) | 169.4 µs | 3.03 ms | 3.45 ms |
| large, per token (2000 calls) | 154.0 µs | 2.90 ms | 3.77 ms |
| **cold first request** (`json_object`) | 79.6 µs | 0.54 ms | 7.43 ms |
| **cold first request** (medium) | 75.7 µs | 0.48 ms | 3.02 ms |
| **cold first request** (large) | 89.5 µs | 2.88 ms | 3.01 ms |

Warm mask median is 0.4–0.7 % of a 4-thread decode step (24 ms). The p99/worst tail is the
first request that reaches a new bounded-string length or a new parent state; after a few
requests the mask memo holds those states. The **cold first request** column is the one that
pays the trie pages and the first parent-state computations; its median is under 90 µs and
its worst 3.0–7.4 ms on this host, most of it first-touch page faults on a 22 MB trie.

Vocabulary preparation: **prep median 221 ms**, min 205, max 276 across five fresh builds
(151 936 tokens: token bytes, classification, three byte tries), one-time on the first
constrained request. On a quiet run earlier the same build was 148 ms.

### Llama 3.2 1B Instruct Q4_K_M (load 2.5–4.0)

| configuration | decode tok/s (median of 5) | vs unconstrained |
|---|---:|---:|
| unconstrained | 22.47 | 1.000 |
| `json_object` | 22.42 | **0.998** |
| medium schema | 22.84 | 1.016 |
| large schema | 22.18 | **0.987** |

| mask | median | p99 | worst |
|---|---:|---:|---:|
| `json_object`, per token (185 calls) | 143.4 µs | 1.34 ms | 1.39 ms |
| medium, per token (635 calls) | 146.0 µs | 5.55 ms | 6.03 ms |
| large, per token (2000 calls) | 145.0 µs | 5.47 ms | 6.06 ms |
| **cold first request** (`json_object`) | 49.0 µs | 2.33 ms | 3.86 ms |
| **cold first request** (medium) | 48.3 µs | 1.02 ms | 5.49 ms |
| **cold first request** (large) | 52.1 µs | 5.42 ms | 6.59 ms |

Warm mask median is 0.3 % of a 4-thread decode step (44 ms). Vocabulary preparation:
**prep median 187 ms**, min 178, max 189 across five fresh builds (128 256 tokens).

**Every constrained/unconstrained decode cell below 1.0 is named above**: Qwen large 0.990;
Llama `json_object` 0.998 and large 0.987. They are inside the ±10 % run-to-run spread of
this shared host and are not a mask cost: the mask is 0.14–0.17 ms against a 24–44 ms step.

The unconstrained path is unchanged: no golden moved, and its rows here are the same code
path as before (`cargo test --release -p peregrine-models --test synthetic`, native and
`PEREGRINE_ISA=scalar`).

## (b) validity through `/v1/chat/completions`

200 requests per model: 10 schemas × 20 seeds, ordinary user messages in plain words, the
server's own chat template, temperature 0.7, `max_tokens: 320`, `strict: true`. Every reply
that ended by itself was parsed and validated with Python's `jsonschema` 4.10.3 (independent
of the Rust matcher). Raw rows: `qwen-peregrine.json`, `llama-p-a.json` + `llama-p-b.json`;
summary: `validity-summary.json`. The `json_object` schema is `{"type":"object"}`.

| model | requests | ended by themselves | valid among them | cut by `max_tokens` | errors |
|---|---:|---:|---:|---:|---:|
| Qwen 2.5 0.5B | 200 | 200 | **200 / 200 (100 %)** | 0 | 0 |
| Llama 3.2 1B | 200 | 200 | **200 / 200 (100 %)** | 0 | 0 |

Per schema both models are 20/20 valid.

## (c) the same 200 requests on llama.cpp v0.9.0

Same schemas, messages, seeds, temperature, `max_tokens`, 4 threads. llama-server commit
`cd959bae9`. Raw rows: `qwen-llama.json`, `llama-l-a.json` + `llama-l-b.json`.

| model | engine | ended by themselves | valid among them | cut (`length`) | errors | tokens/s (median) |
|---|---|---:|---:|---:|---:|---:|
| Qwen 2.5 0.5B | Peregrine | 200 | 200/200 | 0 | 0 | 40.52 |
| Qwen 2.5 0.5B | llama.cpp | 198 | 198/198 | 2 | 0 | 31.52 |
| Llama 3.2 1B | Peregrine | 200 | 200/200 | 0 | 0 | 21.50 |
| Llama 3.2 1B | llama.cpp | 187 | 187/187 | 13 | 0 | 18.96 |

**Peregrine ÷ llama.cpp**, recomputed from the raw medians:

| model | validity (self-ended) | tokens/s | cell below 1.0? |
|---|---:|---:|---|
| Qwen 2.5 0.5B | 100 % ÷ 100 % = 1.000 | 40.52 ÷ 31.52 = **1.285** | no |
| Llama 3.2 1B | 100 % ÷ 100 % = 1.000 | 21.50 ÷ 18.96 = **1.134** | no |

No Peregrine ÷ llama.cpp cell is below 1.0. The truncation difference is real: llama.cpp's
`json_object` produced very long objects on 2/20 Qwen and 13/20 Llama requests and hit
`max_tokens`; Peregrine closed all of them. Those cut llama.cpp replies are not counted
against its validity share — the table follows the rule "100 % among replies that ended by
themselves".

## Commands

```sh
# decode, mask, preparation
cargo run --release -p peregrine-bench --example constrain_bench --features arch-qwen,arch-llama -- \
    ~/models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf 4

# validity, Peregrine (200 requests)
python3 /var/tmp/s66a-measure/measure.py peregrine ~/models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf \
    18422 qwen-peregrine.json --runs 20 --max-tokens 320

# validity, llama.cpp
python3 /var/tmp/s66a-measure/measure.py llama ~/models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf \
    18417 qwen-llama.json --runs 20 --max-tokens 320
```

The measurement script is not committed (it lives in `/var/tmp`); the raw rows and this
document are.
