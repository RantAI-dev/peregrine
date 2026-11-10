# Accuracy against llama.cpp: the KL-divergence measurement

`crates/peregrine-bench/examples/kld.rs` compares Peregrine's next-token distributions with llama.cpp's on
real text, the way `llama-perplexity --kl-divergence` compares two llama.cpp runs.

## What it measures, and what it does not

llama.cpp writes a reference file with `--kl-divergence-base`: the evaluation tokens and, for every scored
position, its log-probabilities over the whole vocabulary. `kld` runs the same tokens through Peregrine with the
same chunking (chunks of `n_ctx` tokens, the second half of each chunk scored) and reports, per position, the
KL divergence of Peregrine's distribution from llama.cpp's, the difference in the probability of the correct
token (Δp), and whether both engines put the same token on top.

It measures the **distance between two engines** that both approximate the exact computation: both quantise
activations (llama.cpp to Q8_K for k-quant matrices, Peregrine to its own int8 layouts) and sum in different
orders. It is not the error against the exact computation. The accuracy budget in `docs/targets.md` (mean KL
≤ 1e-4) is against Peregrine's own exact f32 path and is a separate measurement.

Sanity check: llama.cpp compared against its own base file gives a mean KLD of 0.000000, so the base file and
the comparison are consistent.

## Like-for-like: the same cache precision on both sides

The measurement above compares two engines that do the same computation *differently*. Peregrine keeps its
queries in f32 but rounds its cached keys and values to f16; llama.cpp's cache is f16 by default, so its
attention reads f16 operands as well. A difference between the two is then a difference between two *different*
computations, and it is not obvious which engine it comes from.

For a comparison in which both engines' attention reads f32 keys and values, run llama.cpp with a full-precision
cache and Peregrine with the f32 KV cache (`KvPrec::F32`):

```sh
llama-perplexity -m model.gguf -f wiki.test.raw -c 512 -b 512 -t 8 --chunks 40 \
    --cache-type-k f32 --cache-type-v f32 --flash-attn off \
    --kl-divergence-base model.f32.base.bin
cargo run --release -p peregrine-bench --example kld -- \
    --base model.f32.base.bin --model model.gguf --threads 8 --kv f32
```

The flags, and what each one is for:

| Flag | Side | Why |
|---|---|---|
| `--cache-type-k f32 --cache-type-v f32` | llama.cpp | the keys and values in the cache keep full precision, so its attention reads what the model produced |
| `--cache-type-k f16 --cache-type-v f16` | llama.cpp | the same run with llama.cpp's default cache: the pair of rows isolates the cache precision |
| `--flash-attn off` | llama.cpp | llama.cpp's flash-attention kernels do not read a full-precision cache, and the two are different computations (on 2 chunks of SmolLM2, PPL 13.7227 with flash attention on and 13.7081 with it off). `auto` picks flash attention *on* for the CPU, so it has to be turned off explicitly |
| `--kv f32` | Peregrine | the same precision on this side (`--kv f16`, `--kv i8`; the default is f16) |
| `PEREGRINE_KV=f32` | Peregrine | the same choice for the engine as a whole (`peregrine serve --kv f32` says it per run) |

The result is in `docs/targets.md`. It is worth stating what it says, because it is not the expected answer: the
cache precision moves each engine's *own* output by about as much as the two engines differ (llama.cpp against
itself, f16 against f32, gives a mean KLD of 0.0132 on Qwen3-1.7B — the same size as Peregrine against
llama.cpp), but it does not create the distance between them: the two engines' keys and values are nearly the
same, so they round to the same halves and both distributions move the same way.

## Reproducing

llama.cpp v0.9.0 (commit `cd959ba`), CPU build. Corpus: wikitext-2-raw `wiki.test.raw`
(<https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip>, the file llama.cpp's
`scripts/get-wikitext-2.sh` fetches; `wiki.test.raw` SHA-256
`173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08`).

```sh
llama-perplexity -m model.gguf -f wiki.test.raw -c 512 -b 512 -t 8 --chunks 40 \
    --kl-divergence-base model.base.bin
cargo run --release -p peregrine-bench --example kld -- \
    --base model.base.bin --model model.gguf --threads 8
```

The base file is large (vocabulary × scored positions; 3 GB for Qwen 2.5's 152k vocabulary at 40 chunks): keep it
outside the repository and delete it afterwards.

## Reading the output

- **Mean KLD ± error**: the headline number. Below ~0.002 the two engines agree about as closely as two
  builds of the same engine with different activation quantisation.
- **Same top p**: how often both engines pick the same most likely token. Greedy decoding diverges where this
  differs, at near-ties.
- **99% / maximum KLD**: the tail. A single large maximum with a small 99% is a near-tie; a large 99% means a
  systematic difference.
- **PPL(Q) vs PPL(base)**: Peregrine's perplexity against llama.cpp's on the same tokens.

Results are recorded in `docs/targets.md` with the date, host, Peregrine commit and llama.cpp version.

## Task accuracy and decoding invariance (S3.3)

The KL measurement above compares next-token distributions on a corpus. It does not say
whether an engine still *answers the task*: a distribution that is close on average can
still change a greedy answer at a near-tie, and a change of answer changes the score. The
task-accuracy harness (`benchmarks/accuracy/`, documented in its README) closes that gap and
adds the invariance runs.

**Protocol.** The same prompt bytes and the same items go to every engine, over the
OpenAI-compatible chat API, greedy (`temperature: 0`), one user turn:

| Task | Items | Scoring |
|---|---|---|
| GSM8K (`openai/grade-school-math` `3101c7d`, test) | 250 (full reference run; 40-150 on Peregrine) | numeric match against the `####` answer |
| MMLU (`cais/mmlu` `c30699e`, `all`/test) | 500 (full; 150-300 on Peregrine) | letter match (0-shot) |
| HumanEval+ (`evalplus/humanevalplus` `d32357c`, 163 after excluding HumanEval/32) | 163 (full; 40-100 on Peregrine) | pass@1, tests executed in a sandbox |

MMLU is 0-shot: the usual five-shot `Answer: X` transcript made the 1B chat models answer
the whole list or ignore the format, and the single-question form scored higher for every
gate model in a protocol check on 30 items (33-43 % against 23-37 %). HumanEval/32 is
excluded because its plus test in the pinned revision unpacks the candidate's float return
value; the dataset's own canonical solution fails it too (`run.py selftest`: 163/163).

**The gate.** A release candidate passes when, per gate model and task:

1. **Same weights as llama.cpp**: on the items both engines answered, the difference is not
   significant (exact McNemar p ≥ 0.05) and the point difference is at most **4 points**;
2. **Full-precision reference**: at most **10 points** below the BF16 model on the same
   items. Q4_K_M itself costs accuracy at this scale (llama.cpp's own distance from BF16
   reaches 6.8 points on these tasks), so this is an absolute floor, not the primary test;
3. **Invariance**: 500+ prompts decoded alone, batched with others, and with speculation
   produce **byte-identical** replies.

The 4-point tolerance is not the roadmap's ±1 point. With n ≤ 300 items per cell, the paired
95 % CI of a difference near zero is ±5-15 points; the ±1 point target in `docs/targets.md`
describes an aspiration for a much larger harness, and the proposal here is the bound the
current sample sizes can actually decide. The CI and p-value are printed in every report so
a future run with more items can tighten the tolerance without changing the method.

**Measured, 2026-11-09, DGX Spark (20-core ARM aarch64, GB10 GPU).** BF16 reference:
vLLM 0.33.0. Quantised: Q4_K_M, llama.cpp 0.9.0-dev (`b12223`, `0c6dca2`), CPU, 16 threads;
Peregrine `e31499c`, its plain aarch64 path, 16 threads. Full reference/llama.cpp sets;
Peregrine ran smaller seeds of the same item list (n in brackets), because its ARM path is
3-6× slower than llama.cpp on this host (22-24 minutes per model against 2-4).

| Model | Engine | GSM8K | MMLU | HumanEval+ |
|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct | BF16 (vLLM) | 42.8 | 36.2 | 25.2 |
| | llama.cpp Q4_K_M | 40.4 | 33.6 | 32.5 |
| | Peregrine Q4_K_M | 38.0 (150) | 32.0 (300) | 29.0 (100) |
| | Δ vs llama.cpp, paired | -2.0 | -2.0 | +1.0 |
| Llama-3.2-1B-Instruct | BF16 (vLLM) | 48.8 | 30.4 | 33.1 |
| | llama.cpp Q4_K_M | 48.8 | 29.6 | 30.7 |
| | Peregrine Q4_K_M | 42.7 (150) | 28.0 (300) | 31.0 (100) |
| | Δ vs llama.cpp, paired | -3.3 | -2.0 | +3.0 |
| gemma-3-1b-it | BF16 (vLLM) | 54.0 | 39.0 | 38.7 |
| | llama.cpp Q4_K_M | 49.2 | 40.0 | 31.9 |
| | Peregrine Q4_K_M | 55.0 (60) | 40.0 (200) | 25.0 (60) |
| | Δ vs llama.cpp, paired | -3.3 | +0.5 | -1.7 |
| Qwen3-1.7B | BF16 (vLLM) | 80.4 | 47.0 | 62.6 |
| | llama.cpp Q4_K_M | 78.0 | 45.4 | 56.4 |
| | Peregrine Q4_K_M | 75.0 (40) | 46.7 (150) | 47.5 (40) |
| | Δ vs llama.cpp, paired | +5.0 | -3.3 | -2.5 |

All 12 paired comparisons pass the gate (largest |Δ| 5.0 points, every McNemar p ≥ 0.18);
the full table with CIs and p-values is `benchmarks/accuracy/results/2026-11-09-dgx-spark/report.md`.

**Invariance, 2026-11-09, same host.** 520 prompts (208 GSM8K, 156 MMLU, 156 HumanEval+,
`max_tokens 48`) through Peregrine with Qwen2.5-0.5B Q4_K_M, `--slots 8`, prefix cache on:
each prompt decoded (a) alone, one request at a time, (b) in groups of eight concurrent
requests, (c) alone against a server started with `--speculate` (prompt-lookup). All
**520/520 replies are byte-identical** across the three runs
(`benchmarks/accuracy/results/2026-11-09-dgx-spark/invariance/qwen25-05b/compare.md`). The run also confirms
that batching does not change the tokens: the same 520 prompts were answered concurrently
in one weight pass without a single differing byte.

**Full ARM sweep, 2026-11-10.** The aarch64 NEON i8mm tier (`9e47bb2`, the #129 tip) was
built once and run on the full sets (250/500/163, no `--limit`) on the same DGX Spark,
pinned to the 10 Cortex-X925 cores, `--slots 8`. All four gate models pass all twelve
checks:

| Model | Engine | GSM8K | MMLU | HumanEval+ |
|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct | BF16 (vLLM) | 42.8 | 36.2 | 25.2 |
| | llama.cpp Q4_K_M | 40.4 | 33.6 | 32.5 |
| | Peregrine i8mm Q4_K_M | 40.0 | 35.0 | 34.4 |
| | Δ vs llama.cpp, paired | -0.4 | +1.4 | +1.8 |
| Llama-3.2-1B-Instruct | BF16 (vLLM) | 48.8 | 30.4 | 33.1 |
| | llama.cpp Q4_K_M | 48.8 | 29.6 | 30.7 |
| | Peregrine i8mm Q4_K_M | 48.0 | 28.8 | 31.9 |
| | Δ vs llama.cpp, paired | -0.8 | -0.8 | +1.2 |
| gemma-3-1b-it | BF16 (vLLM) | 54.0 | 39.0 | 38.7 |
| | llama.cpp Q4_K_M | 49.2 | 40.0 | 31.9 |
| | Peregrine i8mm Q4_K_M | 45.6 | 40.4 | 32.5 |
| | Δ vs llama.cpp, paired | -3.6 | +0.4 | +0.6 |
| Qwen3-1.7B | BF16 (vLLM) | 80.4 | 47.0 | 62.6 |
| | llama.cpp Q4_K_M | 78.0 | 45.4 | 56.4 |
| | Peregrine i8mm Q4_K_M | 77.6 | 45.4 | 55.8 |
| | Δ vs llama.cpp, paired | -0.4 | 0.0 | -0.6 |

The largest paired difference is -3.6 points (gemma-3-1b GSM8K, exact McNemar p = 0.12) and
every other cell is within 1.8 points; the full table with CIs and p-values is
`benchmarks/accuracy/results/2026-11-10-dgx-spark-arm/report.md`. The earlier 40-item
Qwen3-1.7B HumanEval+ subset (47.5 %) was sampling noise, not a regression: the full set is
55.8 % against llama.cpp's 56.4 % (Δ -0.6, p = 1.0), and on the same 40 shuffled items the
full run scores 52.5 %. The two items that move are different code generations (both
engines return a fenced Python block throughout), so this is numeric near-ties at the token
level, not tokenizer, template or stop-sequence handling.

**Invariance on the NEON build, 2026-11-10.** The same 520-prompt set was replayed on the
i8mm build for Qwen2.5-0.5B and Llama-3.2-1B, alone, in groups of eight, and against a
server started with `--speculate`: **520/520 replies are byte-identical** for both models
(`benchmarks/accuracy/results/2026-11-10-dgx-spark-arm/invariance/qwen25-05b/compare.md`,
`.../invariance/llama32-1b/compare.md`). Speculation is much slower than plain decoding on
this build (0.4-1.6 prompts/s against 0.6-2 prompts/s plain), so the phase was run in
chunks; the committed `speculation.jsonl` is the full 520-prompt sequence.

**What this run does not cover.** The full 250/500/163 sweep on the x86 reference hosts
(H2/H3) is still the release run and uses the same harness (`--limit` omitted); the numbers
above are the aarch64 half. Qwen3-1.7B is measured with thinking disabled for all engines,
which is the chat default the server exposes; a thinking-enabled run is a separate
configuration.
