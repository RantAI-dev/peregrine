# Peregrine Benchmark Harness

The automated, reproducible benchmark harness for Peregrine and competing CPU inference engines (roadmap item **S3.1**).

This harness measures the **four headline numbers** across reference and competing engines under identical hardware and software conditions, writes self-contained raw JSON manifests, and generates Markdown comparison tables conforming to `docs/targets.md`.

---

## 1. What Is Measured: The Four Numbers

Per `docs/targets.md` (Section 4), every benchmark report measures four distinct numbers:

1. **Single-user decode throughput** (`decode tok/s`, 1 user):
   Autoregressive generation speed with greedy sampling ($T=0$). Bounded by the sustained memory read bandwidth of the host.
2. **Prompt processing / prefill throughput** (`prompt tok/s`):
   Sustained throughput when prefilling the input context before emitting the first token. Compute-bound, scaling with the CPU's vector multiply-accumulate peak.
3. **Multi-user aggregate throughput** (`tok/s` at 4 and 10 concurrent users):
   Total tokens generated per second across all concurrent active slots when batching requests into shared weight passes.
   The report shows tokens over the whole wall time, prefill included. The JSON also records `decode_aggregate_tok_s`:
   tokens after each user's first, over the window from the last first token to the end, when every user is decoding.
   Requests generate 256 tokens by default (`--tokens`), so decode rather than prefill dominates the totals.
4. **Time to first token** (`TTFT` at 10 concurrent users):
   Latency from sending the request until the first token chunk arrives at the client under full concurrency load (10 users).

### Context Lengths

Each number is measured at three standard context lengths defined in `docs/targets.md`:
- **512** tokens (short context / chat turn)
- **4k** (4,096 tokens) (medium context / document)
- **16k** (16,384 tokens) (long context; 32k where model support permits)

---

## 2. Methodology & Discipline

- **Same model file, same host**: Every engine runs against the exact same GGUF weights on the same machine.
- **Interleaved execution (ABAB...)**: To control for thermal throttling, power profile fluctuations, and background OS noise, runs are interleaved (e.g. Engine A, Engine B, Engine A, Engine B) across at least 3 repetitions.
- **Medians and Spread**: The median of each measurement is reported alongside its spread ($\frac{\max - \min}{2}$ or min/max range). Outliers and transient spikes are never averaged away.
- **Prefix cache defeat**: To measure true prefill throughput and avoid artificial prefix-cache hits across repeat runs, every request prepends a unique first-line nonce (`// Benchmark run nonce: <uuid>\n`) while keeping the subsequent token sequence identical across engines.
- **Prompt length calibration**: Prompts are calibrated per model and engine before measurement. The runner sends probe requests, reads `usage.prompt_tokens`, scales the corpus character count, and iteratively adjusts until the actual prompt token count is within ~2% of the target context length (including the nonce line). Every measured request records its actual prompt tokens in the JSON manifest; `report.py` displays it and fails loudly if any request deviates by >5% from the target.
- **Fixed output length (`ignore_eos: true`)**: Without disabling end-of-sequence termination, generation can stop early, skewing decode token counts across engines. Both Peregrine (`crates/peregrine-server`) and llama.cpp accept `"ignore_eos": true` to decode to `max_tokens`. The harness asserts `completion_tokens == max_tokens` per request (warning and dropping samples if mismatched).
- **Slots and context per phase**:
  - In llama.cpp, `-c` specifies the total context divided among `-np` slots (unless `--kv-unified` is supported).
  - In Peregrine, `-c` is per slot.
  - To prevent slot starvation at 16k context and eliminate massive over-allocation of KV cache, servers are started per `(context, user count)` phase with `slots` set to the exact number of concurrent users (1, 4, 10).
  - For llama.cpp, the harness passes `-c ctx*slots -np slots` (or `-c ctx -np slots --kv-unified` if `--kv-unified` is present in `llama-server --help`).
  - For Peregrine, the harness passes `-c ctx --slots slots`.
  - Servers are cleanly stopped and restarted between phases, maintaining ABAB interleaving per round and executing a warm-up request after every start.
- **Fraction-of-Ceiling**: Achieved decode throughput is judged against the physical streaming bandwidth of the machine:
  $$\text{Fraction of Ceiling} = \frac{\text{Bytes moved per token} \times \text{Decode tok/s}}{\text{Machine streaming bandwidth ceiling}}$$
  Where bytes moved per token is measured directly via Peregrine's telemetry (`TokenTelem`), and the streaming bandwidth ceiling is measured via the hardware probe (`Machine::probe`).

### Flash attention (llama.cpp and its forks): compare against the faster setting

llama.cpp v0.9.0 defaults to `-fa auto`, which resolves to flash attention **on** on the CPU
backend. On some hosts that path is slower than `-fa off`, and at long contexts it can be much
slower: on the 4-core bluefin (Zen 2) host it decodes a 4096-token Llama 3.2 1B prompt at 13.0
tok/s by default and 29.1 tok/s with `-fa off`, while short-context decode is about 10% faster
with `-fa off` too.

The policy for every published ratio is therefore:

- **Compare Peregrine against llama.cpp's faster flash-attention setting for each row** (`on` or
  `off`, whichever measures faster with the same harness method and harness settings).
- **Never count a win against a misconfigured competitor.** A ratio produced against the slower
  setting is not a result.
- **Show the default-setting number only as a secondary note**, clearly labelled as
  "llama.cpp defaults (`-fa auto`)", alongside the like-for-like ratio.

`run.py` takes `--llama-fa {auto,on,off}` (default `auto`) for every llama.cpp-family
adapter — upstream `llama_cpp` and the PrismML fork — so one option sets the whole comparison.
The value is passed as `-fa` to both `llama-server` and `llama-bench` and recorded in every JSON
manifest under `settings.llama_flash_attn` and `engine.flash_attn`, so a run can never be
misread. The default is
`auto` so that older invocations reproduce their original configuration; benchmarks intended for
publication should measure both settings on the same day (or reuse a same-day default run) and use
the faster of the two in each ratio.

---

## 3. Pinned Competitor Engines

As specified in `docs/targets.md`:

| Engine | Pinned Version / Commit | Notes |
|---|---|---|
| **Peregrine** | Repository HEAD / tagged release | Built with `cargo build --release -p peregrine` |
| **llama.cpp** | `v0.9.0` (commit `cd959ba`) | Standard CPU release; CMake Release build with AVX2/FMA |
| **ik_llama.cpp** | `master` (`ikawrakow/ik_llama.cpp`) | Pinned fork optimized for x86 CPU matrix kernels |
| **bitnet.cpp** | Pinned release | Pinned for ternary model baselines (Claim 1) |
| **PrismML llama.cpp** | `prism` branch, commit `62c418c` | Pinned fork for ternary PQ2_0 baselines; CMake Release build with native CPU dispatch |

---

## 4. Machine Record

Every JSON run set captures complete hardware and operating system state:
- **CPU Model & Logical Threads**: e.g., AMD Ryzen 3 PRO 4350G, 8 threads.
- **CPU Features**: `avx`, `avx2`, `fma`, `avx512f`, `avx512vnni`, `neon`, etc.
- **Measured Bandwidth Ceiling**: Sustained streaming read bandwidth (GB/s) probed by `Machine::probe`.
- **Measured Compute Ceiling**: Peak FMA GFLOP/s and roofline ridge point.
- **Operating System & Kernel**: Full kernel release (`uname -srm`).
- **CPU Governor**: Frequency scaling governor (e.g. `performance`).

---

## 5. Directory Layout

```
benchmarks/
├── README.md             # This document
├── run.py                # Main benchmark runner (Python 3 stdlib only)
├── report.py             # Markdown report generator (Python 3 stdlib only)
├── prompts/
│   └── corpus.txt        # Deterministic text corpus for prompt generation
├── engines/
│   ├── __init__.py
│   ├── base.py           # EngineAdapter base class & HTTP streaming client
│   ├── peregrine.py      # Peregrine engine adapter (CLI bench & server)
│   ├── llama_cpp.py      # llama.cpp adapter (clone, CMake build, server)
│   ├── ik_llama_cpp.py   # ik_llama.cpp adapter
│   └── prism_llama_cpp.py # PrismML's llama.cpp fork adapter (PQ2_0/PTQ1_0)
├── results/              # Directory where raw JSON run sets are stored
├── accuracy/             # Task-accuracy gate (S3.3): GSM8K, MMLU, HumanEval+,
│                         #   BF16 vs llama.cpp vs Peregrine, and decoding invariance
└── samples/              # Committed sample run JSONs and reports
```

---

The task-accuracy gate lives beside this harness in `accuracy/` (its own README): it runs
GSM8K, MMLU and HumanEval+ against any OpenAI-compatible server, compares Peregrine with a
BF16 reference and with llama.cpp on the same GGUF, and checks that decoding is byte-identical
alone, batched and with speculation.

## 6. How to Run

### Quick Smoke Test (Synthetic Model, Peregrine Only)

```sh
python3 benchmarks/run.py --smoke --output-dir benchmarks/results
```

### Full Benchmark (Interleaved Peregrine vs llama.cpp)

```sh
python3 benchmarks/run.py \
    --models /path/to/model.gguf \
    --engines peregrine,llama_cpp \
    --contexts 512,4096,16384 \
    --runs 3 \
    --threads 8 \
    --output-dir benchmarks/results
```

### llama.cpp with flash attention off (the published comparison setting)

```sh
LLAMA_SERVER_BIN=/path/to/llama-server \
python3 benchmarks/run.py \
    --models /path/to/model.gguf \
    --engines llama_cpp \
    --contexts 512,4096 \
    --runs 3 \
    --threads 4 \
    --llama-fa off \
    --output-dir benchmarks/results
```

`--llama-fa auto` (the default) reproduces llama.cpp's built-in default; `on` forces flash
attention on. The choice is written to the run manifest, so published numbers always say which
setting produced them.

### Dry Run (Inspect Build & Execution Commands)

```sh
python3 benchmarks/run.py \
    --models /path/to/model.gguf \
    --engines peregrine,llama_cpp \
    --dry-run
```

### Split a Long Sweep, or Change One Engine's Options

`--user-counts 1,10` drops the 4-user phase, which lets a slow host keep each invocation
under a wall-clock limit; run the rounds as separate invocations and combine their raw
sample arrays (see `benchmarks/results/2026-11-08-bonsai2-prism/merge.py`). `--engine-args`
adds arguments to one engine only, and `--chat-template-kwargs` sends the same
`chat_template_kwargs` to every engine's chat requests:

```sh
python3 benchmarks/run.py \
    --models /path/to/bonsai.gguf \
    --engines peregrine,prism_llama_cpp \
    --contexts 512 --user-counts 1,10 --runs 3 --threads 12 \
    --engine-args 'peregrine=--opt MTP=0' \
    --engine-args 'prism_llama_cpp=--spec-type draft-mtp' \
    --llama-fa off \
    --chat-template-kwargs '{"enable_thinking": false}' \
    --token-tolerance 1 \
    --output-dir benchmarks/results
```

`--token-tolerance N` keeps a sample whose reply is up to `N` tokens short of `--tokens`,
for servers that do not count an empty control token at the end of a reasoning-template
reply. `--llama-fa {auto,on,off}` sets the same `-fa` value for the llama.cpp-family engines
(upstream and the fork) on both `llama-server` and `llama-bench`; it is recorded as
`engine.flash_attn` in every manifest and as `settings.llama_flash_attn` in the run set, so a
reader can tell which setting produced the numbers.

### Generate Markdown Comparison Report

```sh
python3 benchmarks/report.py benchmarks/results/*.json --output report.md
cat report.md
```

---

## 6. Embedding Mode (S4.2)

`embed.py` measures the embedding path: texts per second, tokens per second and
request latency for `POST /v1/embeddings`, against the same pinned competitors.

```sh
# deterministic, per-model tokenizer-calibrated texts (needs a venv with transformers)
python3 benchmarks/embed_texts.py --tokenizer-dir ~/models/embed/minilm/ref \
    --lengths 16,128,512 --per-length 128 --max-tokens 510 \
    --out /var/tmp/embed-texts/minilm.json

# fp32 reference vectors for the accuracy column
python3 benchmarks/embed_ref.py --model-dir ~/models/embed/minilm/ref \
    --texts /var/tmp/embed-texts/minilm-acc.json --out /var/tmp/embed-texts/ref/minilm-torch-fp32.json

# one round over one row (grid: 16/128/512 tokens x batches 1/8/32/128 x 1 and 10 clients)
python3 benchmarks/embed.py --select minilm:llama_cpp:f16 --round 1 \
    --output-dir benchmarks/results/<run> --texts-dir /var/tmp/embed-texts

# accuracy for that row
python3 benchmarks/embed.py --mode accuracy --select minilm:llama_cpp:f16 \
    --output-dir benchmarks/results/<run> --texts-dir /var/tmp/embed-texts

# merge rounds into medians/spreads and Markdown tables
python3 benchmarks/embed_report.py --results-dir benchmarks/results/<run>
```

Rows live in `embed_models.json` (engine, precision, path under `$EMBED_MODEL_DIR`,
pooling, `max_batch_tokens`). Engines are adapters in `benchmarks/engines/`:
`llama_embed.py`, `tei_embed.py`, `fastembed_embed.py`, `openvino_embed.py` and
`peregrine_embed.py`, the last of which posts the OpenAI request shape and only needs
`PEREGRINE_BIN`/`--peregrine-worktree` once `/v1/embeddings` is on `main`.

Rules the runner enforces, because they change the numbers:

- **One request per text on llama.cpp.** `llama-server` feeds embedding inputs one task at
  a time even inside one request, and caps `n_ctx_slot` at the training context, so `-c` is
  `(max_tokens + 8) x slots` and `-b`/`-ub` must stay inside it (a larger batch aborts in
  `build_pooling`). Some GGUFs abort with `-np > 1`; the adapter retries with one slot and
  records `engine.slots_fallback`.
- **A per-model batch-token cap** (`max_batch_tokens`) skips cells whose padded attention
  buffers would not fit the host's memory. Skips are recorded with their reason.
- **A per-cell time cap** (`--max-cell-seconds`, default 120 s) skips cells that a single
  request would blow past; the decision is made in round 1 and reused by later rounds.
- **A quiescence preflight** (`load average < 0.5`, `MemAvailable >= 5 GB`) before every
  timed invocation; both values are recorded before and after each row.
