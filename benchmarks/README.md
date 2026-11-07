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
- **Fraction-of-Ceiling**: Achieved decode throughput is judged against the physical streaming bandwidth of the machine:
  $$\text{Fraction of Ceiling} = \frac{\text{Bytes moved per token} \times \text{Decode tok/s}}{\text{Machine streaming bandwidth ceiling}}$$
  Where bytes moved per token is measured directly via Peregrine's telemetry (`TokenTelem`), and the streaming bandwidth ceiling is measured via the hardware probe (`Machine::probe`).

---

## 3. Pinned Competitor Engines

As specified in `docs/targets.md`:

| Engine | Pinned Version / Commit | Notes |
|---|---|---|
| **Peregrine** | Repository HEAD / tagged release | Built with `cargo build --release -p peregrine` |
| **llama.cpp** | `v0.9.0` (commit `cd959ba`) | Standard CPU release; CMake Release build with AVX2/FMA |
| **ik_llama.cpp** | `master` (`ikawrakow/ik_llama.cpp`) | Pinned fork optimized for x86 CPU matrix kernels |
| **bitnet.cpp** | Pinned release | Pinned for ternary model baselines (Claim 1) |
| **PrismML llama.cpp** | Pinned release | Pinned fork for ternary PQ2_0 baselines |

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
│   └── ik_llama_cpp.py   # ik_llama.cpp adapter
├── results/              # Directory where raw JSON run sets are stored
└── samples/              # Committed sample run JSONs and reports
```

---

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

### Dry Run (Inspect Build & Execution Commands)

```sh
python3 benchmarks/run.py \
    --models /path/to/model.gguf \
    --engines peregrine,llama_cpp \
    --dry-run
```

### Generate Markdown Comparison Report

```sh
python3 benchmarks/report.py benchmarks/results/*.json --output report.md
cat report.md
```
