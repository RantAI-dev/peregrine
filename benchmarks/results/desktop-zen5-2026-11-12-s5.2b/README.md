# S5.2b: where a prompt pass spends its time, and the attention value mix

Raw measurements for `docs/design/s5-prefill-avx512.md` §12, 2026-11-12 (09:30–14:00 local,
UTC+7) on the desktop: AMD Ryzen AI 9 HX 370 (Zen 5, 12 cores / 24 threads, AVX-512
F/BW/VL/VNNI/VBMI/BF16), WSL2, 11.7 GiB.

Peregrine: branch `s5.2b-prefill-attention` = `origin/main` `dc6beba` + `origin/s5.2-exactness`
`f3f7dd4`; `cargo build --release -p peregrine-bench --example promptbench` with
`CARGO_TARGET_DIR=~/.cache/peregrine-bench/target-s52b`.
ik_llama.cpp: `479074c` at `~/.cache/peregrine-bench/ik_llama.cpp-479074c` (the build of
§10.3, not rebuilt). Models: the same two files as every earlier desktop round, under
`~/.cache/peregrine-bench/models/` — Llama 3.2 1B Instruct Q4_K_M (807,694,464 B) and
Qwen 2.5 0.5B Instruct Q4_K_M (491,400,032 B). No other model file was opened.

**The host was shared throughout.** Another Peregrine worker ran `llama-perplexity`/
`llama-bench` and `cargo test` on the same 24 logical CPUs from 12:00 on, and the agent's own
UI held one core; the one-minute load average in every row below is 1–14 *including this
harness*. That is why every engine-level table is a median of interleaved rounds with its
spread printed, and why the 24-thread rows are marked unusable: they swing by more than 2×
between rounds on both engines.

## The clock, first

`clockprobe.c` (3 s burn, pinned to CPU 4), three times, at the start of the session:

```
MONOTONIC 3.000  RAW 2.728  CPU 2.724  TSC/Gticks 5.445  tsc_ghz_if_cpu_accurate 1.999  cpu_over_wall 0.908
MONOTONIC 3.000  RAW 2.727  CPU 2.720  TSC/Gticks 5.444  tsc_ghz_if_cpu_accurate 2.002  cpu_over_wall 0.907
MONOTONIC 3.000  RAW 2.728  CPU 2.726  TSC/Gticks 5.445  tsc_ghz_if_cpu_accurate 1.998  cpu_over_wall 0.908
```

So CLOCK_MONOTONIC ran **~10 % fast** again (3.000 s of it for 2.728 s of MONOTONIC_RAW and
2.72 s of CPU time): every tok/s below is ~10 % *conservative* in real seconds, and every
ratio between two engines timed with the same clock is untouched. The roof probes and
`peak.c` time with CPU time, so their GMAC/s need no correction.

The roof, measured this session (`peak.c`, one core, CPU 4, minimum of seven passes):

```
fma_zmm_8         3.69 G instr/s   59.0 GMAC/s   32.07 MAC/cyc   2.00 instr/cyc   16 MACs/instr
dp_zmm_16         1.83 G instr/s  117.1 GMAC/s   63.69 MAC/cyc
```

**The fp32 FMA roof is 59–61 GMAC/s per core** (2 × 512-bit FMA per cycle at
1.84–1.90 GHz); the byte-dot roof is 117–122 GMAC/s. Attention computes in f32, so 59 GMAC/s
is the roof the score and mix rates are quoted against.

## The harness

`crates/peregrine-bench/examples/promptbench.rs` (new): reads the **first N tokens of a text
file** through `Engine`, the way a request's prompt is read, and prints one JSON line per
cell — prefill tok/s and seconds from the engine's own accounting, decode tok/s, and the
`PEREGRINE_PROF2` slots. `peregrine bench --prompt-tokens N` reads synthetic ids; this reads
plain prose.

The corpus is the S5.2 recipe (this repository's own documentation, no chat template):

```sh
cat README.md ARCHITECTURE.md CHANGELOG.md CONTRIBUTING.md GOVERNANCE.md SECURITY.md \
    docs/README.md docs/adr/*.md docs/backlog.md docs/benchmark.md docs/compat.md \
    docs/docker.md docs/product.md docs/quickstart.md docs/service.md docs/targets.md \
    docs/design/moe.md docs/design/prefill.md docs/design/s2.6.md docs/design/s2.8-bonsai2.md \
    docs/design/s2.8-neon.md docs/design/s3.4.md docs/design/s3.5-delta-scan.md \
    docs/design/s4-embeddings-decisions.md docs/design/s4.9-sota.md \
    docs/design/s5-prefill-avx512.md docs/design/s5.1-kq-epilogue.md \
    docs/design/s5.1-q6k-zmm.md docs/design/s5.1-wide-pass.md docs/dev/*.md \
    docs/reference/bert.md docs/reference/bonsai2_spec.md \
    docs/reference/gemma-embedding.md docs/reference/mistral3.md > corpus.txt
```

(The `docs/design/s6-*` and `docs/reference/systemone.md` files are chat templates and are
excluded.)

Two drivers, both in this directory:

- `sweep.py <out.jsonl> <series> [models] [lens] [threads] [reps] [ub]` — one process per
  cell, rounds interleaved; series `pgrid` (Peregrine, `PEREGRINE_PROF2=1`), `pprof` (the
  same cell once with `=full`, for the attention split), `ikgrid` (both `-ub` values),
  `scaling`, `decode`, `users`.
- `ab.py <out.jsonl> <lens> <threads> [models] [decode] [slots]` — the before/after A/B:
  two `promptbench` builds that differ **only** in whether the prefill value mix takes the
  two-row body (`pb-before`, `pb-after`), alternating A, B within each round so a slow patch
  of the host lands on both arms. `REVERSE=1` flips the order, which is how the 512-token
  cells were checked for drift.

Everything runs in the foreground; no background job was left running. Scratch (binaries,
corpus, peak probes) is under `~/.cache/peregrine-bench/s52b/`.

---

## Part 1: where a prompt pass spends its time

### 1.1 Prompt tok/s, both engines, four thread counts (before the change)

Peregrine at its defaults (the wide prefill pass, 64 rows, since nobody is decoding);
ik_llama.cpp with `-rtr 1` at its default `-ub 512` and at `-ub 64`, median of its five
`llama-bench` samples (two for the slow 4096-token, one-thread cell). Medians of three
interleaved rounds; `P` is the S5.2b *before* binary.

| model | prompt | threads | Peregrine | ik `-ub 512` | ik `-ub 64` | P ÷ ik512 |
|---|---|---:|---:|---:|---:|---:|
| Llama 3.2 1B | 512 | 1 | 47.8 | 76.0 | 47.6 | 0.63 |
| | 512 | 6 | 203.1 | 302.1 | 186.9 | 0.67 |
| | 512 | 12 | 323.3 | 532.6 | 334.1 | 0.61 |
| | 512 | 24 | 230.8* | 334.2* | 63.7* | — |
| | 4096 | 1 | 34.0 | 46.8 | 33.9 | 0.73 |
| | 4096 | 6 | 135.4 | 238.3 | 148.0 | 0.57 |
| | 4096 | 12 | 228.7 | 361.8 | 273.6 | 0.63 |
| | 4096 | 24 | 206.5* | 544.9* | 329.2* | — |
| Qwen 2.5 0.5B | 512 | 1 | 90.8 | 176.8 | 154.2 | 0.51 |
| | 512 | 6 | 367.7 | 612.2 | 512.7 | 0.60 |
| | 512 | 12 | 577.4 | 1101.0 | 870.7 | 0.52 |
| | 512 | 24 | 103.2* | 1302.5* | 943.5* | — |
| | 4096 | 1 | 56.9 | 110.1 | 102.7 | 0.52 |
| | 4096 | 6 | 235.5 | 384.8 | 326.5 | 0.61 |
| | 4096 | 12 | 377.1 | 667.6 | 485.4 | 0.56 |
| | 4096 | 24 | 191.7* | 862.6* | 465.6* | — |

\* the 24-thread rows are not usable in this window: the spread between rounds was 53 %
(Peregrine, Llama 512) to 150 % (Qwen 4096), on both engines — see §1.3.

`-ub 64` is a large loss for ik at 4096 (35 % below its default) and at 24 threads, and
*level* with its default at 512 on the 1B (334 against 533 at twelve threads is 37 % below).
The fork's ub 512 path is where its requantising kernel lives (§10.3); with `-ub 64` it runs
the 4-bit kernel at every width, which is the comparison that flatters Peregrine.

### 1.2 The stage split

`PEREGRINE_PROF2=full`, `promptbench` per cell, medians of three runs. Shares are of the
engine's own `prefill_s`; the attention sub-buckets are worker-side sums divided by the pool
width. "rest" = norms, activation quantise, RoPE+bias, KV write, residual, SwiGLU, other —
the work the calling thread does itself.

| model | prompt | threads | tok/s | matmul | attention | — scores | — mix | rest | barrier wait |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama 1B | 512 | 1 | 49.0 | 86.5 % | 7.6 % | 2.1 % | 4.8 % | 2.7 % | 0.0 % |
| | 512 | 12 | 304.5 | 79.2 % | 8.0 % | 1.6 % | 3.2 % | 7.3 % | 13.7 % |
| | 4096 | 1 | 33.8 | 59.4 % | 36.4 % | 11.3 % | 24.4 % | 1.9 % | 0.0 % |
| | 4096 | 6 | 136.5 | 54.2 % | 39.6 % | 10.0 % | 19.7 % | 3.7 % | 20.7 % |
| | 4096 | 12 | 227.4 | 56.0 % | 34.5 % | 9.5 % | 18.2 % | 6.3 % | 13.2 % |
| Qwen 0.5B | 512 | 1 | 88.0 | 82.3 % | 10.3 % | 2.7 % | 6.5 % | 2.7 % | 0.0 % |
| | 512 | 12 | 563.5 | 72.3 % | 10.4 % | 1.9 % | 4.1 % | 9.3 % | 19.5 % |
| | 4096 | 1 | 59.1 | 51.1 % | 44.8 % | 13.5 % | 30.4 % | 1.7 % | 0.0 % |
| | 4096 | 6 | 233.2 | 46.6 % | 43.9 % | 11.2 % | 23.3 % | 4.1 % | 14.2 % |
| | 4096 | 12 | 377.6 | 50.3 % | 40.9 % | 10.5 % | 20.9 % | 9.5 % | 23.9 % |

The `attention` bucket is larger than scores + mix + the tile decode by 6–7 points at 4096:
the caller's `Attn` timer also covers the fork–join that runs it and the per-row prologue
(query copy, visible-range arithmetic). "Barrier wait" is the caller waiting for the pool
inside the matmuls, so it is not a summand: matmul + attention + rest + wait ≈ 100 %.

The by-format matmul slots (same runs) say the same thing as §10.5: Llama's Q4_K is 70 % and
Q6_K 30 % of the matmul at both lengths; Qwen's is Q5_0 70 %, Q6_K 22 %, Q4_K 8 %.

**What this says.** At 512 tokens the pass is a weight-matmul pass on both models (72–87 %)
and attention is 8–10 %. At 4096 attention is **35–45 % of the pass**, the largest thing
after the matmuls, and the value mix alone (18–30 %) is *twice* the score stage (9.5–13.5 %).

### 1.3 Thread scaling, and what stops scaling

Speed-up over one thread, from §1.1 (medians):

| model | prompt | engine | 6 | 12 | 24 |
|---|---|---:|---:|---:|---:|
| Llama 1B | 512 | Peregrine | 4.25× | 6.76× | — |
| | 512 | ik | 3.97× | 7.01× | — |
| | 4096 | Peregrine | 3.98× | 6.73× | — |
| | 4096 | ik | 5.09× | 7.73× | — |
| Qwen 0.5B | 512 | Peregrine | 4.05× | 6.36× | — |
| | 512 | ik | 3.46× | 6.23× | — |
| | 4096 | Peregrine | 4.14× | 6.63× | — |
| | 4096 | ik | 3.49× | 6.06× | — |

Twelve threads buy 6.2–6.8× on both engines on both models at both lengths — the thread
scaling is *not* a Peregrine deficit at 12 threads (§10.5's finding stands). The 24-thread
rows are not.

Per stage, wall time at one thread against twelve (Llama 1B, 4096; the profiles above):

| stage | 1 thread | 12 threads | speed-up | core-time at 12 t ÷ 1 t |
|---|---:|---:|---:|---:|
| weight matmuls | 72.0 s | 10.1 s | 7.1× | 1.68× |
| attention scores | 13.7 s | 1.71 s | 8.0× | 1.50× |
| attention mix | 29.6 s | 3.28 s | 9.0× | 1.33× |
| rest (caller, serial) | 2.3 s | 1.13 s | 2.0× | 5.9× |
| barrier wait | — | 2.4 s | — | — |

Qwen at 4096 is the same shape: matmul 6.5×, scores 8.2×, mix 9.3×, rest 1.2×. **The stage
that stops scaling is the weight matmul** (7.1× of 12, and its per-core rate falls 41 %),
because twelve cores stream the same weights through one memory system; attention scales
*better* than the matmul (8–9×) and the serial buckets barely move. At twelve threads
**6–10 % of a pass is on the calling thread** (1.1 s of Llama's 18.0 s, 1.0 s of Qwen's
10.8 s), and another 13–24 % is the caller waiting for its workers (imbalance, not work).

**Do the documented switches move it?** 12 threads, Llama 1B / Qwen 0.5B at 512 tokens,
medians of three rounds, default chunk (64) unless named:

| setting | Llama | Qwen |
|---|---:|---:|
| default (spin 1000 µs, dynamic rows, chunk 64) | 320.1 | 581.3 |
| `PEREGRINE_PREFILL_CHUNK=16` | 276.1 (−14 %) | 488.9 (−16 %) |
| `PEREGRINE_PREFILL_CHUNK=128` | 311.8 (−3 %) | 550.3 (−5 %) |
| `PEREGRINE_SPIN_US=0` (park at once) | 139.6 (−56 %) | 123.5 (−79 %) |
| `PEREGRINE_SPIN_US=200` | 205.0 (−36 %) | 261.9 (−55 %) |
| `PEREGRINE_SPIN_US=4000` | 313.4 (−2 %) | 506.0 (−13 %) |
| `PEREGRINE_AFFINITY=1` | 331.1 (+3 %) | 503.2 (−13 %) |
| `PEREGRINE_DYNAMIC=0` (static rows) | 265.8 (−17 %) | 520.2 (−11 %) |

So the pool's spin window and the dynamic row split are already at their tuned values: parking
costs 56–79 % on this host, and the static partition 11–17 %. The chunk is at its optimum too.
None of these is a lever on the 12-thread pass.

**At 24 threads the pool falls over, and the chunk is most of it.** Llama 1B, 512 tokens, 24
threads, medians of three rounds: default chunk 64 gives 124.9–285.8 tok/s (spread 129 %),
`PEREGRINE_PREFILL_CHUNK=128` gives 282–312 (median 308), `chunk=16` 62–155, `SPIN_US=0` 90,
`SPIN_US=4000` 64–331. Qwen is the same with a wider swing (default 103–270, chunk 128
189–512). The other tenant and 24 workers on 24 logical CPUs make every join a scheduling
lottery; **ik keeps scaling where Peregrine does not** (Qwen 512: ik 1302 tok/s at 24 threads
against 1101 at twelve, +18 %), which is the thread-scaling item §6 item 5 named, still open.
It is not shipped here: nothing about it improves the 12-thread pass, and a chunk default
chosen from measurements taken on a loaded host is not a decision this round can justify.

### 1.4 Attention: the arithmetic and the rate

Multiply-accumulates for one prefill of `N` tokens, causal, from the tensor shapes:

```
score MACs = layers × query heads × (N·(N+1)/2) × head_dim
mix   MACs = the same count (one dot per key, one weighted add per key per dim)
```

| model | layers | heads | head dim | N=512 | N=4096 |
|---|---:|---:|---:|---:|---:|
| Llama 3.2 1B | 16 | 32 | 64 | 4.30 GMAC each | 274.9 GMAC each |
| Qwen 2.5 0.5B | 24 | 14 | 64 | 2.82 GMAC each | 180.3 GMAC each |

Against the measured times (profile, wall-normalised; roof 59.0 GMAC/s per core):

| model | N | stage | one thread | rate | 12 threads (aggregate) | rate | % of roof, 1 t / 12 t |
|---|---|---:|---:|---:|---:|---:|---:|
| Llama 1B | 512 | scores | 0.219 s | 19.6 GMAC/s | 0.027 s | 159 GMAC/s | 33 % / 27 % |
| | 512 | mix | 0.494 s | 8.7 | 0.056 s | 77 | 15 % / 13 % |
| | 4096 | scores | 13.7 s | 20.1 | 1.71 s | 161 | 34 % / 27 % |
| | 4096 | mix | 29.6 s | 9.3 | 3.28 s | 84 | 16 % / 14 % |
| Qwen 0.5B | 512 | scores | 0.157 s | 18.0 | 0.017 s | 166 | 31 % / 28 % |
| | 512 | mix | 0.378 s | 7.5 | 0.037 s | 76 | 13 % / 13 % |
| | 4096 | scores | 9.36 s | 19.3 | 1.14 s | 158 | 33 % / 27 % |
| | 4096 | mix | 21.1 s | 8.6 | 2.27 s | 79 | 15 % / 13 % |

**The score kernel runs at about a third of the machine's f32 FMA roof, and the value mix at
about a sixth.** The mix is the worse of the two by a factor of two on identical MAC counts,
and it is context-independent (8.6–9.3 GMAC/s at 1 thread at both 512 and 4096), so it is not
a memory-traffic or cache problem: it is the kernel's own structure.

### 1.5 The attention kernels the prefill path uses, as they were

Read from `crates/peregrine-core/src/attn.rs` at `f3f7dd4`. One KV head at a time; the tile is
`BLOCK = 64` positions; two paths by visible context:

**Below 1024 positions (the 512-token prompt): the widened tile.** `kv.decode_block` widens
the tile's K (then V) from the cache's f16 halves into f32 scratch once per tile.

- *Scores*: `scores_avx2` (`attn.rs:2200`). 256-bit (`ymm`), **8 keys per pass, 8
  accumulators**; the head is walked in ascending 8-dimension blocks and each key's products
  go into its own accumulator with `vfmadd231ps`; the eight scores leave in one register via
  two `vhaddps` and a `vperm2f128`+`vaddps`. One score's product order is therefore
  `((d0+d1)+(d2+d3)) + ((d4+d5)+(d6+d7))`, in that association, with the same fused
  multiply-adds; the tail (fewer than 8 keys) scores one key at a time with
  `hsum_grouped_ps`, which is the same tree. The scalar twin (`scores`) sums the same
  products left to right and is the definition the tier's tests hold the vector tree to.
- *Mix*: `mix_avx512` (`attn.rs:1618`) when `head_dim % 16 == 0` (it is 64): 512-bit, **four
  16-lane accumulators = 64 dimensions at a time**, `acc = fma(p_j, v_j, acc)` over ascending
  positions `j`. One row at a time.
- *Softmax*: `vmath::max` + `exp_shift_sum` (16-lane `exp`, eight-value grouping so the
  denominator is the AVX2 tier's) + `det_exp`.

**From 1024 positions (the 4096-token prompt): the F16-direct tile** (`DIRECT_MIN_VIS`,
`attn.rs:85`). The tile is never widened: the kernels read the cache's halves and `vcvtph2ps`
them in registers.

- *Scores*: `scores_f16_pair` (`attn.rs:1373`) when two query rows share the tile — four keys
  per pass, **8 `ymm` accumulators (4 keys × 2 rows)**, each row's accumulator takes the same
  `fma(q_row, k, acc)` in the same ascending 8-dimension order as `scores_f16`, and the same
  reduction tree per key. K is converted **once per use**, i.e. once per (row pair, key,
  8 dimensions); two rows share that converted vector.
- *Mix*: `mix_f16` → `mix_f16_avx512` (`attn.rs:1653`), **one row at a time**: four 16-lane
  accumulators, `_mm512_cvtph_ps` of a 32-byte half-vector per accumulator per position,
  `acc = fma(p_j, v_j, acc)` ascending in `j`. Two rows share a V vector nowhere in this
  path — `mix_f16_pair` existed but was only called by the decode path
  (`run_f16`, `attn.rs:565`), which passes a one-entry chunk-boundary list and no banking.

**Where the thread split is.** `plan()` (`attn.rs:178`) makes `n_kv_head × row_blocks ×
pos_chunks` tasks; a prefill chunk is `nb > 1`, so `pos_chunks = 1` and `row_blocks =
ceil(2·workers / n_kv_head)` capped at `nb`. At twelve threads that is 24 tasks on both models
(8 KV heads × 3 row blocks for Llama, 2 × 12 for Qwen), each task owning whole rows of one KV
head and walking its own canonical position chunks. A row's arithmetic depends only on the
row's own visible context and the model shape — never on the task, the chunk or the pool —
which is what the invariance tests pin.

### 1.6 The candidates, ranked by what each could save

| # | candidate | share of a 4096 pass | share at 512 | evidence / estimate | verdict |
|---|---|---|---:|---|---|
| 1 | **value mix: two rows per V pass** | 18.2–30.4 % | 4.1–6.5 % | pair body measured 1.7× the two solo mixes on a hot tile; in situ 1.24× on the mix | **shipped** (§2) |
| 2 | score kernel: four rows per K block | 9.5–13.5 % | 1.6–2.7 % | already paired; a 4-row/4-key zmm body is ~1.2–1.3× the pair, so ~+1 % of a pass | not worth the test surface |
| 3 | barrier wait at 12 threads | 13–24 % | 14–24 % | the caller waits for the slowest worker; the measured switches (spin, chunk, dynamic) are all at their optimum, and the alternatives lose 11–79 % | not a kernel job |
| 4 | 24 threads / SMT | — | — | chunk 128 recovers much of it, but the host is too noisy to choose a default, and it does not touch the 12-thread pass | not shipped |
| 5 | f32 mix pair (the 512-token path) | — | 4.1–6.5 % | the same 1.7× would be +3 % of a 512 pass; a second pair body for the widened tile | not attempted |

---

## Part 2: the value mix, two rows per V pass

### 2.1 What was built

`mix_f16_pair_same[_zmm|_ymm]` (`crates/peregrine-core/src/attn.rs`): two rows of one KV head
that see the same positions are mixed in one pass over V. Per position the half-vector is
loaded and widened **once** and feeds both rows' fused multiply-adds; each row keeps its own
four accumulators and adds `p_j · v_j` ascending in `j`, exactly as `mix_f16` did. The
sixteen-lane body takes `head_dim % 16 == 0`, the eight-lane body `% 8`, and anything else
falls back to two single-row mixes.

The prefill tile's mix loop (`run_range`) pairs a row with the next one when both see the same
context (hence the same visible range and the same chunk boundaries — the `gqa` query heads of
one batch row), cuts the pass at the same chunk boundaries, and banks both rows' partials
exactly where the single-row loop banked one. The generic `mix_f16_pair` is *not* used here:
its per-position branches and its per-row V reloads measured slower than the two solo mixes in
situ (43.6 s against 29.3 s of mix core-time), which is why this body exists.

### 2.2 Why the bits cannot move

A score's value mix is `o = α·o + Σ_j p_j·v_j`, applied for `j` ascending with a fused
multiply-add per position, element by element. The pair body changes only **where the
multiplied operand came from** — one `vcvtph2ps` of the same half-vector, shared — and keeps
each row's accumulation in the same order with the same operands. f16 → f32 widening is
exact; the broadcast of `p_j` is the same value; the accumulator update is the same
instruction. The chunks a row is cut into, their boundaries and the merge order are
untouched, and grouping rows into pairs depends only on the two rows seeing the same
positions, which for a prefill chunk means they already shared a score pass.

The test `the_prefill_pair_mix_is_each_rows_own_mix` (new) checks the body against
`mix_f16` row by row, bit for bit, for every visible length 1…70, and 127, 128, 129, 511, 512,
513, 4095, 4096; head dims 64, 128, 80, 72 (the eight-lane body) and 16; on random scores and
on the edge values a softmax produces (all-zero alphas, alpha 1, a whole row of equal weights,
negative zero, a large alpha). The whole suite, and the real-model goldens, are below.

### 2.3 The measurement, before and after

Two builds that differ only in that dispatch, alternating within each round (`ab.py`), three
rounds unless stated. `before` = the S5.2 branch, `after` = this change.

| model | prompt | threads | before (median) | after (median) | ratio | per-round ratios | n |
|---|---|---:|---:|---:|---:|---|---:|
| Llama 1B | 4096 | 12 | 224.7 | **237.5** | **1.057** | 1.054, 1.054, 1.059, 1.053\* | 4 |
| Qwen 0.5B | 4096 | 12 | 365.5 | **386.3** | **1.057** | 1.046, 1.050, 1.067, 1.051\* | 4 |
| Llama 1B | 4096 | 12 | (after first) | | 1.053 | 1.053, 0.817, 1.128 | 3 |
| Qwen 0.5B | 4096 | 12 | (after first) | | 1.051 | 1.051, 1.107, 1.017 | 3 |
| Llama 1B | 4096 | 6 | 127.7 | 132.7 | 1.039 | 1.026, 1.053 | 2 |
| Qwen 0.5B | 4096 | 6 | 223.1 | 228.3 | 1.023 | 1.024, 1.022 | 2 |
| Llama 1B | 4096 | 1 | 30.6 | 31.3 | 1.023 | 1.033, 1.023 | 2 |
| Qwen 0.5B | 4096 | 1 | 56.4 | 58.5 | 1.036 | 1.055, 1.012, 1.091, 1.028\*\* | 4 |

\* the fourth round is the reversed-order run (`REVERSE=1`, "after" first), which is the check
that the forward order's small bias is not the effect: at 4096 both orders agree.
\*\* one round at 4096/1 t ran in a quiet window and read 49.9–59.6 tok/s; the median of the
four is used.

The mix's own time (profile, same cells): Llama 1B 4096 at one thread **29.3 s → 23.6 s**
(core-time, 1.24×), the pass +8 % in the same quiet window; at twelve threads the mix is
3.28 s of an 18.1 s pass before.

At 512 tokens nothing changes, because that prompt takes the widened f32 tile and the change
is in the f16-direct path — the same bytes of code run. The forward-order A/B reads 0.96
(Llama, 12 threads) and 0.98 (Qwen); with the arm order reversed, so that the host's drift
penalises the other arm, it reads 1.07 and 1.00/1.06. The ±5 % between the two orders is the
host, which is why both are printed; neither model shows a change larger than that, as
expected of a cell that does not take the new body.

Nothing else moved:

| cell (12 threads) | before | after | ratio |
|---|---:|---:|---:|
| Llama 1B decode tok/s, 512 ctx | 59.1 | 59.5 | 1.01 |
| Llama 1B decode tok/s, 4096 ctx | 47.8 | 47.5 | 0.99 |
| Qwen 0.5B decode tok/s, 512 ctx | 91.0 | 90.9 | 1.00 |
| Qwen 0.5B decode tok/s, 4096 ctx | 69.7 | 68.9 | 0.99 |
| Llama 1B, 10 users × 512 prompt + 32 tokens | 305.2 | 308.7 | 1.01 (median of 3; arms reversed 1.05) |
| Qwen 0.5B, 10 users × 512 prompt + 32 tokens | 573.7 | 524.9 | 0.92 (median of 3; arms reversed 1.05) |

The 10-user cells are ±10 % between rounds on this host (the prompt rate of ten conversations
sharing one twelve-worker pool); the reversed-order pair says the Qwen 0.92 is drift, not the
change.

---

## Part 3: the table after the change

Peregrine after = the §2 binary; ik as §1.1 (same window); three interleaved rounds, medians,
`before → after` from `ab.py`. 24-thread rows omitted (spread over 100 % on both engines).

| model | prompt | threads | Peregrine before | Peregrine after | ik `-ub 512` | ik `-ub 64` | after ÷ ik512 |
|---|---|---:|---:|---:|---:|---:|---:|
| Llama 1B | 512 | 1 | 45.1 | 45.3 | 76.0 | 47.6 | 0.60 |
| | 512 | 6 | 202.6 | 198.9 | 302.1 | 186.9 | 0.66 |
| | 512 | 12 | 315.5 | 303.4 | 532.6 | 334.1 | 0.57 |
| | 4096 | 1 | 30.6 | 31.3 | 46.8 | 33.9 | 0.67 |
| | 4096 | 6 | 127.7 | 132.7 | 238.3 | 148.0 | 0.56 |
| | 4096 | 12 | 224.7 | **237.5** | 361.8 | 273.6 | **0.66** |
| Qwen 0.5B | 512 | 1 | 87.5 | 89.0 | 176.8 | 154.2 | 0.50 |
| | 512 | 6 | 379.2 | 389.1 | 612.2 | 512.7 | 0.64 |
| | 512 | 12 | 526.3 | 515.6 | 1101.0 | 870.7 | 0.47 |
| | 4096 | 1 | 56.4 | 58.5 | 110.1 | 102.7 | 0.53 |
| | 4096 | 6 | 223.1 | 228.3 | 384.8 | 326.5 | 0.59 |
| | 4096 | 12 | 365.5 | **386.3** | 667.6 | 485.4 | **0.58** |

The 512-token rows compare two builds measured in the same rounds; their ±5–15 % spread is
the host (the "before" column of the same cells ranges 480–612 tok/s for Qwen at 512/12 t
across the session).

**What is still behind ik_llama.cpp, and why.** At 4096 tokens Peregrine is now at 0.66
(Llama) and 0.58 (Qwen) of the fork, against 0.63 and 0.56 before; at 512, 0.57 and 0.47.
The remaining gap is not attention any more: after this change attention is 35–45 % of a
4096 pass at 12 threads (down from ~40–45 %), and its kernel runs at 27 % (scores) and 14 %
(mix) of the f32 roof. The fork's lead is still the weight pass — its 512-row batch runs a
requantising kernel at 91 GMAC/s against Peregrine's exact 72–75 (§10.3, §11.4) — plus the
fact that its attention uses 512-bit kernels throughout while Peregrine's score kernel is
256-bit on every tier by measurement (§1.5). The kernels are also not the whole story at 24
threads, where the fork keeps scaling and Peregrine's pool does not.

## Files

| file | what |
|---|---|
| `sweep.py` | the grid driver (both engines, all series) |
| `ab.py` | the before/after A/B driver |
| `tables.py` | the tables above, from the JSONL |
| `pgrid-*.jsonl` | Peregrine, `PEREGRINE_PROF2=1`, before |
| `pprof-*.jsonl` | the same cells with `=full`, for the split |
| `ik-*.jsonl` | ik_llama.cpp, both `-ub` values |
| `ab-*.jsonl` | before/after, interleaved |
| `scaling-512*.jsonl` | the switch sweeps at 12 and 24 threads |

The committed evidence is the JSONL and the drivers; the corpus, the two builds and the peak
probes stay under `~/.cache/peregrine-bench/s52b/` (the corpus is a concatenation of files in
this repository, reproducible with the command above).

## Checks

All on this branch, `CARGO_TARGET_DIR=~/.cache/peregrine-bench/target-s52b`, from `~/wt-s52b`:

| command | result |
|---|---|
| `cargo fmt --all --check` | clean |
| `cargo clippy --workspace --all-targets -- -D warnings` | clean |
| `cargo test --release -p peregrine-kernels -p peregrine-core` | 456 passed, 0 failed |
| the same with `PEREGRINE_ISA=avx2` | 456 passed, 0 failed |
| the same with `PEREGRINE_ISA=scalar` | 256 passed, 0 failed (the SIMD-only tests are `cfg`-excluded); `options::tests::the_server_calibrates_when_nothing_pins_the_knobs` failed once under load ("the workers did not come back in 2000 ms") and passed on three re-runs alone |
| `cargo test --release -p peregrine-models` (native) | 184 passed, 0 failed with `PEREGRINE_GOLDEN_MODELS=Llama-3.2-1B`; see the note below |
| the same with `PEREGRINE_ISA=avx2` | 184 passed, 0 failed |
| the same with `PEREGRINE_ISA=scalar` | run in pieces (it exceeds ten minutes whole): `attn_invariance` 4, `golden` 1, `prefill_chunk` 5, `batch` 5, `engine` 15, `kq` 1, `synthetic` 32, `runtime_parity` 10, and 61 + 42 across the remaining binaries — 0 failed |
| `cargo test --release -p peregrine-server -p peregrine` | 147 passed, 0 failed |
| `cargo hack check --each-feature --no-dev-deps -p peregrine-kernels -p peregrine-core` | all 20 + 10 feature sets clean (cargo-hack is installed) |
| `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --workspace` | clean |
| `git status` under any `tests/golden/` | **nothing changed** |

**The Qwen golden cannot pass in this checkout, with or without this change.** The stored
`crates/peregrine-models/tests/golden/x86_64-avx2-avx512-vnni-gfni-vbmi/Qwen2.5-0.5B-Instruct-Q4_K_M.golden`
begins `bytes 397808192`; the file this host has is `qwen2.5-0.5b-instruct-q4_k_m.gguf`,
491,400,032 bytes — a different file, with a different token embedding. `git stash`ed to the
S5.2 branch, `cargo test --release -p peregrine-models --test golden` fails on the same
`bytes` line and the same `logits_fnv`. The Llama 1B golden — a file that *does* match — passes
on this branch, native and avx2 and scalar, i.e. the real-model logits are unchanged. The
synthetic goldens, `attn_invariance`, and `prefill_chunk` (chunks 16/32/64/128, and the KV
cache and logits compared directly) all pass on all three tiers.
