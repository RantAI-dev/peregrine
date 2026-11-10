# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). Before 1.0, minor versions may break things.

## [Unreleased]

### Added
- A release pipeline for the first release: `scripts/release/build.sh` builds a stripped, portable
  archive per target (`peregrine-vX.Y.Z-linux-x86_64.tar.gz` / `-linux-aarch64.tar.gz` with `LICENSE`/`NOTICE` and a SHA256SUMS
  line), and a `release` workflow on tags `vX.Y.Z` runs the full test suite on both Linux runners,
  asserts the x86-64 binary keeps the baseline ISA with AVX2/AVX-512 picked at run time (no
  `-C target-cpu=native`), and uploads the archives and checksums to the GitHub release with notes
  generated from PR labels. The release step itself — versioning, CHANGELOG conventions, the
  AVX2-host (H2) full-suite gate on every release candidate (S0.5) — is documented in
  `docs/dev/releasing.md`.

### Fixed
- A request body that is not valid JSON is now rejected with the OpenAI error object instead of
  axum's plain-text rejection. The status was already 400; a client that parses `error.message` can
  now read the reason. The compatibility suite checks the body as well as the code.
- Small-model decode no longer runs a Q5_0 or Q8_0 row at a time. A Q5_0 row is 22 bytes
  per 32 weights and a Q8_0 row 34, so the 32-byte activation block (and, for Q5_0, its
  32-byte lane sums) is more traffic than the weights; the per-row kernel reloaded both
  for every output row. The new eight-row kernel loads them once per block for eight
  rows, and each row still runs the per-row kernel's operations block by block, so the
  bits are unchanged (a row's result does not depend on the batch, tested bit for bit).
  On the 4-core Xeon (Qwen2.5-0.5B Q4_K_M) decode went 32.1 → 37.8 tok/s at 512 context
  and 27.1 → 29.0 at 4096 (medians of 5 and 3 alternating rounds). The eight-row kernels
  also prefetch one page ahead per row stream (`PEREGRINE_PF=0` still turns it off).
- The SwiGLU activation's quantisation now runs in the workers' SwiGLU pass instead of
  serially on the calling thread afterwards. A block's scale comes from that block's 32
  values alone, so a worker's contiguous block range writes exactly the bytes one call
  would; `QuantAct::prepare`/`fill_range` are tested against `fill` for any split. The
  decode quantise on the 0.5B fell from 0.73 ms to 0.17 ms per token.
- The prefill chunk's default now follows the worker pool. 16 is right on a pool as wide as the physical
  cores, but the engine builds `available_parallelism()` workers, which on an SMT machine is twice the
  core count. A 16-row pass is only 16 rows deep: its row-deep blocks (the rotation) give one row to 16
  workers and leave the other 8 spinning on the siblings of the cores doing the work, and every parallel
  region is short enough that its slowest worker is a large share of it. Measured on the reference
  machine at 24 workers, a forward pass carries about 20 ms that does not scale with rows, on both Bonsai
  sizes (2B and 9B), so a 512-token prompt pays it 32 times instead of 8. The default is now 64 when the
  pool is wider than the physical cores and stays 16 otherwise (`options::prefill_chunk_default_for`).
  At 24 workers, medians of four rounds: Bonsai-2B 304 prompt tok/s against 212 (+43 %), Bonsai-9B 90.7
  against 79.4 (+14 %), Llama-3.2-1B 225 against 189 (+19 %); decode is unchanged on all three. At 12
  workers 16 keeps winning (257 against 248, 69.0 against 64.9, 180 against 168), and the default stays
  16 there. `PEREGRINE_PREFILL_CHUNK` still overrides either.

### Added
- A repacked eight-row store for **Q5_0** (`Q5_08`, 176 bytes = 8 × 22, size-neutral) and an AVX2
  GEMM/GEMV on it, for the small "Q4_K_M" files whose `d_model` is not a multiple of 256 and whose
  matrices llama.cpp therefore left at Q5_0. The tile groups the fifth bits by eight-weight window
  (one byte per row per window), so the activation keeps its ordinary per-32 layout: no interleave
  to build, no 256-alignment requirement, and the GEMV and GEMM are one code path, bit-identical per
  row. Gated by the existing `PEREGRINE_KREPACK` (on below the AVX-512 tier, off on it); rows past
  the last whole group of eight keep the per-row kernel. With the repack off, decode uses the
  eight-row Q5_0/Q8_0 GEMV above and Q8_0 does regardless (it has no repacked layout). On bluefin,
  5 alternating rounds against the tree #104 lands on: Qwen2.5-0.5B prefill 271.7 → 458.2 tok/s at
  512 and 206.5 → 299.8 at 4096, decode 71.6 → 83.6 and 58.8 → 65.5; SmolLM2-360M 299.1 → 436.5 /
  199.9 → 261.1 and 88.4 → 109.8 / 57.3 → 64.5; Llama-3.2-1B flat. Against llama.cpp v0.9.0 at its
  faster flash-attention setting per row, the small models are 1.43×/1.20× (0.5B prefill at 512
  and 4096) and 1.09×/1.27× (decode); the 360M is 1.27×/1.12× and 1.17×/1.42×.
- `PEREGRINE_PROF2=1` makes `peregrine bench` print a per-token phase breakdown (matmuls
  by projection, attention, norms, quantisation, RoPE, KV writes, residuals, SwiGLU,
  barrier wait); `=full` adds the attention subtimers. Off by default, no timers in the
  default path.
- An adaptive spin window (`PEREGRINE_ADAPTIVE_SPIN`, on by default). A pool thread samples its own
  run-queue delay from `/proc/thread-self/schedstat` every 8 ms; a smoothed reading of that delay is
  the gate. Below 35 % (quiet, and two runnable threads per core) the configured window is used
  exactly as before and no A/B runs. Above it the pool tests its own join wait: it holds the current
  window or parking for a few thousand joins, then measures one window on each and keeps the lower
  mean (with a 10 % margin, so it does not flap). This is outcome-based because the same run-queue
  reading maps to different winners on different days and models — a first version mapped the
  reading to a half-window middle rung, which lost to both the full window and parking and regressed
  Gemma-3-1B under load. `PEREGRINE_ADAPTIVE_SPIN=0` keeps the window fixed for measurement.
  `PEREGRINE_ADAPTIVE_LOG=1` prints the rung, the smoothed reading and each trial decision. On H0
  (Xeon E5-2680 v4, four threads pinned to four cores, shared) the medians of five alternating
  rounds with four CPU hogs: Llama-3.2-1B prefill/decode 13.2/1.98 with the default window,
  27.2/6.17 adaptive and 27.4/6.31 at `PEREGRINE_SPIN_US=0`; Gemma-3-1B 11.1/1.41, 28.4/6.07 and
  28.1/6.11. With eight hogs: Llama 10.3/1.43, 24.4/4.62, 21.6/5.20. Scheduling only: no kernel,
  quantisation or arithmetic changes.
- `peregrine bench --threads N` fixes the worker pool width and records it in the JSON `config`, so a
  run can be pinned to the same core count as a comparison engine. The default is unchanged.
- A tunable prefill chunk (`PEREGRINE_PREFILL_CHUNK`, default 16 on a pool as wide as the physical
  cores and 64 when the pool is wider; see the fix below): the prompt tokens one weight pass
  carries. The engine prefills a prompt in pieces of this size, and the piece is how the prompt is
  *read*, not what it produces: the same prompt, read in pieces of 16, 32, 64 or 128 tokens, gives the
  same KV cache, the same logits and the same tokens (`crates/peregrine-models/tests/prefill_chunk.rs`).
  The default comes from an A/B on the reference machine (12 threads, four rotating rounds): Llama-3.2-1B
  165.9 prompt tok/s against 154.3 at the old 64 (+7.5 %), Ternary-Bonsai-2-2B 239.2 against 227.0
  (+5.4 %), Qwen2.5-0.5B 257.3 against 270.7 (-5.0 %). The tenth user's first token does not get later
  (19.8 s against 21.8 s at 64). A smaller chunk reads the weights more often (18.9 GB against 4.9 GB for
  a 512-token prompt on the 1B model) and keeps the activations in cache; the prefill is compute-bound at
  that width, so the traffic is free.
- The `gpt-oss` model family (S2.5 step 5): OpenAI's mixture of experts — attention sinks (a learned
  per-head logit in the softmax), alternating sliding-window and full attention, biases on the attention
  output, the router and every expert projection, YaRN rope scaling, the router's softmax over the
  *selected* experts, and OpenAI's `swiglu_oai` activation. The `gpt-4o` (o200k) pre-tokenizer is
  transcribed from llama.cpp v0.9.0's regex. The synthetic F16 and MXFP4 files give the same greedy
  tokens as llama.cpp; the check against a real gpt-oss file is pending.
- OpenAI's Harmony chat template, which gpt-oss's own Jinja template renders (`--chat-template harmony`):
  the fixed system block, the developer block for the caller's system message, user and assistant
  turns with the `<|channel|>final<|message|>` header, and the generation prompt. Rendered strings are
  pinned against llama.cpp v0.9.0; the synthetic gpt-oss file now carries the model's Jinja template
  and is detected as `harmony`.
- The server splits gpt-oss's Harmony reply into `reasoning_content` (the `analysis` channel) and
  `content` (the `final` channel), non-streamed and as SSE deltas, with the control tokens consumed.
  A `commentary` preamble is content and a `<|constrain|>` header is accepted. `reasoning_format`
  (`none`, `auto`, `deepseek`, `deepseek-legacy`) and `--reasoning-format`, and `reasoning_effort:
  "none"`, map as llama-server maps them: `none` keeps the analysis text in `content`, `auto`/`deepseek`
  split it, and `deepseek-legacy` also mirrors reasoning into streaming content. Qwen3's `<think>` span
  obeys the same switch. Earlier assistant turns replay only their final channel, matching gpt-oss's
  Jinja, and a Harmony `Session` re-renders its transcript so the incremental path cannot retain an
  analysis the full render drops. The reasoning-budget forcing stays Qwen3-only: Harmony has no
  `</think>` to close.
- Support for plain Qwen 3.5 files (S2.2c): a `qwen35` file with no `prism.hadamard.*` metadata is loaded
  without a rotation, and its ordinary quantised matmuls (Q8_0, F16, ...) run through the dense kernels
  instead of the ternary ones. The synthetic plain file gives the same greedy tokens as llama.cpp v0.9.0;
  the check against a real Qwen 3.5 file is pending.
- Support for GGML's Q2_K and Q3_K weight formats (AVX2 single-row and batched kernels, scalar reference).
- Support for GGML's MXFP4 weight format (AVX2 single-row and batched kernels, scalar reference).
- Support for the Gemma 3 model family (S2.4a): interleaved sliding-window (local) and global layers, four RMSNorms per layer (pre/post attention, pre/post FFN), per-head q/k RMSNorms, GeGLU activation, embedding scaled by sqrt(d_model), tied output head, sliding-window attention restriction, and distinct RoPE frequency bases for local vs global layers matching llama.cpp v0.9.0.
- An opt-in f32 KV cache (`KvPrec::F32`, `PEREGRINE_KV=f32`, `peregrine serve --kv f32`, `kld --kv f32`), for a
  like-for-like comparison against an engine whose attention reads f16 operands; the f16 cache stays the default.
- `peregrine-bench` example `kld`: KL divergence and top-1 agreement against llama.cpp on real text (`docs/dev/accuracy.md`); first results in `docs/targets.md`.
- `peregrine bench`, `peregrine tune` and tuning profiles (S1.7).
- The `qwen3moe` mixture-of-experts family (S2.5 step 3): the synthetic models load, run, and give the
  same tokens alone and batched; the check against llama.cpp on a real file is step 4.
- Support for GGML's IQ4_NL and IQ4_XS weight formats (AVX2 single-row and batched kernels, scalar reference).
- Support for GGML's Q5_K weight format (AVX2 single-row and batched kernels, scalar reference).
- Support for Gemma, Mistral (v1, v7), and Llama 2 chat templates with exact llama.cpp parity (S2.3b).
- Support for the pre-update Mistral v0.2 chat template (no system branch), and `--chat-template <name>`
  for a model file that declares no chat template (S2.3c).
- Support for SentencePiece (llama) tokenizers transcribed from llama.cpp (S2.3a).
- Support for GGML's Q4_0 weight format (AVX2 and scalar kernels).
- `peregrine-core`: CPU inference engine for GGUF models.
  - Architectures: Llama, Qwen 2, Qwen 3, Qwen 3.5 / Bonsai 2 (Gated DeltaNet hybrid).
  - Weight formats: F32, F16, BF16, Q5_0, Q8_0, Q4_K, Q6_K, and the ternary PQ2_0 / PTQ1_0.
  - x86-64 kernels for AVX2 and AVX-512 (VNNI, GFNI, VBMI), chosen at run time; scalar code elsewhere.
  - Batching engine: many users per weight pass, chunked prefill, a shared prefix store, position-keyed
    sampling (identical tokens alone or batched).
  - Lossless MTP speculation for a single user; DSpark and prompt-lookup speculation.
  - Bytes-moved telemetry against the machine's bandwidth ceiling.
- `peregrine-server`: OpenAI-compatible chat API (streaming, `n > 1`, `logprobs`, `/metrics`).
- Golden-output tests per kernel tier.
- `peregrine-bench`: the micro-benchmarks and experiments, moved out of the engine library.
- The `peregrine` facade crate and command: `peregrine serve`, `peregrine inspect <model>`, `peregrine tier`.
- Synthetic test models (llama, qwen3 in six formats; qwen35 hybrid in PQ2_0 / PTQ1_0) generated at test time,
  so CI runs every model code path and format without model files; the qwen35 hybrid also has an MTP (NextN)
  variant that runs lossless speculation with it on and off.
- Synthetic coverage for Qwen2 attention biases, Llama 3.x and linear rope scaling, engine batching (shared
  prefixes, mixed prefill/decode, mapped and per-row caches), LittleBit sidecars and accepted MTP drafts.
- Synthetic coverage of what the session and a one-slot engine must agree on (ADR 0007): the same prompt
  through `Session` and a one-slot `Engine`, the taps a DSpark drafter needs, a second user arriving
  mid-speculation, and the prefix store with a drafter attached.
- `peregrine-core`: the `Drafter` interface (ADR 0007) — one way of proposing tokens, told what the target
  committed and what was rolled back — with wrappers over today's drafters (the suffix index, DeepSeek's
  DSpark head, the model's own MTP block). Unstable until 1.0.
- `peregrine-core`: one verification step in the engine — the slot's drafter proposes, the engine prices the
  proposal against the verify-cost model and verifies it against the target's own samples. MTP speculation
  runs through it, so the engine no longer knows what a draft is (ADR 0007, S1.4 step 3).
- `peregrine-core`: the engine speculates with the suffix index and with DeepSeek's DSpark head, through the
  same verification step the MTP block goes through. `SlotCfg` names the drafter a slot asks for (none, MTP,
  suffix, DSpark with its directory); the per-slot drafter is built when the slot is admitted, and the engine
  prices the proposal, cuts the accepted run where the turn ends, and parks the DSpark head's drafting when a
  round does not pay for itself (ADR 0007, S1.4 steps 4-5).
- `SlotCfg`: the sampling `turn` (mixed into the position key) and the KV rotation choice, so a conversation
  can be given the same settings it uses today.
- Sampling penalties (`presence_penalty`, `frequency_penalty`, `repeat_penalty`, `repeat_last_n`, llama.cpp
  semantics); sampling fields the engine does not implement are refused with a 400 instead of ignored.
- `PEREGRINE_ISA` caps the kernel tier (`scalar`, `avx2`, `avx512`); CI tests the lower tiers.
- Project setup: CI on x86-64 / aarch64 Linux and macOS, MSRV and supply-chain checks, issue forms,
  governance, security policy, product, targets, roadmap, architecture, coding and structure rules, ADRs.

### Changed
- Q5_K prefill keeps eight activation accumulators in registers. `wide_q5_k` decoded a
  super-block once and then ran it against four activations at a time, folding each row
  into a `MAX_WIDE`-long stack array; it now walks eight at a time with the accumulators
  in registers, the shape #118 gave `wide_q6_k`, and the five-bit codes are still decoded
  once per super-block for the group. The arithmetic is unchanged — the same lane sums,
  the same per-block fused multiply-add and the same final horizontal sum — and a new
  test compares every window and ragged tail against the one-row kernel bit for bit.
  On bluefin (`PERGEGRINE_PROF2=full`, 512-token prompt) the Q5_K bucket runs
  140.5 → 171.9 GMAC/s, level with wide Q6_K's 183.4 and against the Q4_K tile's 282.5;
  its share of the prompt's wall falls from 30 % to 26 %. Five alternating rounds on
  Qwen3.5-2B Q4_K_M: prompt 139.4 → 148.9 tok/s at 512 (+6.8 %) and 132.4 → 139.3 at
  4096 (+5.2 %), decode flat; against llama.cpp v0.9.0 at its faster flash-attention
  setting the prompt rows move from 1.11× to 1.19× and 1.17×. Llama-3.2-1B (Q4_K/Q6_K)
  and Qwen3-1.7B (Q4_K/Q6_K) controls stay inside 1 %. The AVX2 real-model goldens for
  the three files this covers were re-recorded after the prefill-attention change
  (#115) that left them stale on main; the branch reproduces main's logits exactly on
  all of them.
- Attention's value mix widens to sixteen lanes on the AVX-512 tier. The mix is
  conversion-bound: it widens every stored half of V into f32 and accumulates it into
  the output, so `vcvtph2ps zmm` with 512-bit fused multiply-adds halves that work.
  Every path takes it — solo decode's F16-direct mix, the prefill tile's widened and
  F16-direct mixes, and a mapped multi-user row's chunked mix — and the pair kernel
  still shares each V read between two rows. The per-element order is the AVX2
  kernel's (`α·o`, then `p_j·v_j` over ascending positions) and F16C's widening is
  exact, so the widened mix is bit-identical to AVX2; the tier's real-model goldens
  for Llama-3.2-1B and Qwen2.5-0.5B came back unchanged. The softmax's `exp` is also
  sixteen wide (`exp16` is `exp8` lane for lane) with the eight-value f64 sum grouping
  kept, so its denominator is unchanged too. Attention-only at 4096 positions, one
  thread: 32.9 -> 29.9 ms/token (medians of three); at twelve threads the mix pass
  falls from ~2.55 to ~2.08 ms per token and attention from ~6.51 to ~5.93 ms.
  Whole-pass decode (medians of five alternating rounds, twelve threads): Llama-3.2-1B
  59.9 -> 60.9 tok/s at 512 and 47.9 -> 49.1 at 4096; Qwen2.5-0.5B 98.4 -> 99.8 at 512
  (nine rounds) and 71.7 -> 73.3 at 4096. Prompt processing is a wash inside the
  host's spread. Scores stay on the AVX2 tree on every tier: a zmm score row needs a
  sixteen-lane horizontal reduction and, measured at every visible length at head
  width 64, the zmm variants (including one that packs two keys into the two halves
  to keep the AVX2 reduction tree exactly) were 30–70 % slower. One score tree
  everywhere also keeps a row's score bits independent of its path by construction.
- Prefill attention reads the F16 K/V cache directly. The multi-row tile used to be
  widened f16 -> f32 before every score and every value mix; it now hands the cache's
  halves to the same register-blocked kernels the decode path reads, so a K or V tile
  is half the bytes and the widening pass is gone. The f16 kernels are the widened
  kernels' FMAs in the same order (`f16_direct_kernels_match_the_widened_kernels_bit_for_bit`),
  so a row's bits are unchanged: the real-model goldens for Llama-3.2-1B and
  Qwen2.5-0.5B do not move, nor do the synthetic ones at either tier. The path is
  taken once a row's visible context reaches 1024 positions; shorter prompts keep the
  widened tile, where the two measure the same. Prompt processing at 4096 tokens on
  four cores (H0, Llama-3.2-1B Q4_K_M): 75.9 -> 79.4 tok/s (medians of five
  alternating rounds); attention's share of the pass fell from 51.8 to 46.1 ms per
  16-token chunk.
- The repacked Q4_K prefill kernel takes eight activation rows per weight decode, and a
  sixteen-row prompt chunk shares one scale/min/code unpack between its two groups. The old
  kernel decoded a tile's scales, its mins and its sixteen windows once per group of four
  activations, so a 16-row chunk decoded every tile four times; the per-row arithmetic is
  untouched (the same `vpmaddubsw` pairs, the same int32 sum and float fold in the same order),
  so a row's bits do not move and the GEMV still agrees with the GEMM bit for bit. The
  sixteen-row block predecodes the super-block's window codes once and hands them to both
  groups, and the min term is one `vpmaddwd` per sub-block per row. On H0 (Xeon E5-2680 v4,
  AVX2) a 2048-wide row streamed from memory with sixteen activations went 32.4 -> 40.1 GMAC/s
  (33.1 -> 45.7 with the tile in L2); four pinned cores, on top of the Q6_K and attention
  changes already merged: Llama-3.2-1B Q4_K_M prefill 512 tokens 110.2 -> 120.1 tok/s and
  4096 83.2 -> 89.6, Qwen2.5-0.5B 232.4 -> 235.7 and 156.0 -> 161.5 (medians of five
  alternating rounds; Qwen is mostly Q5_0 and Q8_0, so little of it is this kernel). Decode is
  untouched: the GEMV is arithmetically the same and still agrees with the GEMM bit for bit.
  The prefill chunk is unchanged.
- The wide Q6_K prefill kernel keeps its activations' accumulators in registers. A
  super-block's codes are still decoded once, but they are now consumed by a group of
  eight activations whose eight integer lane sums and eight f32 accumulators never leave
  the registers: the old form kept a `MAX_WIDE`-long array in memory and reloaded it once
  per block per row. The arithmetic and its order are unchanged — the same `maddubs`/`madd`
  lane sums, the same per-block fused multiply-add, the same lane-wise float accumulation
  and final horizontal sum — so a row's bits do not move and do not depend on the width of
  the window it lands in. On H0 an 8192-wide row with sixteen activations went 6352 ->
  5486 ns (20.6 -> 23.9 GMAC/s), and the width cliff above sixteen activations is gone
  (32 activations: 17144 -> 11809 ns). Llama-3.2-1B Q4_K_M prefill at 4096 tokens on four
  cores: 76.1 -> 82.8 tok/s (medians of five alternating rounds).
- Prefill's remaining serial element-wise passes run in the pool. The attention output's
  quantisation and the two residual adds were loops on the calling thread between parallel
  regions; above one row they now go through the same row-parallel helper the norms and
  SwiGLU use. Element-wise and row-disjoint, so the bits do not move; decode keeps the
  serial form.
- Prefill attention now tiles over query rows as well as positions. A prefill task
  carries tens of query rows over one KV head, all scoring the same positions, so the
  64-position tile is widened once and walked in two phases over one buffer: keys
  first, scored for every row, then values, mixed into every row. Widening K and V
  together overflowed L1 and every row re-read both tiles from L2. Two rows share each
  loaded K vector in a register-blocked score kernel (their own accumulators, chunk
  boundaries and merge order are untouched), and the widened value mix now loads a
  position's eight value vectors into registers before the FMAs. A row's bits are
  unchanged: the tests read a 512-position context as a solo decode and as prompt
  pieces of 3, 16 and 64 rows on pools of 1, 2, 4 and 8 workers and require identical
  bits, for three attention geometries. On the 4-core Xeon (H0), attention-only
  prefill at 4096 fell from 16.6 s to 12.4 s on Llama-3.2-1B Q4_K_M and from 11.2 s to
  6.2 s on Qwen2.5-0.5B Q4_K_M. End-to-end prompt processing (medians of three
  alternating rounds) went 74.1 -> 79.9 tok/s (+7.8 %) and 130.7 -> 159.0 (+21.6 %) at
  4096, and 99.4 -> 103.5 (+4.1 %) and 221.0 -> 234.8 (+6.3 %) at 512. Decode does not
  move: attention-only decode at 4096 is 10.7-11.0 ms/token on both builds, and the
  end-to-end decode deltas of six alternating rounds have paired medians of -0.2 %
  and +3.1 % against a per-round spread of 20 %.
- Decode attention reads the F16 KV cache directly instead of widening a 64-position
  tile to f32 first. The stored halves are widened in registers (`vcvtph2ps`), the
  score pass is register-blocked four keys deep with two query rows sharing each
  loaded K vector, and the value mix keeps two rows' accumulators in registers while
  they share each loaded V vector. Prefill still widens a block once and lets every
  query row of the task read that tile: with tens of rows per block one widening
  amortises, whereas re-converting per row pair does not. Both paths use the same
  per-key reduction tree, so a row's bits are the same whether it is decoded alone,
  read in a prefill chunk, verified in a batch or run on a different pool width
  (`crates/peregrine-core/src/attn.rs`, `crates/peregrine-models/tests/attn_invariance.rs`).
  The tail key of a score block reduces with the same tree as a key inside one, so a
  position's score no longer depends on where its block ends; that is a one-time change
  to logits for contexts whose last tile ends mid-group, and the goldens are re-recorded.
  On the 4-core Xeon (H0), attention-only at 4096 context fell from 8.17 to 6.88
  ms/token on Qwen2.5-0.5B Q4_K_M and from 14.28 to 11.68 on Llama-3.2-1B Q4_K_M
  (medians of five alternating runs; decode attention is 25-30 % of a token, so the
  end-to-end token rate moved +0.4 % and +2.8 % respectively, within the host's
  noise). Prefill attention was unchanged (11.4-12.0 s and 16.8 to 16.4 s over a 4096
  token prompt), and its end-to-end rate was within noise.
- A decode step's attention is one parallel region for every decoding user, not one per
  user. Each user has a separate cache, so the dense layer used to call attention once
  per user per layer — ten fork-joins per layer at ten users, each splintering the same
  KV head's positions across the pool. The single-row runs now become one task per
  (KV head, row) in a single region; a row walks its own canonical position chunks with
  the same per-chunk kernel a solo decode chooses (the F16-direct one where the cache
  allows it) and merges them in the same order, so its arithmetic is its solo arithmetic
  and does not depend on the batch (a new test pins three users' rows against their solo
  decodes bit for bit). On the 12-core desktop with the server's default 24-thread pool
  (Llama-3.2-1B Q4_K_M, 512 positions, ten users) the step fell from 82-90 to 69-72 ms
  and ten-user aggregate throughput from 63.2 to 76.8 tok/s (+21.5 %; llama.cpp v0.9.0
  `-fa auto`: 78.0 tok/s), with decode-only aggregate 119.4 -> 157.4 (+31.8 %). On a
  12-thread pool the change is neutral (ten users 76.4 -> 76.5), because ten one-row
  calls already fit one wave of twelve workers.
- A prompt longer than 1024 tokens is prefilled in 64-token pieces instead of the tuned
  16. At 4096 positions with ten users the time to first token fell from 158.3 s to
  152.3 s (llama.cpp v0.9.0 `-fa auto`: 144.2 s), and single-user prefill of that prompt
  is no slower. The chunk changes how the prompt is read, not what it produces.
- `PEREGRINE_PROF2=1` now prints a periodic per-bucket breakdown of the server's engine
  thread to stderr (steps, rows per step, wall per step, matmul by phase, attention,
  quantise, barrier wait); `=full` adds the inner-loop timers.
- The startup calibration is off by default: `PEREGRINE_CALIBRATE=1` turns it on (S2.6 decision 6).
  A wrong thread count costs far more than the tuned default it replaces, and the calibration is only
  switched on by default once it has passed A/Bs on two machines.
- `Session` is a conversation layer over a one-slot `Engine` (ADR 0007, S1.4): the chat history, the template,
  the prompt, the reuse of the positions the previous turn computed and the context shifting stay in the
  session; the generation, the speculation and the verification of the drafts are the engine's. Its public API
  is unchanged.
- `peregrine bench` measures the engine. The benchmark used to walk a prefill and a decode loop of its own,
  beside the runtime the server runs; it now admits a slot and lets the engine do the work, so the numbers it
  reports are the numbers the engine charges a request. The bytes a token moves are the same, and the JSON
  keys are unchanged.
- The GeGLU kernel's `tanh` runs on the CPU's FMA instruction where there is one: the scalar arithmetic is
  compiled a second time inside a `#[target_feature(enable = "fma")]` function, so the `mul_add`s become
  `vfmadd` instead of a call to the C library's `fmaf` (8.3x on the kernel, bit for bit the same result).
- F32/F16/BF16 matrices multiply f32 activations instead of int8-quantised ones (output change for those formats).

### Removed
- The last of the old runtime and the old speculation paths (ADR 0007, S1.4): `Session::probe_verify_curve`
  (the engine's verify-cost model measures those widths now, on the engine's own forward passes),
  `dspark::Drafter::draft`, `SuffixIndex::propose`, `Engine::slot_cost`, `DrafterChoice::Mtp` (an alias for
  `DrafterChoice::None`) and `Needs::history` (the engine never read it). The interfaces are unstable until 1.0.

### Fixed
- The startup calibration no longer stops its decode-thread sweep at the first candidate that fails to beat
  the previous best by 5%: one noisy high early sample froze it at one or two threads and the 9B decoded
  at 5.6 tok/s instead of 23.6. Every candidate is measured now, the default (the physical cores) stands
  unless a smaller count is at least as good (median bandwidth ≥ 98% of the default's, both measurements
  quiet), and the count never goes below half the physical cores.
- The A1 dequantisation oracle is re-cut from one pinned SmolLM2-360M-Instruct Q4_K_M file (size and
  FNV-1a checked at test time, SHA-256 in the failure message), so another upload fails loudly instead
  of looking like a kernel bug.
