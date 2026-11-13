# Benchmarks: Peregrine against llama.cpp

Where Peregrine stands against llama.cpp on the same machine, the same model file and the same thread
count, and, on the desktop, against the ik_llama.cpp fork. Embedding and decision models are also
compared with OpenVINO, ONNX Runtime, fastembed and text-embeddings-inference. Wins and losses are both reported here. The raw data, method and machine records are in
[`benchmarks/results/`](../benchmarks/results/); the targets these numbers are held to are in
[targets.md](targets.md).

![Where Peregrine is fastest](img/bench-wins.svg)

## Summary

- **Single-user decode wins every completed throughput row.** On the 12-core desktop, Peregrine leads
  llama.cpp's faster flash-attention setting by **1.13× to 1.69×**: Llama 3.2 1B is 1.31× at 512 and
  1.60× at 4096, Qwen 2.5 0.5B is 1.37× and 1.57×, SmolLM2 360M is 1.34× and 1.69×, and gemma-3-1b-it is
  1.35× and 1.13×. On the 4-core Ryzen 3 (bluefin) all three models are ahead, from **1.07× to 1.34×**.
  The lead grows with context because attention reads each cached key and value once per group of heads.
- **Multi-user throughput is 1.28–1.80× faster.** With 10 concurrent users, Peregrine serves **1.28–1.43×**
  llama.cpp's aggregate throughput on the 12-core desktop (1.13–1.38× at 4 users) and **1.47–1.80×** on
  the 4-core Ryzen 3 (1.80× on SmolLM2 360M at 4096; 1.73× at 512).
- **Prompt processing is ahead on all three hosts.** Prompt throughput is **1.03–1.30×** on the 12-core
  desktop, **1.03–1.43×** on the 4-core Ryzen 3, and **1.78–2.45×** on the 20-core ARM Cortex-X925/A725
  (`aarch64-neon-i8mm`).
- **Bonsai 2 (ternary): 1.4–2.3× decode, 1.9–2.6× prompt, and 1.8–2.8× with 10 users** against PrismML's
  llama.cpp fork across x86-64 (AVX-512 and AVX2) and ARM64 (NEON). Upstream llama.cpp cannot load
  `PQ2_0` files, so PrismML's fork is the comparison engine for ternary models.

- **Against ik_llama.cpp, prompt processing is behind.** On the desktop the fork processes prompts about
  twice as fast (Peregrine at 0.39-0.58×); decode is level (0.89-1.06×) and Peregrine is ahead with 10
  users (1.05-3.09×). See [Against ik_llama.cpp](#against-ik_llamacpp-desktop-avx-512).

- **Embeddings and decisions: fastest engine in 55 of 68 embedding cells and all nine decision cells on
  the 4-core Ryzen 3**, against llama.cpp, OpenVINO, ONNX Runtime, fastembed and
  text-embeddings-inference at equal accuracy. bge-m3, EmbeddingGemma, Qwen3-Embedding and Nemotron 3.5
  Embed are **1.18–1.62×** llama.cpp and decisions **1.33–1.84×**. The 13 cells behind are the two small
  BERT encoders at 128 and 512 tokens (**0.85–0.99×** OpenVINO and ONNX Runtime). On ARM every measured
  model is ahead of llama.cpp except four near-parity cells (0.95–0.99×). See
  [Embeddings, rerank and decisions](#embeddings-rerank-and-decisions).

**For chat models against llama.cpp, no completed throughput row is below 1.0×** on any host. The single remaining loss is time to first
token (TTFT) with 10 users on bluefin for Llama 3.2 1B at 4096 context (1.04×, Peregrine slower); at
512 context Peregrine is faster on every TTFT row (0.66–0.86× on bluefin, 0.70–0.86× on the desktop).

## Test machines

| | Desktop (H1) | Bluefin (H2) | DGX Spark (H3) |
|---|---|---|---|
| CPU | AMD Ryzen AI 9 HX 370 (Zen 5), 12C/24T | AMD Ryzen 3 PRO 4350G (Zen 2), 4C/8T | 10 × Cortex-X925 + 10 × Cortex-A725, 20C |
| SIMD tier | `x86_64-avx2-avx512-vnni-gfni-vbmi` | `x86_64-avx2` | `aarch64-neon-i8mm` / `aarch64-neon-dotprod` |
| Threads used | 12 (one per physical core) | 4 (one per physical core) | 10 (X925 cluster) and 20 (all cores) |
| OS | Linux (WSL2) | Fedora (bare metal) | Ubuntu Linux (bare metal) |
| Results | [`desktop-zen5-2026-11-09`](../benchmarks/results/desktop-zen5-2026-11-09/README.md) | [`bluefin-zen2-2026-11-08-q5`](../benchmarks/results/bluefin-zen2-2026-11-08-q5/README.md) | [`spark-arm-2026-11-09-i8mm`](../benchmarks/results/spark-arm-2026-11-09-i8mm/README.md) |

All comparisons run llama.cpp v0.9.0 (`cd959ba`) against Peregrine on the exact same GGUF files
(verified by SHA-256), with 256 greedy tokens per request, prompts calibrated to the target context,
three interleaved rounds, and the median reported. **Each ratio is measured against llama.cpp's faster
flash-attention setting (`-fa auto` or `-fa off`) for that specific row.**

## Single-user decode

![Single-user decode](img/bench-decode.svg)

| Machine | Model | Context | Peregrine (tok/s) | llama.cpp (tok/s) | Peregrine ÷ llama.cpp |
|---|---|---|---|---|---|
| Desktop | Llama 3.2 1B | 512 | 60.6 | 46.4 | **1.31×** |
| Desktop | Llama 3.2 1B | 4096 | 48.5 | 30.4 | **1.60×** |
| Desktop | Qwen 2.5 0.5B | 512 | 95.5 | 69.5 | **1.37×** |
| Desktop | Qwen 2.5 0.5B | 4096 | 71.9 | 45.8 | **1.57×** |
| Desktop | SmolLM2 360M | 512 | 102.9 | 77.0 | **1.34×** |
| Desktop | SmolLM2 360M | 4096 | 70.8 | 41.9 | **1.69×** |
| Desktop | gemma-3-1b-it | 512 | 59.9 | 44.3 | **1.35×** |
| Desktop | gemma-3-1b-it | 4096 | 46.3 | 41.0 | **1.13×** |
| Bluefin | Llama 3.2 1B | 512 | 46.8 | 42.5 | **1.10×** |
| Bluefin | Llama 3.2 1B | 4096 | 36.8 | 29.3 | **1.26×** |
| Bluefin | Qwen 2.5 0.5B | 512 | 83.5 | 77.7 | **1.07×** |
| Bluefin | Qwen 2.5 0.5B | 4096 | 65.9 | 52.2 | **1.26×** |
| Bluefin | SmolLM2 360M | 512 | 108.9 | 94.5 | **1.15×** |
| Bluefin | SmolLM2 360M | 4096 | 63.8 | 47.5 | **1.34×** |

## Prompt processing

![Prompt processing](img/bench-prefill.svg)

| Machine | Model | Context | Peregrine (tok/s) | llama.cpp (tok/s) | Peregrine ÷ llama.cpp |
|---|---|---|---|---|---|
| Desktop | Llama 3.2 1B | 512 | 272.6 | 252.7 | **1.08×** |
| Desktop | Llama 3.2 1B | 4096 | 195.3 | 174.3 | **1.12×** |
| Desktop | Qwen 2.5 0.5B | 512 | 479.3 | 423.8 | **1.13×** |
| Desktop | Qwen 2.5 0.5B | 4096 | 355.1 | 272.4 | **1.30×** |
| Desktop | SmolLM2 360M | 512 | 451.1 | 436.3 | **1.03×** |
| Desktop | SmolLM2 360M | 4096 | 292.9 | 263.9 | **1.11×** |
| Desktop | gemma-3-1b-it | 512 | 284.9 | 258.1 | **1.10×** |
| Desktop | gemma-3-1b-it | 4096 | 279.6 | 247.9 | **1.13×** |
| Bluefin | Llama 3.2 1B | 512 | 195.7 | 188.1 | **1.04×** |
| Bluefin | Llama 3.2 1B | 4096 | 145.0 | 141.4 | **1.03×** |
| Bluefin | Qwen 2.5 0.5B | 512 | 434.8 | 303.7 | **1.43×** |
| Bluefin | Qwen 2.5 0.5B | 4096 | 277.6 | 236.5 | **1.17×** |
| Bluefin | SmolLM2 360M | 512 | 443.2 | 318.9 | **1.39×** |
| Bluefin | SmolLM2 360M | 4096 | 228.5 | 217.1 | **1.05×** |

Re-measured quiet for S5.4i (medians of three interleaved rounds, 12 threads): Llama 3.2 1B
**349.1** against 290.7 (1.20×) at 512 tokens and **264.6** against 195.0 (1.36×) at 4096, and
Qwen 2.5 0.5B **641.1** against 477.5 (1.34×) and **450.1** against 305.1 (1.48×). The Qwen
512-token cell that S5.2c read at 0.89× on a host loaded at 16-22 is 1.34× here; that window did
not separate the two engines, as S5.2c said.

## Multi-user throughput

![Total throughput, 10 users at once](img/bench-users10.svg)

| Machine | Model | 4 users (tok/s) | 10 users (tok/s) | 10 users, Peregrine ÷ llama.cpp |
|---|---|---|---|---|
| Desktop | Llama 3.2 1B | 70.3 vs 60.3 | 83.6 vs 61.8 | **1.35×** |
| Desktop | Qwen 2.5 0.5B | 108.4 vs 96.0 | 121.7 vs 93.6 | **1.30×** |
| Desktop | SmolLM2 360M | 117.1 vs 85.0 | 146.6 vs 102.4 | **1.43×** |
| Desktop | gemma-3-1b-it | 63.4 vs 49.6 | 69.3 vs 54.1 | **1.28×** |
| Bluefin | Llama 3.2 1B | 53.7 vs 39.6 | 55.5 vs 37.7 | **1.47×** |
| Bluefin | Qwen 2.5 0.5B | 104.3 vs 65.7 | 108.4 vs 68.4 | **1.59×** |
| Bluefin | SmolLM2 360M | 114.2 vs 66.2 | 116.8 vs 67.6 | **1.73×** |

Aggregate tokens per second across all concurrent users at a 512-token context. At a 4096-token context
on bluefin, Peregrine leads by **1.43×** (4 users) and **1.52×** (10 users) on Llama 3.2 1B, **1.35×**
and **1.40×** on Qwen 2.5 0.5B, and **1.55×** and **1.80×** on SmolLM2 360M.

## Against ik_llama.cpp (desktop, AVX-512)

ik_llama.cpp is the llama.cpp fork most often reported faster than upstream on x86. It runs the
same GGUF files, so it was measured on the desktop on 2026-11-11 with its best documented CPU
settings (an AVX-512 build, `-rtr` run-time repacking, its own flash attention), beside llama.cpp
and Peregrine in the same session. Ratios are Peregrine ÷ ik_llama.cpp; above 1 means Peregrine is
faster. Medians of three interleaved rounds.

| Model | Context | Decode (tok/s): Peregrine / ik | Ratio | Prompt (tok/s): Peregrine / ik | Ratio | 10 users (tok/s): Peregrine / ik | Ratio |
|---|---|---|---|---|---|---|---|
| Llama 3.2 1B | 512 | 56.6 / 54.6 | **1.04×** | 274.1 / 526.8 | 0.52× | 85.1 / 58.5 | **1.45×** |
| Llama 3.2 1B | 4096 | 40.6 / 42.3 | 0.96× | 208.4 / 359.0 | 0.58× | not completed | n/a |
| Qwen 2.5 0.5B | 512 | 93.9 / 91.9 | **1.02×** | 470.9 / 1116.9 | 0.42× | 150.4 / 100.4 | **1.50×** |
| Qwen 2.5 0.5B | 4096 | 65.3 / 69.1 | 0.95× | 379.1 / 662.9 | 0.57× | 20.2 / 6.5 | **3.09×** |
| SmolLM2 360M | 512 | 97.9 / 92.4 | **1.06×** | 426.5 / 1030.6 | 0.41× | 132.0 / 74.8 | **1.76×** |
| SmolLM2 360M | 4096 | 57.8 / 55.5 | **1.04×** | 304.5 / 556.0 | 0.55× | not measured | n/a |
| gemma-3-1b-it | 512 | 54.3 / 54.0 | **1.01×** | 254.3 / 644.7 | 0.39× | 76.3 / 72.7 | **1.05×** |
| gemma-3-1b-it | 4096 | 47.5 / 53.4 | 0.89× | 281.2 / 569.4 | 0.49× | not measured | n/a |

What this says:

- **Prompt processing: ik_llama.cpp is about twice as fast on every row** (Peregrine at 0.39-0.58×,
  and 0.61-0.81× of it measured again quiet in S5.4i: Qwen 2.5 0.5B 641.1 against 1 051.0 tok/s at
  512 tokens and 450.1 against 691.8 at 4096, Llama 3.2 1B 349.1 against 440.2 and 264.6
  against 326.4).
  This is Peregrine's largest measured loss. The fork repacks the k-quant weights at load (`-rtr`)
  for its own interleaved AVX-512 kernels. Where Peregrine's time goes in comparison has not been
  profiled yet.
- **Decode: level.** Peregrine is 1.01-1.06× at 512 context and 0.89-1.04× at 4096.
- **Many users: Peregrine is ahead on every completed row** (1.05-3.09× with 10 users), because it
  batches all users into one pass over the weights. At 4 users one row is behind (gemma-3-1b-it
  at 512 context, 0.85×).
- Time to first token with 10 users is behind on two 512-context rows (SmolLM2 1.09× and
  gemma-3-1b-it 1.34× of ik_llama.cpp's time) and ahead on the others.

Not measured: OpenVINO GenAI, the 4096-context multi-user rows for SmolLM2 and gemma-3-1b-it, and
ik_llama.cpp on bluefin and on ARM. The 4096-context multi-user rows are single rounds. Method,
settings and raw data: [`benchmarks/results/desktop-zen5-2026-11-11-ik/`](../benchmarks/results/desktop-zen5-2026-11-11-ik/README.md).
The other tables on this page were measured earlier and against llama.cpp only.

### The opt-in fast prompt mode (`--fast-prompt`, lossy, off by default)

Not the default engine — a separate, opt-in mode, and this table is kept apart from the
tables above because it trades accuracy for speed. Prompt tokens read their Q4_K and Q6_K
weights from an 8-bit copy built at load, and on AVX-512 a prompt row's attention is computed in
the mode's own tiled summation order too (S5.4j); generated rows keep the exact kernels and the
exact attention on every tier.
2026-11-13, desktop, AVX-512, the whole CPU set pinned (`taskset -c 0-23` for 12 threads,
`taskset -c 0` for 1), five interleaved rounds per 512-token cell and three per 4096-token cell,
medians, the 1-minute load recorded beside every run. Prompt tok/s:

| Model | prompt / threads | Per. default | Per. fast | ik (defaults) | ik (`-rtr`) | llama.cpp | fast ÷ default | fast ÷ ik | fast ÷ ik `-rtr` | fast ÷ llama |
|---|---|---|---|---|---|---|---|---|---|---|
| Llama 3.2 1B | 512 / 12 | 332.8 | **462.2** | 431.0 | 561.9 | 287.4 | **1.39×** | **1.07×** | **0.82×** | **1.61×** |
| Llama 3.2 1B | 512 / 1 | 54.1 | **79.1** | 65.3 | 83.8 | 43.8 | **1.46×** | **1.21×** | **0.94×** | **1.81×** |
| Llama 3.2 1B | 4096 / 12 | 257.6 | **338.6** | 319.5 | 380.2 | 198.9 | **1.32×** | **1.06×** | **0.89×** | **1.70×** |
| Llama 3.2 1B | 4096 / 1 | 39.8 | **52.6** | — | 57.0 | 30.5 | **1.32×** | — | **0.92×** | **1.72×** |
| Qwen3 1.7B | 512 / 12 | 227.9 | **300.5** | 289.7 | 368.3 | 189.7 | **1.32×** | **1.04×** | **0.82×** | **1.58×** |
| Qwen3 1.7B | 512 / 1 | 36.7 | **53.3** | 43.6 | 55.3 | 26.8 | **1.45×** | **1.22×** | **0.96×** | **1.99×** |
| Qwen3 1.7B | 4096 / 12 | 168.8 | **215.5** | 206.1 | 245.2 | 125.5 | **1.28×** | **1.05×** | **0.88×** | **1.72×** |
| Qwen3 1.7B | 4096 / 1 | 24.9 | **31.3** | — | 32.2 | — | **1.26×** | — | **0.97×** | — |

**Cells below 1.0, named:** against ik_llama.cpp with `-rtr` (its own fast path, the one
that uses the same `q8_k_r16` arithmetic) the fast mode is behind on **all eight** cells
(0.82-0.97×); against ik at its defaults it is ahead on all six measured cells (1.04-1.22×).
The fast mode is ahead of the exact default everywhere (1.26-1.46×) and of llama.cpp everywhere
(1.58-1.99×). S5.4j's change is in this table already: the fast mode was 0.80-0.95× `-rtr` in
S5.4i's window (+2-6 % per cell from the attention order and the pool's gap-matched spin
window), and the mode's bound is re-baselined to 0.0040 (design §4; the owner's confirmation
pending).

**Where the fast pass loses, measured (S5.4i, extended by S5.4j).** Not in the 8-bit kernel: in
situ on Llama 3.2 1B at 512 tokens it runs at **96.2 GMAC/s for Q4_K and 103.9 for Q6_K**
(`PEREGRINE_PROF2`), against the fork's own `q8_k_r16` at **94.9 GMAC/s** at 512 activation
rows and 85.1 at 256, called through the fork's library. At 12 threads **36 % of the
512-token pass is not the kernel** (the caller's wait for parked workers 10.4 %, the caller's
serial steps 11.2 %, attention 9.1 %, the activation's fill and the output's untile 3 %), and at
4096 tokens **attention is 38 % of the pass against 20.5 % of the fork's own** (`-fa 0`:
399.9 → 317.9 tok/s), i.e. 2.1× the fork's attention in absolute time. S5.4j gives a prompt row
its own attention order (17 % off attention's own time at 512/12, 3 % at 4096/12, 2-6 % of a pass) and makes the pool's
spin window follow the measured gap between joins (a prompt pass's joins are 1-5 ms apart
against a 1 ms window: +3.3 % on the 512-token 12-thread cell); the rest of the attention gap is
the fork's **int8** dot product, a different arithmetic. The account, both engines' timings and
the eight cells: S5.4i's
[`benchmarks/results/desktop-zen5-2026-11-12-s5.4i/`](../benchmarks/results/desktop-zen5-2026-11-12-s5.4i/README.md)
and S5.4j's
[`benchmarks/results/desktop-zen5-2026-11-13-s5.4j/`](../benchmarks/results/desktop-zen5-2026-11-13-s5.4j/README.md).

**Accuracy** (the full report and the F16 comparison:
[`benchmarks/results/desktop-zen5-2026-11-12-s5.4c/`](../benchmarks/results/desktop-zen5-2026-11-12-s5.4c/README.md)):
on Llama 3.2 1B Q4_K_M the prompt's logits move **0.0035 nats of mean KL divergence** from
the default engine's, median 0.0025, 99 % 0.024, with 96.75 % of positions keeping the same
top token; RMS Δp 1.41 % (S5.4j re-measured the same corpus with the attention order in:
0.003464, 96.67 % same top token). That is 1.16 % of the model's distance from F16 (0.0386), and it is
the same arithmetic ik_llama.cpp's `-rtr` path measures (0.0034 on the same file and corpus).
The other four gate models: 0.0026-0.0171 nats. Task scores (GSM8K/MMLU/HumanEval+, the four
gate models) are not significantly different from the default engine's in any cell (largest
+7.5 points at n = 80, all McNemar p ≥ 0.15). The mode costs +957 MiB of resident memory and
0.5-1.7 s of load on the 1B; decode is unchanged (63.1 → 64.5 tok/s after the prompt).
It stays off by default. These are the AVX-512 numbers, where the mode takes Q4_K and Q6_K;
which formats take the form is per tier (S5.4f): AVX2 takes Q6_K only (a fast prompt is
**1.05-1.08×** the exact one on the 1B there, `benchmarks/results/bluefin-2026-11-12-s5.4f/`),
aarch64 takes Q4_K and Q6_K, and a tier with no SIMD body takes none, so the option changes
nothing there.

**The alternative the owner has to choose against:** the same 8-bit codes with one scale per
**32** weights (the fork's *default* wide path, 0.000945 nats on the 1B against `-rtr`'s
0.003401). Measured as a kernel prototype (S5.4i §5j, nothing dispatches it):
**0.61-0.63× the per-256 form's speed** — 66.0 against 104.3 GMAC/s at 64 activation rows,
0.81-0.84× the exact kernel — for **1.25 bytes per weight** (1 164 MiB on the 1B against 957)
and a per-weight RMS only **1.39×** smaller (1.048e-4 against 1.458e-4 on the 1B's Q4_K). At
that rate a per-32 fast prompt mode would be **0.86× the exact engine**, i.e. slower than
turning the option off. The numbers, side by side:
[`benchmarks/results/desktop-zen5-2026-11-12-s5.4i/`](../benchmarks/results/desktop-zen5-2026-11-12-s5.4i/README.md).

## Bonsai 2 (ternary) vs PrismML's fork

Bonsai 2's ternary models (`PQ2_0`, ggml type 142) are compared against
[PrismML's llama.cpp fork](https://github.com/PrismML-Eng/llama.cpp) (`62c418c`, branch `prism`).
"Fork best" is the faster of the fork's flash-attention settings (`-fa on` or `-fa off`) for each row.
Full manifests: [desktop](../benchmarks/results/2026-11-09-bonsai2-prism/) and
[bluefin](../benchmarks/results/bluefin-zen2-2026-11-08-bonsai2-prism/).

![Bonsai 2: Peregrine against PrismML's fork](img/bench-bonsai.svg)

### Desktop (12 cores, AVX-512)

| Model | Context | Users | Metric | Peregrine | Fork best | Ratio |
|---|---|---|---|---|---|---|
| Bonsai 2 2B PQ2_0 | 512 | 1 | Decode tok/s | 75.14 | 34.89 | **2.15×** |
| Bonsai 2 2B PQ2_0 | 512 | 1 | Prompt tok/s | 270.24 | 119.35 | **2.26×** |
| Bonsai 2 2B PQ2_0 | 512 | 10 | Aggregate tok/s | 75.77 | 29.56 | **2.56×** |
| Bonsai 2 2B PQ2_0 | 512 | 10 | TTFT (s) | 14.10 | 44.38 | **0.32×** |
| Bonsai 2 2B PQ2_0 | 4096 | 1 | Decode tok/s | 64.60 | 32.65 | **1.98×** |
| Bonsai 2 2B PQ2_0 | 4096 | 1 | Prompt tok/s | 247.30 | 116.35 | **2.13×** |
| Bonsai 2 9B PQ2_0 | 512 | 1 | Decode tok/s | 25.47 | 10.92 | **2.33×** |
| Bonsai 2 9B PQ2_0 | 512 | 1 | Prompt tok/s | 71.65 | 28.45 | **2.52×** |
| Bonsai 2 9B PQ2_0 | 512 | 10 | Aggregate tok/s | 21.66 | 7.87 | **2.75×** |
| Bonsai 2 9B PQ2_0 | 512 | 10 | TTFT (s) | 52.12 | 181.96 | **0.29×** |
| Bonsai 2 9B PQ2_0 | 4096 | 1 | Decode tok/s | 22.69 | 10.12 | **2.24×** |
| Bonsai 2 9B PQ2_0 | 4096 | 1 | Prompt tok/s | 63.04 | 27.85 | **2.26×** |

With lossless multi-token prediction (MTP) on the 9B `-MTP-Q8_0` model (512 context, 1 user), Peregrine
decodes at **28.80 tok/s** (1.15× over its non-speculative 25.14 tok/s, and **3.09×** against the fork's
9.33 tok/s with `--spec-type draft-mtp`).

### Bluefin (4 cores, AVX2)

| Model | Context | Users | Metric | Peregrine | Fork best | Ratio |
|---|---|---|---|---|---|---|
| Bonsai 2 2B PQ2_0 | 512 | 1 | Decode tok/s | 46.40 | 33.63 | **1.38×** |
| Bonsai 2 2B PQ2_0 | 512 | 1 | Prompt tok/s | 118.67 | 56.95 | **2.08×** |
| Bonsai 2 2B PQ2_0 | 512 | 10 | Aggregate tok/s | 33.69 | 19.08 | **1.77×** |
| Bonsai 2 2B PQ2_0 | 512 | 10 | TTFT (s) | 32.82 | 82.05 | **0.40×** |
| Bonsai 2 2B PQ2_0 | 4096 | 1 | Decode tok/s | 42.28 | 30.79 | **1.37×** |
| Bonsai 2 2B PQ2_0 | 4096 | 1 | Prompt tok/s | 109.04 | 58.07 | **1.88×** |
| Bonsai 2 4B PQ2_0 | 512 | 1 | Decode tok/s | 24.59 | 14.95 | **1.64×** |
| Bonsai 2 4B PQ2_0 | 512 | 1 | Prompt tok/s | 48.05 | 18.55 | **2.59×** |
| Bonsai 2 4B PQ2_0 | 512 | 10 | Aggregate tok/s | 14.12 | 6.37 | **2.22×** |
| Bonsai 2 4B PQ2_0 | 512 | 10 | TTFT (s) | 78.84 | 279.73 | **0.28×** |

## ARM64 NEON and i8mm (DGX Spark)

Measured on the 20-core DGX Spark (10 × Cortex-X925 + 10 × Cortex-A725, `aarch64-neon-i8mm` and
`aarch64-neon-dotprod`) at a 512-token context:

| Model | Threads | Metric | Peregrine (tok/s) | Competitor (tok/s) | Ratio |
|---|---|---|---|---|---|
| Llama 3.2 1B Q4_K_M | 10 (X925) | Prompt tok/s | 573.8 | 234.2 (llama.cpp) | **2.45×** |
| Llama 3.2 1B Q4_K_M | 10 (X925) | Decode tok/s | 112.9 | 99.0 (llama.cpp) | **1.14×** |
| Llama 3.2 1B Q4_K_M | 20 (all) | Prompt tok/s | 629.8 | 322.0 (llama.cpp) | **1.96×** |
| Llama 3.2 1B Q4_K_M | 20 (all) | Decode tok/s | 111.6 | 86.4 (llama.cpp) | **1.29×** |
| Qwen 2.5 0.5B Q4_K_M | 10 (X925) | Prompt tok/s | 935.3 | 524.5 (llama.cpp) | **1.78×** |
| Qwen 2.5 0.5B Q4_K_M | 10 (X925) | Decode tok/s | 199.9 | 194.2 (llama.cpp) | **1.03×** |
| Qwen 2.5 0.5B Q4_K_M | 20 (all) | Prompt tok/s | 886.9 | 547.5 (llama.cpp) | **1.62×** |
| Qwen 2.5 0.5B Q4_K_M | 20 (all) | Decode tok/s | 198.8 | 128.3 (llama.cpp) | **1.55×** |
| Bonsai 2 2B PQ2_0 | 10 (X925) | Prompt tok/s | 404.0 | 167.7 (PrismML fork) | **2.41×** |
| Bonsai 2 2B PQ2_0 | 10 (X925) | Decode tok/s | 140.4 | 88.7 (PrismML fork) | **1.58×** |
| Bonsai 2 2B PQ2_0 | 20 (all) | Prompt tok/s | 433.1 | 200.5 (PrismML fork) | **2.16×** |
| Bonsai 2 2B PQ2_0 | 20 (all) | Decode tok/s | 146.8 | 64.5 (PrismML fork) | **2.28×** |
| Bonsai 2 4B-MTP PQ2_0 | 10 (X925) | Decode tok/s (MTP on) | 80.7 | 50.8 (PrismML fork) | **1.59×** |
| Bonsai 2 4B-MTP PQ2_0 | 20 (all) | Prompt tok/s | 185.2 | 80.9 (PrismML fork) | **2.29×** |
| Bonsai 2 4B-MTP PQ2_0 | 20 (all) | Decode tok/s | 67.7 | 32.5 (PrismML fork) | **2.08×** |

Full details: [`spark-arm-2026-11-09-i8mm`](../benchmarks/results/spark-arm-2026-11-09-i8mm/README.md)
and [`spark-arm-2026-11-10-bonsai2`](../benchmarks/results/spark-arm-2026-11-10-bonsai2/README.md).

## Embeddings, rerank and decisions

Embedding and decision models are compared with every engine that can run the model on the host, not
only llama.cpp. A ratio is Peregrine ÷ the fastest other engine **at equal accuracy** for that cell:
engines whose vectors drift from the full-precision reference (the int8 ONNX exports, at cosine
0.92–0.98) are shown in the raw tables and left out of the ratios. The embedding result cache is off
for every timed row, so every engine computes every text.

### Bluefin (4 cores, AVX2): every engine

4 threads, three interleaved rounds, medians. Cells are 16, 128 and 512
tokens at batch 1, 8 and 32, a mixed-length batch of 32 texts (8–512 tokens), and 2 000 tokens for the
models that accept them. Engines: llama.cpp v0.9.0, OpenVINO 2026.4, ONNX Runtime 1.31, fastembed 0.9
and text-embeddings-inference 1.9. The four larger families are the S4.9l proof run at `2a5e9f6`; the

two small BERT rows are re-measured at `4500a6f` plus S4.9p and S4.9r
([`bluefin-2026-11-12-s4.9r`](../benchmarks/results/bluefin-2026-11-12-s4.9r/README.md)).

| Model | File | Fastest other engine | 16 tokens | 128 tokens | 512 tokens | 2 000 tokens | mixed batch |
|---|---|---|---|---|---|---|---|
| all-MiniLM-L6-v2 | F16, Q8_0 | OpenVINO, ONNX Runtime, llama.cpp | **1.01–1.46×** | 0.97–1.08× | 0.99–1.03× | n/a | **1.60–1.65×** |
| bge-small-en-v1.5 | F32, Q8_0 | OpenVINO, ONNX Runtime, fastembed | 0.95–1.23× | 0.93–1.01× | 0.90–0.95× | n/a | **1.31×** |

| bge-m3 | Q8_0 | llama.cpp | **1.18–1.53× across every cell** | | | **1.35×** | |
| EmbeddingGemma 300M | Q8_0 | llama.cpp | **1.30–1.51× across every cell** | | | | |
| Qwen3-Embedding 0.6B | Q8_0, F16 | llama.cpp | **1.25–1.51×** (Q8_0), **1.24–1.44×** (F16) | | | | |
| Nemotron 3.5 Embed 1B | Q8_0 | llama.cpp | **1.34–1.62× across every cell** | | | | |


The four larger families are ahead in every cell (1.05× and up). The two small BERT encoders are
now level on MiniLM at 512 tokens (0.99–1.03×) and still behind on bge-small (0.90–0.95× at 512,
0.93–0.95× at 128 b1/b8), plus MiniLM's 128 b8 (0.97×) and the 16 b32 cells (0.95–0.98×). S4.9p
showed the matrix kernels are level with oneDNN's on this CPU (165–178 against 173 GMAC/s) and put
the remainder in the element-wise block — 9.3 ms of a 512-token MiniLM pass against OpenVINO's
2.4 ms and 17.8 against 3.8 ms on bge-small. S4.9r took that block apart step by step: threaded the
q/k/v biases and the two residual reads (they ran on one thread), ran four softmax denominators at
once instead of one serial `f64` chain, and widened F16 embedding rows with `vcvtph2ps` — all with
every vector bit-identical. Element-wise is now **7.5 ms** and **16.0 ms**, attention 10.8 and 21.4
(from 12.7 and 25.5), and MiniLM's 512-token cells reached level (0.82→0.99×, 0.91→1.00×, 0.93→1.01×). What still holds bge-small back is now one step: its exact-`erf` GELU, 10.7 ms of a
98.8 ms pass, against OpenVINO's fused approximation, which costs 0 ms as a node because oneDNN
fuses it into the GEMM epilogue; and its 25 LayerNorms against the `MVN` nodes' 0.8 ms. The

mixed-length batch is a win on both models because Peregrine never pads: the padding engines are
1.6–4.5× slower there.

Accuracy on the same texts: Q8_0 agrees with llama.cpp's Q8_0 within 3e-4 cosine and is closer to the
fp32 reference on all three BERT models; the F16 and F32 rows are exact against the fp32 engines;
Nemotron's F16 vectors are at cosine 0.9999998 to llama.cpp's.

**Decisions** (`POST /v1/systemone`, `tinyopenjev` Q8_0; llama.cpp with one slot per question,
`-np 11 -c 16384`), questions per second:

| State tokens | Questions | Peregrine | llama.cpp | Ratio |
|---:|---:|---:|---:|---:|
| 64 | 1 | 179.2 | 97.5 | **1.84×** |
| 64 | 3 | 243.9 | 161.3 | **1.51×** |
| 64 | 10 | 310.1 | 203.2 | **1.53×** |
| 300 | 1 | 64.7 | 46.4 | **1.39×** |
| 300 | 3 | 116.4 | 86.5 | **1.35×** |
| 300 | 10 | 198.8 | 133.0 | **1.50×** |
| 1000 | 1 | 20.7 | 15.6 | **1.33×** |
| 1000 | 3 | 48.8 | 35.7 | **1.37×** |
| 1000 | 10 | 95.3 | 67.6 | **1.41×** |

Full details, per-cell tables, stage profiles and the excluded rows:
[`bluefin-2026-11-12-s4.9l-proof`](../benchmarks/results/bluefin-2026-11-12-s4.9l-proof/README.md).

### DGX Spark (ARM64): against llama.cpp

10 threads (the Cortex-X925 cluster), `aarch64-neon-i8mm`, llama.cpp v0.9.0 with one slot per text.
The embedding result cache is off on every timed row (`--opt EMBED_CACHE_MB=0`), so every engine
computes every text; three interleaved rounds, medians.

| Model | File | One text (16 to 2 000 tokens; BERT to 512) | Batch of 32 × 128 | Mixed batch of 32 |
|---|---|---|---|---|
| EmbeddingGemma 300M | Q8_0 | **2.02–2.13×** | **1.99×** | **1.96×** |
| Qwen3-Embedding 0.6B | Q8_0 | **1.38–3.20×** | **1.99×** | **1.80×** |
| Qwen3-Embedding 0.6B | F16 | 0.96–1.57×† | **1.17×** | **1.10×** |
| Nemotron 3.5 Embed 1B | Q8_0 | **2.40–2.72×** | **2.70×** | **2.62×** |
| bge-m3 | Q8_0 | **1.99–2.19×** | **2.08×** | **2.07×** |
| gte-modernbert-base | F16 | **1.34–1.59×** | **1.33×** | **1.37×** |
| gte-modernbert-base | Q8_0 | **1.66–1.78×** | **1.66×** | **1.75×** |
| all-MiniLM-L6-v2 | F16 | **1.49–1.65×** (16 to 512 tokens) | **1.33×** | **1.36×** |
| all-MiniLM-L6-v2 | Q8_0 | **1.58–1.90×** (16 to 512 tokens) | **1.63×** | **1.66×** |

The batch cells are still wins because llama.cpp runs one embedding task at a time inside a
request while Peregrine shares weight passes across the texts, but the cache-off numbers are much
smaller than the ones this table first published: the old run reused one text list inside a timed
cell with the exact result cache on, so a timed wave after the first was a lookup. Qwen3-Embedding
0.6B F16's batch of 32 read 2.72–3.21× that way and reads **1.17×** with every text computed. One
cell is now below 1.0×: Qwen3-Embedding 0.6B F16 at a single 2 000-token text, **0.96×** (its
512-token cell is 1.06× and its 128-token cell 1.14×). The F16 matrix pass and llama.cpp's CPU
flash attention meet at long context; the Q8_0 file of the same model is 1.38–3.20× ahead. The
narrowest wins above 1.0 are MiniLM F16 batch and gte F16 batch at 1.33× and the rerank
32-document cell at 1.35×. S4.9q's `i8mm` body for the Q8_0 int8 tile (`smmla` instead of `sdot`,
3.05× the kernel) and S4.9o's load-time widening of small Q8_0 ModernBERT files still stand; the
Q8_0 rows here are measured on the same main as those changes.

† **S4.9s** is the step aimed at that cell. Its account put 35 % of a 2 000-token pass in
attention and 24 % in the value mix alone, at 12.8 GMAC/s — 19 % of this core's f32 roof —
because the mix read V once per query row where llama.cpp's flash-attention tile reads it once
per 64 rows. The mix now shares each loaded V half-vector across two rows and keeps both rows'
outputs in registers: same bits (every length and width held to the scalar twin; no golden
changed), the mix 8.93 → 5.39 s and the one-thread pass 36.76 → 32.97 s (−10.3 %). In the paired
window of that step the 2 000-token cell moved 0.67× → 0.75× against llama.cpp; on the quiet
host that −10.3 % puts it at about **1.05–1.07×**, but the cell has not been re-read on a quiet
host, so the table keeps 0.96× until it is. The Q8_0 file of the same model moves 0.82× → 0.93×
on the same cell; `gte-modernbert-base` F16 is an encoder and does not call this kernel.
`benchmarks/results/spark-arm-2026-11-12-s4.9s/`.

Full details: [`spark-arm-2026-11-12-embeddings-cache-off`](../benchmarks/results/spark-arm-2026-11-12-embeddings-cache-off/README.md)
(every row of this table, cache off, with the per-cell spreads),
[`spark-arm-2026-11-12-q8-i8mm`](../benchmarks/results/spark-arm-2026-11-12-q8-i8mm/README.md)
(the Q8_0 int8-tile rows, before/after the `i8mm` body),
[`spark-2026-11-12-embeddings`](../benchmarks/results/spark-2026-11-12-embeddings/README.md),
[`spark-arm-2026-11-12-modern-bert`](../benchmarks/results/spark-arm-2026-11-12-modern-bert/README.md)
(the F16 row) and [`spark-arm-2026-11-12-q8-encoders`](../benchmarks/results/spark-arm-2026-11-12-q8-encoders/README.md)
(the Q8_0 ModernBERT row). The three older runs timed their batch cells with the result cache on;
their single-text cells remain comparable.

### Rerank

`POST /v1/rerank` on bge-reranker-v2-m3 and bge-reranker-base gives the same token ids and the same
ranking order as llama.cpp on the reference pairs (320 pairs, eight languages:
[reference/bert.md](reference/bert.md)).

Bluefin, 4 threads, one query against 1, 8 and 32 documents of 123 and 449 tokens, both engines on the
same GGUF, one engine at a time through HTTP, median of five requests (`scripts/rerank-bench.py`).
llama.cpp got one slot per document of the largest request (`-np 32 -c 16640 -b 1024 -ub 1024
--reranking`); Peregrine ran `serve --threads 4 --ctx 16640`, which clamps to the file's 8 192 and runs
32 rows per pass. Pairs per second:

| File | tokens/pair | documents | Peregrine | llama.cpp | Ratio |
|---|---|---:|---:|---:|---:|
| bge-reranker-v2-m3 Q8_0 | 123 | 1 / 8 / 32 | 3.87 / 3.91 / 3.92 | 3.31 / 2.77 / 2.76 | **1.17× / 1.41× / 1.42×** |
| bge-reranker-v2-m3 Q8_0 | 449 | 1 / 8 / 32 | 0.96 / 0.96 / 0.96 | 0.78 / 0.73 / 0.73 | **1.23× / 1.32× / 1.32×** |
| bge-reranker-base Q8_0 | 123 | 1 / 8 / 32 | 12.37 / 13.11 / 13.14 | 11.10 / 9.66 / 9.68 | **1.11× / 1.36× / 1.36×** |
| bge-reranker-base Q8_0 | 449 | 1 / 8 / 32 | 3.25 / 3.26 / 3.25 | 2.76 / 2.52 / 2.52 | **1.18× / 1.29× / 1.29×** |

The top-1 document agrees in every cell. bge-reranker-base's whole ranking is identical (Kendall
τ = 1.000, largest score difference 0.10); v2-m3 swaps one adjacent pair at each length, and those
two documents are 0.013–0.018 apart in llama.cpp's own scores — smaller than the engines' score
difference on the same document (0.04–0.11), so the swap is the near-tie. Method, flags and the full
data: [`bluefin-2026-11-12-s4.9p`](../benchmarks/results/bluefin-2026-11-12-s4.9p/README.md).

On the ARM host, bge-reranker-v2-m3 Q8_0 with one query and documents of 124–128 tokens, five
interleaved rounds, `pairs/s` (Peregrine ÷ llama.cpp, above 1.00 is Peregrine ahead):

| Documents | Peregrine pairs/s | llama.cpp pairs/s | Ratio |
|---:|---:|---:|---:|
| 8 | 7.41 | 5.39 | **1.38×** |
| 32 | 7.19 | 5.31 | **1.35×** |

The table above is bluefin (AVX2, 4 threads); on the ARM host Peregrine is 35–38%
ahead at 124–128-token pairs too. Full details:
[`spark-arm-2026-11-12-embeddings-cache-off`](../benchmarks/results/spark-arm-2026-11-12-embeddings-cache-off/README.md).

## Task accuracy and decoding invariance

Speed numbers are paired with the task-accuracy gate and decoding-invariance suite in
[`docs/dev/accuracy.md`](dev/accuracy.md) and [`benchmarks/accuracy/`](../benchmarks/accuracy/):

- **Task accuracy (GSM8K 250, MMLU 500, HumanEval+ 163):** all four gate models
  (Qwen 2.5 0.5B Instruct, Llama 3.2 1B Instruct, gemma-3-1b-it, Qwen 3 1.7B, Q4_K_M) pass every
  paired check against same-host llama.cpp (`|Δ| ≤ 4` percentage points, exact McNemar `p ≥ 0.12`) and
  the BF16 vLLM floor across all three hardware tiers — AVX2, AVX-512, and ARM64 NEON i8mm.
- **Batch and speculation invariance:** 520 prompts decoded alone, in concurrent batches of 8, and
  with `--speculate` produce **520/520 byte-identical replies** on every host and tier.

## Reproducing

```sh
python3 benchmarks/run.py --help     # run the benchmark harness on a model and engine
python3 benchmarks/charts.py         # regenerate the SVG charts in docs/img/
```
