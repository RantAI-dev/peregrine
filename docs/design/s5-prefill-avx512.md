# S5: why ik_llama.cpp processes prompts faster on AVX-512, and what to build

- Status: Study (research sub-agent), no engine change
- Date: 2026-11-11
- Host: AMD Ryzen AI 9 HX 370 (Zen 5, 4 full + 8 compact cores, 12 cores / 24 threads,
  AVX-512 F/BW/VL/VNNI/VBMI/BF16), WSL2, 11.7 GiB RAM, Windows 11 laptop
- Peregrine: `origin/main` `267ac22`, `cargo build --release -p peregrine`
- ik_llama.cpp: `master` `479074ccac2ac94f06b6d24a9ce26f0f9e9aa6ae` (2026-11-11 08:54:37 +0200),
  CMake Release with `-DGGML_NATIVE=ON -DGGML_AVX512=ON -DGGML_AVX512_VBMI=ON
  -DGGML_AVX512_VNNI=ON -DGGML_AVX512_BF16=ON`. The build prints "Using optimized iqk matrix
  multiplications" and "Enabling IQK Flash Attention kernels"; `objdump -d` of its
  `libggml.so` finds 26,526 `vpdpbusd`.
- Raw measurements: `benchmarks/results/desktop-zen5-2026-11-11-prefill-study/` (the README
  there names the command behind every file).
- Every claim about either code base below is read from the commits named above, with
  `file:line`. No code was copied from ik_llama.cpp (MIT); it is described only.

---

## 0. Summary

On this machine a prompt of 512 tokens is **2.4× slower on Peregrine than on ik_llama.cpp for
Qwen 2.5 0.5B and 1.9× slower for Llama 3.2 1B** (measured). The gap is not one thing; it is
two, and they multiply:

1. **The k-quant matrix-multiply kernel moves 1.7–3.2× fewer multiply-accumulates per second
   per core**, depending on the weight format (measured, one core, the same weights, at the
   widths the engines use). That is the whole gap at one thread.
2. **Peregrine's thread scaling is worse**: 6.0× going from 1 to 12 threads on the 1B against
   ik's 6.7× (measured). That adds the remaining 1.1× at 12 threads.

Pass size and non-matmul time are second order: Peregrine's pass-size default is already at
its own optimum for the 1B (a larger chunk is *slower*), and its attention, quantise and other
buckets are 19–27 % of a step, not the 50 % the factor of two would need.

The single most useful number in the study: the machine's byte-dot roof is
**455 GMAC/s per core** (`vpdpbusd zmm`, measured, 453–463 across cores), Peregrine's Q4_K
kernel reaches **76–79** (17 %), and ik's Q4_K kernel reaches **134–148 GMAC/s**. ik's number is
60–66 % of the 256-bit roof it uses (227 GMAC/s), because it runs the repacked Q4_K kernel with
`__m256i` even on an AVX-512 build (`ggml/src/iqk/iqk_gemm_kquants.cpp:3169`). So the gap is
not "VNNI versus no VNNI": both use `vpdpbusd` into int32 with an exact per-sub-block scale.
The gap is how many of those instructions each loop keeps in flight.

---

## 1. Measurements

### 1.1 Method

- `peregrine bench --prompt-tokens 512 --tokens 1 --threads N --no-profile --json`, and
  `PEREGRINE_PROF2=full` for the phase table (with `--tokens 0` so the printed wall equals the
  engine's own `prefill_s`).
- `llama-bench -m <same file> -p 512 -n 0 -t N -tb N -r 5 -o json`, `-rtr 1 -fa 1` unless a row
  says otherwise. `llama-bench`'s `avg_ts` is a *mean* and WSL2's clock jumps make single
  samples meaningless (one sample can read a negative time), so every row is the **median of
  its `samples_ts`**.
- Kernel tables: `cargo run --release -p peregrine-bench --example kqbench -- 16 200`
  (`taskset -c 2`) and a new `ik_kqbench` that links against the ik_llama.cpp build and calls
  its own `ggml_mul_mat` after `iqk_repack_tensor` (`ik_kqbench.cpp`, in the results folder).
- Roofs: `peak4.c`, `peak6.c`, `peak9.c` in the results folder — 16 independent accumulators
  and constant operands, so nothing is hoisted and no accumulator chain limits the rate.
- Model files: Qwen 2.5 0.5B Instruct Q4_K_M `74a4da8c…` (491,400,032 B) and Llama 3.2 1B
  Instruct Q4_K_M `6f85a640…` (807,694,464 B) — the same two files as
  `benchmarks/results/desktop-zen5-2026-11-11-ik/README.md`.
- The host was not otherwise idle: the agent's own UI kept a load average of 1–8 on some
  series. Every table row is a median of 3 (or 5) interleaved rounds, and the same-day
  Peregrine/ik ratio is the quantity to trust, not the absolute tok/s.

### 1.2 The machine's roof (measured, one core)

| probe | result | derived |
|---|---|---|
| `vpdpbusd zmm`, 16 accumulators | 453–463 GMAC/s | **455 GMAC/s** = 7.1 G instructions/s × 64 byte-MACs |
| `vpdpbusd ymm`, 16 accumulators | 223–227 GMAC/s | 7.1 G instructions/s × 32 byte-MACs |
| `vfmadd512`, 16 accumulators | 7.08–7.23 G instructions/s | 2 instructions/cycle at **3.55 GHz** |
| `add` latency chain | 1.78 GHz | rejected: it read 2 cycles per `add`; two independent probes agree on 3.5 GHz |

So the core issues **two 512-bit vector instructions per cycle** at ~3.5 GHz, and one
`vpdpbusd zmm` is 64 byte-MACs. The roof for the instruction the k-quant kernels use is
**455 GMAC/s per core**, and the same instruction rate on 256-bit registers is 227 GMAC/s.

### 1.3 Engine level, 512 tokens, 12 threads

Median of 3 interleaved rounds (ik: median of 5 samples in one invocation).

| model | Peregrine | ik_llama.cpp | P ÷ ik |
|---|---|---|---|
| Qwen 2.5 0.5B Q4_K_M | **484.2 tok/s** | **1158.3 tok/s** | 0.42× |
| Llama 3.2 1B Q4_K_M | **288.6 tok/s** | **549.9 tok/s** | 0.52× |

This reproduces the published rows in `desktop-zen5-2026-11-11-ik/README.md` (471 and 274 for
Peregrine, 1117 and 527 for ik) from the other harness, which is the check that the two
measurements are of the same thing.

### 1.4 Peregrine's prefill, by phase (measured)

`PEREGRINE_PROF2=full`, 12 threads, chunk 16, medians of 3 runs. `wall` is the engine's own
`prefill_s / chunks`; the listed buckets account for 90 % (Qwen) and 94 % (1B) of it, the rest
being the forward's untimed tail.

**Qwen 2.5 0.5B Q4_K_M — 475.8 tok/s, 33.6 ms per 16-token pass**

| bucket | µs/pass | share |
|---|---|---|
| matmul gate/up | 11 939.7 | 35.6 % |
| matmul down | 7 570.6 | 22.6 % |
| matmul qkv | 2 380.1 | 7.1 % |
| matmul wo | 2 397.3 | 7.1 % |
| **matmul total** | **24 287.7** | **72.4 %** |
| attention | 3 418.5 | 10.2 % |
| quantise (activations) | 724.5 | 2.2 % |
| residual | 902.3 | 2.7 % |
| rope + bias | 431.3 | 1.3 % |
| KV write | 439.8 | 1.3 % |
| norms | 84.9 | 0.3 % |
| barrier wait (inside the matmul wall) | 7 805.7 | 23.3 % |

by weight format, share of the pass: Q5_0 14 356.7 µs (42.8 %), Q6_K 3 234.9 (9.6 %), Q4_K 2 924.8 (8.7 %),
Q8_0 92.8 (0.3 %). The Qwen file is mostly Q5_0: `peregrine inspect` lists 133 Q5_0 tensors
(254.4 MiB), 13 Q8_0, 12 Q6_K, 12 Q4_K.

**Llama 3.2 1B Q4_K_M — 286.6 tok/s, 55.8 ms per pass**

| bucket | µs/pass | share |
|---|---|---|
| matmul gate/up | 20 900.7 | 37.5 % |
| matmul down | 15 246.6 | 27.4 % |
| matmul qkv | 4 853.0 | 8.7 % |
| matmul wo | 4 062.8 | 7.3 % |
| **matmul total** | **45 063.1** | **80.9 %** |
| attention | 4 197.3 | 7.5 % |
| quantise | 958.1 | 1.7 % |
| KV write | 848.4 | 1.5 % |
| residual | 653.9 | 1.2 % |
| rope + bias | 440.6 | 0.8 % |
| norms | 190.2 | 0.3 % |
| barrier wait | 7 860.6 | 14.1 % |

by format, share of the pass: Q4_K 31 264.8 µs (56.1 %), Q6_K 8 566.5 (15.4 %). `peregrine inspect` on this file:
96 Q4_K tensors (445.5 MiB), 17 Q6_K (317.1 MiB — the output head is most of it, and the head
is not computed during a prompt pass).

With `PEREGRINE_KREPACK=0` the same table reads: Qwen 83.0 % matmul at 314.8 tok/s, 1B 87.6 %
matmul at 196.0 tok/s.

### 1.5 Peregrine's switches (measured, 12 threads, 5 interleaved rounds)

| switch | Qwen 0.5B | Llama 1B |
|---|---|---|
| `PEREGRINE_KREPACK` default (on) | 485.7 tok/s | 288.8 tok/s |
| `PEREGRINE_KREPACK=1` | 477.0 | 294.4 |
| `PEREGRINE_KREPACK=0` | 310.3 | 195.9 |
| `PEREGRINE_ISA=native` (ZMM kernel) | 470.5 | 284.2 |
| `PEREGRINE_ISA=avx2` (AVX2 kernel) | 404.9 (−16 %) | 241.2 (−18 %) |

`peregrine tier` on this host: `kernel tier x86_64-avx2-avx512-vnni-gfni-vbmi`, `kq_repack on
(tier default)`, `zmm=true`. `peregrine support` lists `Q4_K scalar, avx2, avx512`; `Q5_K` and
`Q6_K` **scalar, avx2** — there is no 512-bit kernel for either. The two commands say the same
thing as the source: `crates/peregrine-kernels/src/table.rs:60-95` sets `kq_tiles` from
`have_batched() && options::get().kq_repack` and `kq_zmm` from `kq_tiles && have_zmm512()`,
and `table.rs` dispatches Q6_K to the AVX2 tile kernel.

History, from `docs/design/s3.4.md` §10: the repacked tile was on for AVX2 and **off** for the
AVX-512 tier on 2026-11-07 (the AVX2 tile cost 3.5 % decode there), then turned **on** for
AVX-512 on 2026-11-08 at the user's decision, and the 512-bit kernel landed on 2026-11-09. The
numbers above are what that turned into: the repack is worth +57 % (Qwen) / +47 % (1B) on
prompt, and the 512-bit kernel another +16 % / +18 %.

### 1.6 Kernel level, one core, MACs per second (measured)

`kqbench`, one tile of eight rows held in L1 (n_sb = 16), best of five timed rounds; row = the
kernel the engine runs with `PEREGRINE_KREPACK=0`, tile = the repacked kernel.

| kernel (Peregrine) | 1 act | 16 acts | 64 acts | 128 acts |
|---|---|---|---|---|
| Q4_K row (`dot_batch` → `wide_q4_k`) | 22.5 | 35.4 | 32.5 | 23.9 |
| Q4_K tile, AVX2 | 31.1 | 57.6 | 57.6 | 57.1 |
| **Q4_K tile, ZMM** | **43.0** | **79.2** | **77.1** | **75.8** |
| Q6_K row | 21.0 | 38.0 | 36.3 | 32.8 |
| **Q6_K tile, AVX2 (no ZMM kernel exists)** | **21.0** | **39.2** | **38.5** | **37.0** |
| Q5_K row | 17.4 | 36.8 | 34.6 | 32.2 |
| Q5_K tile, AVX2 | 17.7 | 46.6 | 43.7 | 42.3 |
| Q5_0 tile, AVX2 | — | 37.6 (4 acts) | 36.0 | — |
| Q5_0 tile, ZMM | — | 50.7 (4 acts) | 48.7 | — |

`ik_kqbench`, the same question asked of ik_llama.cpp's own `ggml_mul_mat` on a 2048×4096
matrix (4.7 MB, so the weights stream from L3 rather than L1 — this flatters Peregrine, not ik):

| kernel (ik_llama.cpp) | 1 act | 16 acts | 64 acts | 128 acts |
|---|---|---|---|---|
| Q4_K_R4 (`-rtr` on), ymm | 73.2 | **134.0** | 113.6 | **148.1** |
| Q4_K (no repack), ymm | 44.3 | 65.8 | 137.3 | 142.7 |
| Q6_K_R4 (`-rtr` on), ymm | 58.8 | **123.1** | 120.4 | **152.8** |
| Q6_K (no repack) | 33.0 | 58.4 | 93.3 | 114.5 |
| Q5_0_R4 (`-rtr` on) | 47.3 | 108.0 | 167.3 | 170.2 |
| Q8_0_R8 (`-rtr` on) | 89.3 | 165.0 | 168.1 | 170.5 |

Ratios at the widths the engines actually use. Peregrine's chunk is 16 activation rows; ik's
ub is 512, but its kernel walks activation rows in blocks of at most 8, so the closest column
is 64–128 (the two differ by under 10 %). Taking 16 and 128: **Q4_K 1.7–1.9×**, **Q6_K 3.1×**,
**Q5_0 2.1–3.5×**.

Against the roof: Peregrine's Q4_K ZMM kernel runs at **17–18 % of 455 GMAC/s**; ik's ymm
kernel at **60–66 % of 227 GMAC/s**. Peregrine's Q6_K (9 % of 455) is the worst placed of all.

### 1.7 Threads (measured)

Peregrine at `PEREGRINE_PREFILL_CHUNK=16`; ik at `-ub 512`. Prompt tok/s.

| threads | Qwen P | Qwen ik | 1B P | 1B ik |
|---|---|---|---|---|
| 1 | 93.6 | 183.9 | 47.6 | 82.0 |
| 2 | 155.2 | 319.7 | 82.7 | 145.4 |
| 4 | 265.5 | 495.9 | 138.0 | 232.0 |
| 8 | 415.9 | 810.0 | 230.0 | 383.9 |
| 12 | 482.5 | 1158.3 | 286.1 | 549.9 |
| 24 | 285.9 | 1623.3 | 277.4 | 832.1 |

Speed-up from 1 thread: Qwen P 5.2× / ik 6.3×; 1B P 6.0× / ik 6.7×. At 24 threads ik keeps
going (8.8× / 10.1×) while Peregrine falls over: its chunk default at 24 threads is 64
(`options.rs:80-92`), which lifts the 24-thread rows to 451.1 (Qwen) and 304.9 (1B) — still
6.4× for the 1B, i.e. no gain from SMT.

### 1.8 Pass size (measured, 12 threads and 1 thread)

Peregrine's pass size is `PEREGRINE_PREFILL_CHUNK` (default 16 up to 12 threads, 64 above
them; `crates/peregrine-core/src/options.rs:60-98`); ik's is `-ub`.

| chunk / ub | Qwen P | Qwen ik | 1B P | 1B ik |
|---|---|---|---|---|
| 64 | 559.5 | 917.2 | 276.7 | 347.7 |
| 128 | 557.2 | 1065.1 | 264.1 | 450.3 |
| 256 | — (max 128) | 1159.9 | — | 524.1 |
| 512 | — | 1174.3 | — | 550.9 |
| 16 | 484.2 | — | 288.6 | — |
| 32 | 525.7 | — | 283.8 | — |

`PEREGRINE_PREFILL_CHUNK` is capped at 128 (`MAX_PREFILL_CHUNK = MAX_BATCH`,
`crates/peregrine-core/src/options.rs:95-98`), so 16-to-512 in ik's units means 16-to-128 here
plus the 24-thread pass budget.

One thread, same sweep: Qwen 94.0 / 94.3 / 90.3 / 89.9 and 1B **47.5 / 43.2 / 36.1 / 33.0**
for chunk 16 / 32 / 64 / 128. A wider pass makes Peregrine's 1B prefill *worse*, at one thread
and at twelve. ik gains +28 % (Qwen) and +58 % (1B) from 64 to 512. **This is a real
inefficiency in Peregrine's large-batch path, not a tuning preference** — see §4.

### 1.9 Cores (measured, `taskset`, chunk 16)

CPUs 0–7 are cores 0–3, CPUs 8–23 are cores 4–11; every core has two SMT threads. Pinning
threads by hand:

| pin | Qwen | 1B |
|---|---|---|
| 4 threads on cores 0–3 | 244.5 | 135.2 |
| 4 threads on cores 4–7 | 269.7 | 140.5 |
| 4 threads on cores 8–11 | 264.3 | 139.6 |
| 8 threads on cores 0–3 (SMT) | 404.3 | 232.3 |
| 8 threads on cores 4–11 | 413.5 | 230.8 |
| 12 threads, free | 493.0 | 294.0 |

**The "fast" and "compact" cores are indistinguishable on this host for this workload** (4
threads on cores 0–3 are, if anything, slower than on the compact cores — probably because the
agent's UI runs there). Two SMT threads on one core do as much as two separate compact cores
(404 vs 413), which says the per-core machine is latency-bound, not issue-bound, before the
thread count is even high. Pinning is not a lever here.

### 1.10 ik_llama.cpp's switches (measured, 12 threads, ub 512)

| setting | Qwen | 1B |
|---|---|---|
| `-rtr 0 -fa 0` | 951.4 | 400.9 |
| `-rtr 0 -fa 1` | 1072.5 | 427.9 |
| `-rtr 1 -fa 0` | 1031.2 | 511.5 |
| **`-rtr 1 -fa 1`** (its defaults) | **1166.2** | **553.7** |

So `-rtr` is +9 % (Qwen) and +29 % (1B), `-fa` is +13 % / +7 %, and both together are +23 % /
+38 % over the plain path. The same Qwen file in the 2026-11-11 ik README measured 967.5 →
1056.4 tok/s for `-rtr`, also +9 %, so the setting is not cherry-picked. (The 1166.2 here and
the 1158.3 in §1.3 are the same setting measured in two invocations, 0.7 % apart.)

---

## 2. How ik_llama.cpp does it

### 2.1 What `-rtr` changes: the layout, in place

`-rtr` calls `iqk_repack_tensor` for every host tensor at load (`src/llama.cpp:5620-5649`).
That function (`ggml/src/iqk/iqk_quantize.cpp:9763-9810`) looks the tensor's type up in
`get_repack_info` (`iqk_quantize.cpp:9714-9753`) and rewrites the bytes **in place, at the same
size**, then changes `tensor->type` to the `_R4`/`_R8` variant. For the formats that matter
here: `Q4_K → Q4_K_R4` (4 output rows per block), `Q5_K → Q5_K_R4`, `Q6_K → Q6_K_R4`,
`Q5_0 → Q5_0_R4`, `Q8_0 → Q8_0_R8`. The four bits of every weight are the file's; no
re-quantisation, no growth. That is why `-rtr` costs only load time and not accuracy.

`block_q4_k_r4` (`ggml/src/ggml-common.h:392-398`) is 576 bytes = four 144-byte Q4_K blocks:
`d[8]` (each row's super-block scale and min scale), `scales_h[16]`, `scales_l[32]` (the same
6-bit packed scales and mins, four rows interleaved) and `qs[512]` — the four rows' nibbles
**interleaved in eight-byte windows**: bytes `8w..8w+8` of each of the four rows, side by side.
`block_q6_k_r4` (`ggml-common.h:433-439`) is the same idea for Q6_K: `d[4]`, `scales[64]`,
`qh[256]`, `ql[512]`.

### 2.2 What the activation becomes

`Q4_K_R4`'s `vec_dot_type` is **`GGML_TYPE_Q8_K32`** (`ggml/src/ggml.c:963-975`), which is a
plain `block_q8_K` (`ggml-common.h:441-447`: one f32 scale, 256 int8 codes, 16 int16 run sums
of 16 codes). It is filled once per matrix product, for every activation row, by
`iqk_quantize_any` inside `ggml_compute_forward_mul_mat` (`ggml/src/ggml.c:18852-18876`),
before the product starts. So ik's activation is **int8, one row at a time, per 256 weights**,
with the scales kept per row — the same quantisation llama.cpp uses, and the same kind
Peregrine uses. There is no four-row interleaved activation buffer in this path.

### 2.3 The kernel: `mul_mat_q4_k_r4_q8_k`

`ggml/src/iqk/iqk_gemm_kquants.cpp:1497-1562`, selected for `Q4_K_R4` at
`iqk_gemm_kquants.cpp:3219-3221`. It is templated on `nrc_y` = the number of **activation rows**
it handles at once, and instantiated for 1..8 (`IQK_MAX_NY`, `iqk_common.h:144`;
`IQK_SET_MUL_MAT_FUNCTIONS`, `iqk_common.h:166-175`). One call:

```
for ix in 0..nrc_x step 4            # 4 output rows per block_q4_k_r4
  for ibl in 0..n/256                # super-blocks
    decode the 4 rows' d, dmin, scales (256-bit unpack, 6-bit bit tricks)
    process_min_r4_b32(...)          # the min term for the whole super-block, once
    for ib in 0..8                   # 8 sub-blocks of 32 weights
      two 32-byte weight loads -> four nibble vectors (4 x 32 bytes)
      for iy in 0..nrc_y             # activation rows
        load 32 bytes of that row's q8_K
        sumi  = dpbusd(sumi, qx[0], shuffle(y,0x00))   # four vpdpbusd,
        sumi  = dpbusd(sumi, qx[1], shuffle(y,0x55))   # 32 byte-MACs each,
        sumi  = dpbusd(sumi, qx[2], shuffle(y,0xaa))   # 128 MACs in all
        sumi  = dpbusd(sumi, qx[3], shuffle(y,0xff))
        isum += mullo(scales_d, sumi)                  # exact int32 scale, per sub-block
    acc[iy] += fma(d4 * act_scale, cvtepi32_ps(isum))  # one float fold per super-block
```

Line by line: the four `vpdpbusd` are at `iqk_gemm_kquants.cpp:1537-1540` (`HAVE_VNNI256`
branch), the int32 scale multiply at `:1541`, the float fold at `:1552`, the activation load at
`:1534`, the min term at `:1464-1496` and its call at `:1518`.

Facts that matter for speed:

- **Instruction set**: `_mm256_dpbusd_epi32` — 256-bit VNNI, 32 byte-MACs per instruction
  (`iqk_config.h:52-58`, `HAVE_VNNI256` defined because `__AVX512VNNI__` and `__AVX512VL__`
  are). Even on an AVX-512 build this kernel is `__m256i`. It reaches 4.6 G of those per
  second, i.e. 1.3 per cycle at 3.5 GHz, against a roof of 2 (measured, `peak9.c`).
- **Rows per call**: 4 weight rows (the `_R4` block) × up to 8 activation rows (the template),
  × 256 weights per super-block. Four `vpdpbusd` and one `vpmulld` + `vpaddd` per
  (sub-block, activation row).
- **Activation format**: int8 `q8_K`, read once per (sub-block, activation row) as one 32-byte
  load and broadcast four times with `vpshufd` (`iqk_gemm_kquants.cpp:1534-1540`); the weight bytes
  are loaded once per (sub-block, 4 rows) and reused across all `nrc_y` activation rows.
- **Accumulators per load**: the weight load's 32 bytes feed four `vpdpbusd` (2 nibble halves ×
  2 halves of the sub-block) and are then reused by the other `nrc_y - 1` rows from registers.
- **Scale**: applied as an `vpmulld` on int32 partials, one per (sub-block, activation row) —
  exact, and it runs on a different execution unit from `vpdpbusd` (`s3.4.md` §12 measured the
  same on this host).
- `mul_mat_q6_k_r4_q8_k` (`iqk_gemm_kquants.cpp:1638-1775`) is the same shape for Q6_K, with
  the 6-bit codes split from `ql`/`qh` with a `vpshufb` shuffle (`:1642`), and reaches 123–153
  GMAC/s.

### 2.4 How threads and tiles split the work

`iqk_mul_mat` (`ggml/src/iqk/iqk_mul_mat.cpp`) is the entry point. For a prompt-sized product
(`Ny` > 16, so `min_step = 32`, `:706-719`) it cuts the output into **32-row × 32-activation-row
tiles** and hands them to threads round-robin: `for (itile = ith; itile < ntile; itile += nth)`
(`:712`). Inside a tile, `mul_mat_NxM` (`:63-100`) walks the activation rows in blocks of at
most 8 (`ny = funcs.size()`, `:83-100`) and the output rows in steps of 64 (`k_x_step`, `:67`).
There is a second split when threads outnumber tiles (`npt <= 16 && nth even && Ny >= 16`,
`:764-780`): half the threads take the first half of the activation rows and half the second,
so that 12 threads still get 32-row activation tiles rather than 8-row ones. Every thread also
waits at one barrier while the activations are quantised (`ggml.c:18852-18876`).

### 2.5 Its flash attention

`iqk_flash_attn_noalibi` (`ggml/src/iqk/iqk_flash_attn.cpp:744`) splits a *prompt* pass over
(query-row block, KV head) pairs (`:755-770`): each thread handles one head and a slice of the
query rows (`neq1g = neq1/ntg`, `:757`), and calls `iqk_flash_attn_impl` once for that slice.
For decode it instead splits the **KV positions** into blocks of 32 (`nk = 32*nm`, `:640`), each thread
leaves a partial (max, sum, output) in a work buffer, a barrier follows, and a second pass
merges them with `accumulate_qkv` (`:620-700`).

The kernel itself (`ggml/src/iqk/fa/iqk_fa_templates.h`) is templated on `(Dk, Dv, q_step,
k_step)`; for a 128-wide head it uses `k_step` 128/64/32 by divisibility
(`ggml/src/iqk/fa/iqk_fa_128_128.cpp:38-48`) and picks `q_step` 16/32/64 from the query-row
count, or 8 for a few rows (`iqk_fa_templates.h:2064-2103`) — 64 for a 512-token prompt. It
keeps the whole `q_step × k_step` score block in registers (`FlashMS::cache`,
`iqk_fa_templates.h:964`), and
does the QK product as 512-bit FMA over f16 halves widened with `_mm512_cvtph_ps` (the bf16
variant uses `vdpbf16ps`, `:1678-1685`). So ik's attention is 512-bit on this host.

---

## 3. How Peregrine does it

### 3.1 The same layout decision, made in Rust

`crates/peregrine-kernels/src/kq.rs` defines the eight-row tiles (`Q4K8` 1152 B, `Q5K8`
1408 B, `Q6K8` 1680 B, `Q5_08`, `Q8_08`) and `repack_*8`; the store is built at load and
pointed at by `TensorInfo::kq_off`. `MatKernels::detect`
(`crates/peregrine-kernels/src/table.rs:60-95`) sets `kq_tiles` from the repack switch and
`kq_zmm` from `have_zmm512()`, and `table.rs`'s `path()` sends Q4_K/Q5_K/Q6_K/Q5_0/Q8_0 to
`MatPath::KqTiles` when the matrix carries a store offset.

Eight output rows per tile rather than four, and **four activation rows interleaved in a
separate buffer** (`kq::KqActs`, filled once per matmul group,
`crates/peregrine-core/src/blocks/linear.rs:154-159`), rather than ik's one-row q8_K. The
activation codes are int8 with a per-256 scale, plus per-16 run sums — the same quantisation
family as ik's.

### 3.2 The Q4_K ZMM kernel

`q4k8_group_zmm` (`crates/peregrine-kernels/src/kqsimd.rs:1574-1663`), called by
`q4k8_gemm_zmm` (`:1665-1690`) in groups of four activation rows. Per super-block and
activation row:

- 16 windows of 64 bytes (`q4_qs()`), each loaded once and split into low/high nibbles with one
  `vpand` for the low half and a `vpsrlw` + `vpand` for the high: 16 loads and 48 ops per
  super-block, shared by the four activation rows;
- 2 `vpdpbusd` per (window, activation row) → **32 `vpdpbusd zmm` per (super-block, activation
  row)** = 2048 MACs (8 rows × 256 weights); 64 byte-MACs each, all useful;
- the scale: one `vpmulld` + `vpaddd` per (4-window group, activation row, half) = 8 + 8 per
  (super-block, activation row) — exact int32, on the same "different port" argument as ik's;
- the min term: 8 sub-blocks × (a scalar `vpbroadcastd` from `src.min_sum(..)` + `vpmulld` +
  `vpaddd`) per activation row — 24 instructions, and the broadcast comes from memory each
  time (`kqsimd.rs:1645-1652`);
- the fold: `fold()` (`kqsimd.rs:373`) once per (super-block, activation row).

Two things follow from the code and are visible in the measurement. First, the kernel's
byte-dot share of instructions is high — 32 `vpdpbusd` against roughly 90 other instructions
per (super-block, activation row) — so if it issued at the roof it would be a ~120 GMAC/s
kernel. Second, the epilogue is **per activation row**, not per group: the min-term broadcasts,
the mullo/add pairs and the fold all repeat for each of the four rows. Measured, it reaches 76–
79 GMAC/s (17–18 % of the roof) and is flat from 16 to 128 activation rows (79.2 / 77.1 /
75.8), which says adding activation rows does not amortise what is left.

### 3.3 Q6_K and Q5_K have no 512-bit kernel

`q6k8_gemm` (`kqsimd.rs:933-1294`) is the AVX2 kernel: `vpmaddubsw` + `vpaddw`, widened to int32
every two `vpmaddubsw` because four products per int16 lane are the bound (`docs/design/s3.4.md`
§13.2). It reaches 37–39 GMAC/s — 9 % of the byte-dot roof — and ik's Q6_K_R4 reaches 123–153.
Q6_K is 15.4 % of the 1B's prefill time. This is the largest single kernel gap in the engine.

### 3.4 The pass, the threads, and the write-out

A prompt is read in pieces of `prefill_chunk` rows
(`crates/peregrine-core/src/serve.rs:52-56`; 16 up to the core count, 64 above it, 64 for
prompts longer than 1024 tokens, `serve.rs:69-80`). Each piece is one fork–join per matmul
group and one barrier per layer. Within a piece, rows of each matrix are handed out in tiles of
eight (`linear.rs:246-330`), and **each tile's eight results are written out transposed**:

```rust
for i in 0..KQ_TILE_ROWS { ... for b in 0..nb { out.set((b0+b)*rows + row, out[b*8+i]) } }
```

(`crates/peregrine-core/src/blocks/linear.rs:289-303`). For `nb = 16` that is 128 scattered
stores per tile; for `nb = 128` it is 1024 stores with a stride of `rows` floats (32 KiB on the
1B), which is where the larger chunk starts to cost more than it saves. That is the mechanism
behind §1.8's "a wider pass makes the 1B slower", and it is a *scheduling* cost only: the
result does not depend on `nb` (`crates/peregrine-models/tests/prefill_chunk.rs` pins it).

### 3.5 Attention

The score kernel is AVX2: `scores_f16` / `scores_f16_pair`
(`crates/peregrine-core/src/attn.rs:1236`, `:1287`) are `#[target_feature(enable =
"avx2,fma,f16c")]`, four query rows at a time, 8 floats per FMA. Only the V mix has a 512-bit
variant (`mix_f16_avx512`, `attn.rs:1567`, chosen by `avx512_attn`, `:1205`). The plan
(`attn::plan`, `attn.rs:178`) splits a multi-row pass by rows and lets a one-row call split
positions. Attention is 7.5 % (1B) and 10.2 % (Qwen) of a 512-token prefill here; at 4096 it
grows to ~29 % (`docs/design/s3.4.md` §12).

---

## 4. Accuracy: is either fast path exact?

**ik_llama.cpp.** Its fast path is exact with respect to the file's weights: `-rtr` is a
permutation of the same bytes, in place (`iqk_quantize.cpp:9763-9810`), and the kernel reads
the file's 4-bit and 6-bit codes. What it is *not* exact about is the activation: every
`mul_mat` quantises the forward activations to int8 per 256 weights
(`ggml.c:18852-18876`) and the product is an int8 × int8 dot, accumulated in int32 with an exact
per-sub-block int32 scale and folded in f32 (`iqk_gemm_kquants.cpp:1528-1555`). That is llama.cpp's
normal behaviour, not something `-rtr` adds.

**Peregrine.** The same shape: int8 codes from the file's nibbles, int8 activations per 256
weights, int32 `vpdpbusd`, exact int32 scale, f32 fold (`kqsimd.rs:1574-1663`). The AVX-512
kernel is **bit-identical to the AVX2 one**
(`the_avx512_kernel_agrees_with_the_avx2_one_bit_for_bit`, quoted in `s3.4.md` §12.1), and the
repacked path is bit-identical at every batch width because the int32 sums are exact and the
fold is the same expression in the same order (ADR 0003,
`crates/peregrine-core/src/blocks/linear.rs:246-330`; the invariance tests are named in
`s3.4.md` §4.4).

**Consequence for this study.** A faster kernel that stays inside this shape — int8 weights
from the file, int8 activations per 256, exact int32 accumulation, one exact int32 scale
multiply per sub-block, the same float fold — **cannot move the accuracy gate**
(`docs/targets.md` §1, the KL against llama.cpp). The two things that would move it are (a)
changing the activation quantiser (for example to fp8 or bf16 with a different rounding) and
(b) reordering the *float* fold. Both are avoidable, and the second is how `s3.4.md` argued the
AVX-512 kernel through: the integer sums are order-free, and only the f32 expression order has
to be reproduced.

---

## 5. The account

Llama 3.2 1B, 512-token prompt, 12 threads. Measured: P 288.6 tok/s, ik 549.9 tok/s, ratio
1.90.

| part | value | ratio | how |
|---|---|---|---|
| Peregrine's matmul share of a pass | 80.9 % | — | measured (§1.4) |
| kernel rate, Q4_K at 16 acts | P 79.2 vs ik 134.0 GMAC/s | 1.69× | measured (§1.6) |
| kernel rate, Q4_K at 128 acts | P 75.8 vs ik 148.1 | 1.95× | measured |
| kernel rate, Q6_K | P 37.0 vs ik 152.8 | 4.1× (3.1× at 16 acts) | measured |
| kernel rate weighted by the format mix | 0.69×1.75 + 0.19×3.2 + 0.12×1.7 ≈ **2.0×** | 2.0× | measured rates; the shares are the measured fmt shares, and the 12 % unattributed matmul is assumed Q4_K-like (estimate) |
| thread scaling 1→12 | P 6.0× vs ik 6.7× | 1.12× | measured (§1.7) |
| pass size | P at its own optimum; ik +10 % from 16→128 acts | ≈1.0–1.1× | measured (§1.6, §1.8) |
| non-matmul | P 19.1 % of the pass | ≤1.25× cap | measured; ik's share is not measured |
| **one thread, expected** | 2.0 × 1.0 × ~1.0 | **≈2.0×** | predicted |
| **one thread, measured** | 82.0 / 47.6 | **1.72×** | measured |
| **12 threads, measured** | 549.9 / 286.1 | **1.92×** | measured |

Qwen 2.5 0.5B, same shape: measured 2.42× at 12 threads and 1.96× at one. Its weight mix is
Q5_0-dominant, where the measured kernel ratio is 2.1–3.5× (Q5_0) and 3.2× (Q6_K), so a
weighted 2.4–2.8× against a measured 2.42× is the same story with a larger Q5_0 term.

Read it as: **the per-core k-quant kernel is the whole gap at one thread, and thread scaling
adds the rest at twelve.** Pass size, pinning and non-matmul overhead are not the gap. The
error bars are large — the 1-thread prediction is 2.0 against 1.72 measured, and ik's matmul
share inside its own pass was not measured (no `perf` under WSL2) — so the table claims the
*kernel* term is dominant, not that the terms multiply out exactly.

A second, independent way to see the same thing, on Qwen (whose weights are mostly Q5_0):
181 GMAC of trunk work per 512-token prompt, at one thread. ik finishes it in 2.78 s
(183.9 tok/s) = **65 GMAC/s**; Peregrine in 5.47 s (93.6 tok/s) = **33 GMAC/s**. Against their
own Q5_0 kernels — ik 108 GMAC/s at 16 activation rows, Peregrine's ZMM tile 48.7 — those are
60 % and 68 % of the micro-benchmark rate, i.e. the two engines lose a similar fraction in
situ, and the 1.97× engine ratio is the 2.1× kernel ratio at the widths the engines use.

---

## 6. What to build

Ranked by the gain divided by the size. Nothing here makes a result depend on batch width,
thread count or pass size; item 3 removes such a dependence in *speed*, not in results.

### 1. Cut the per-activation epilogue in the Q4_K/Q5_0 ZMM kernels (biggest)

- **What**: make the min term and the fold run once per *group of four activation rows* instead
  of once per row: keep the 8 sub-block min-sum broadcasts in registers across the group, and
  replace the eight scalar `min_sum(..)` broadcasts per (super-block, activation row) with one
  loaded vector per group. Also stop re-decoding the tile's scales per activation row.
- **Evidence**: the kernel is at 17–18 % of the 455 GMAC/s roof while its instruction mix says
  ~26 %; the flat 79/77/76 GMAC/s at 16/64/128 activation rows says the cost is per row, not per
  weight. ik's equivalent loop reaches 60–66 % of its roof with less amortisation opportunity
  (4 rows, one activation row at a time). Target 76 → 120 GMAC/s (1.6×). Q4_K is 56 % of the
  1B's pass, so that would be **+20–25 %** on the 1B (31.3 ms → 19.9 ms of a 55.8 ms pass); on
  Qwen, Q5_0 (43 %) and Q4_K (9 %) together give **+15–20 %**. Scale the estimate by the
  fraction of the 1.6× that the built kernel actually reaches.
- **Bits**: bit-identical, because the integer sums and the f32 fold can be kept exactly as
  they are (the same argument `s3.4.md` §4.4 makes for the existing ZMM kernel).
- **Size**: 1–2 weeks, the most delicate item here (register pressure is the whole point).

### 2. A 512-bit Q6_K tile kernel (the simplest of the two kernel jobs)

- **What**: `q6k8_gemm_zmm` beside `q6k8_gemm`, the same eight-row `Q6K8` tile, ZMM registers,
  `vpdpbusd` into int32 with the run's `16·Σq` offset subtracted in int32 (as `q6k8_gemm` does),
  widened after every two windows as the int16 bound requires — the AVX-512 form of the same
  arithmetic.
- **Evidence**: Q6_K is 15.4 % of the 1B's prefill time (measured) and its kernel runs at 37–39
  GMAC/s against ik's 123–153 (measured). Bringing it to Q4_K-ZMM's 76 GMAC/s (= 2×) removes
  8.6 ms × (1 − 1/2) = 4.3 ms of a 55.8 ms pass: **+8 % on the 1B, +2 % on Qwen**. If it reaches
  ik's 120 GMAC/s (3.2×) it is **+11 % on the 1B**.
- **Bits**: bit-identical if the integer sums and the fold order are the ones `q6k8_gemm` uses
  (that is what `q4k8_gemm_zmm` does against `q4k8_gemm`); goldens do not move. If the fold has
  to change, the goldens move and are re-recorded per tier.
- **Size**: ~1 week (one kernel + GEMV, plus the `kq_zmm` dispatch and the bit-identity test).

### 3. Make the wide pass pay: write the output in the tile's order

- **What**: the matmul writes `out[(b)*rows + row]` from a tile-major buffer, so a wider pass
  costs more scattered stores (§3.4). Either write the tile block into a row-major scratch and
  copy it once per group, or change the output layout the callers read to tile-major and
  untile at the end of the layer. A per-`nb` block of 8 rows × `nb` columns fits in L1 for
  `nb ≤ 128`.
- **Evidence**: `PEREGRINE_PREFILL_CHUNK` 16 → 128 costs the 1B 8 % at twelve threads and 30 %
  at one (measured), while Qwen *gains* 15 %; ik gains 28–58 % from the same move. If the 1B
  could take the chunk 64 point without the penalty it would be at Qwen's +15 %, and it would
  also make the 24-thread rows usable.
- **Bits**: bit-identical. The chunk is scheduling, pinned by
  `crates/peregrine-models/tests/prefill_chunk.rs`.
- **Size**: 3–5 days.

### 4. A 512-bit attention score kernel

- **What**: `scores_f16_zmm` beside `scores_f16`, 16 floats per FMA, the same eight-lane
  reduction tree widened to sixteen so the per-key dot order is preserved.
- **Evidence**: attention is 7.5 % (1B) and 10.2 % (Qwen) of a 512-token prefill and ~29 % at
  4096 (`s3.4.md` §12); the score kernel is AVX2 today (`attn.rs:1236`) while the mix already
  has a 512-bit form. A 1.5× score kernel is **+2–3 % at 512 and +8–10 % at 4096**.
- **Bits**: bit-identical is achievable by keeping the same reduction tree (the existing
  AVX-512 mix is held to that standard); otherwise the goldens move.
- **Size**: 1 week.

### 5. Thread scaling past 12 (SMT and 24 threads)

- **What**: find why 24 workers add nothing (barrier count per layer, tile granularity of eight
  rows, chunk default). ik gains +40–51 % at 24 threads on this host (measured); Peregrine gains
  0–6 %.
- **Evidence**: §1.7. Worth **+20–40 % at 24 threads only**; it does not touch the 12-thread
  headline, which is why it is last.
- **Bits**: bit-identical (thread count already does not change results).
- **Size**: ~1 week, and it may end as "Peregrine's barrier is the floor" rather than a win.

**Not proposed**: a 16-column tile (measured in `s3.4.md` §12: the 512-bit byte-dot issues at
the same instruction rate as the 256-bit one, so width buys nothing here); an int8 activation
interleave (Peregrine already has one, and ik does without); anything that changes the activation
quantiser, because that is the accuracy gate.

---

## 7. What was not measured, and why

- **`perf` is not available under WSL2** (no `perf` binary, no `perf_event` access). ik's
  matmul share *inside its own prefill* is therefore inferred (§5), not measured. Everything
  about ik is measured either from `llama-bench` or from its own `ggml_mul_mat` in
  `ik_kqbench`.
- **The instruction-level count inside ik's kernel** is read from the source and not confirmed
  with a disassembly of the hot loop. The `vpdpbusd` population of the build was counted
  (26,526).
- **ik's own kernel width choice**: the source says `Q4_K_R4` uses `mul_mat_q4_k_r4_q8_k`
  (`__m256i`) and that is what the build's dispatch selects; a *disassembly* of that function
  was not checked.
- **Fast versus compact cores**: `lscpu -e` reports no MHz here (`MAXMHZ` is `-`), so the
  clusters were separated by CPU number and `taskset` instead. They measured the same, but the
  agent's own UI was running on cores 0–3, so the comparison is confounded in the direction
  that hides a fast-core advantage.
- **24-thread rows** for Peregrine are confounded with its chunk default (64), which is why
  both numbers are reported.
- **Nothing was measured at 4096 tokens** in this study; the 2026-11-11 ik README has those
  rows (P ÷ ik 0.49–0.58), and the note says where the picture is expected to shift
  (attention, not the matmul).
- **No accuracy run was made**: no engine code changed. The KL gate (`docs/targets.md` §1) has
  to be re-run for whichever kernel of §6 is built, on the reference hosts, as `s3.4.md` §10
  decision 4 requires.
