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

> **Corrected in §10 — strike this paragraph.** The table above is not reproducible. The core
> runs at **1.90 GHz**, one `vpdpbusd zmm` per cycle and two 256-bit ones, so the roof is
> **121-122 GMAC/s**, externally checked at 113.8 GMAC/s with a clock outside the VM. That
> session's time base ran ~1.8× fast, and the two widths were given the same instruction rate
> but different MAC counts: that is the 2× that is not there. See §10.0 and §10.4.

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

> **Corrected in §10 — strike the two rows above and the ik columns of this table.** ik's
> Q4_K_R4 kernel measures 57-67 GMAC/s through the same library at a real layer's shape
> (2048 × 8192), not 134-148; Peregrine's is 72-75 on the same shape, so at equal width the
> two kernels are level to Peregrine's advantage. Against the corrected 121 GMAC/s roof,
> Peregrine's Q4_K is at 61 % and the path ik actually runs at ub 512 is at 75 %.

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

> **Corrected in §10 — the whole account above is superseded.** The kernel ratios it is built
> on (1.69-1.95× for Q4_K, 3.1-4.1× for Q6_K) do not reproduce: at equal pass width the two
> engines are level (P ÷ ik 0.93-1.03 at 16 and 64 activation rows) and Peregrine's own
> kernel is 1.2-1.4× the fork's in isolation. The "thread scaling 6.0× against 6.7×" row also
> does not reproduce: today 6.50× against 6.43×. What the gap really is: the fork's pass is
> 512 rows wide, where it requantises 32 rows of weights to 8 bits once and runs a different
> kernel at 91 GMAC/s, and Peregrine's ceiling is a 128-row pass. See §10.5 and §10.6.

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

---

## 8. As built: a wide pass when nobody is decoding (S5.1a)

- Status: built and measured, no result moved
- Date: 2026-11-12
- Host: AMD Ryzen AI 9 HX 370 (Zen 5, 12 cores / 24 threads, AVX-512 F/BW/VL/VNNI/VBMI), WSL2
- Baseline for this change: the S5.1 stack (`s5.1-kq-epilogue`, PR #199)
- Raw measurements and the two drivers: `benchmarks/results/desktop-zen5-2026-11-12-s5.1-adaptive-chunk/`

Item 3 of §6 made a wide pass pay (PR #209), but the default chunk stayed 16 because the chunk
is also what a *decoding* user waits for while another user's prompt is read. This change stops
paying that price when nobody is waiting.

### 8.1 What was built

The chunk is now decided per pass, at the two places that pick one:

- `mixed_step` (the default schedule): if no slot is ready to decode (`dec.is_empty()`), the
  pass takes the **wide chunk**; otherwise today's rule (the chunk, or `LONG_PROMPT_CHUNK` = 64
  for a prompt over 1024 tokens). A long prompt keeps its 64-token pieces in both branches.
- the alternating schedule (`PEREGRINE_MIX=0`): a prefill pass widens unless another slot is
  waiting to decode (`slot.ready()`), where the narrow chunk still bounds that user's wait.

`PEREGRINE_PREFILL_CHUNK` still forces one chunk: when it is set, the wide chunk is that same
number, so the switch reproduces the old behaviour exactly (that is how the "before" rows below
were taken, on the same binary).

The wide chunk is **64 on both measured tiers** (`options::wide_chunk_default_for`). At 512
tokens, medians of three interleaved rounds, `peregrine bench --prompt-tokens 512`, prompt tok/s:

| model | tier | 16 | 64 | 128 |
|---|---|---|---|---|
| Llama 3.2 1B | avx512 | 301.9 | **336.4** | 327.8 |
| Qwen 2.5 0.5B | avx512 | 487.6 | 588.9 | **619.9** |
| Gemma 3 1B | avx512 | 268.8 | **329.8** | 322.4 |
| Llama 3.2 1B | avx2 | 244.9 | **272.1** | 263.3 |
| Qwen 2.5 0.5B | avx2 | 371.2 | 456.5 | **459.6** |
| Gemma 3 1B | avx2 | 229.6 | **261.4** | 260.5 |

64 wins on both 1B models on both tiers; 128 wins on Qwen by 5 % (avx512) and 1 % (avx2). The
two are within noise of each other overall, and 64 is free in memory (below), so 64 it is. The
embedding path's pass-size rule (`options::pass_rows_default_for`) returns 16 for both models
here — the L2 bound gives 32 and 73 rows, but the ridge bound (0.56 bytes per weight × 10.8) is
6, clamped to 16 — so that rule is today's chunk and not a wide one.

**Memory.** The pass's scratch is sized once, in `Engine::build`, and nothing allocates in the
pass: `chat_rows` is `max(chunk, wide_chunk, 64, 64) + max_slots`, so a wide chunk of 64 needs
exactly the rows the engine already reserved (`SCRATCH_CHUNK` = 64, for speculation's rows) and
the change costs **zero bytes**. 128 would have cost 64 more rows, measured as a peak-RSS
difference of 17.6 MiB on the 1B and 2.3 MiB on Qwen 0.5B — one of the reasons 64 was chosen.

### 8.2 Single user, before and after

`peregrine bench`, 12 threads, before = `PEREGRINE_PREFILL_CHUNK=16`, after = the new default;
three interleaved rounds, medians; host load average 0.1–0.6. The time to first token is the
prefill time the bench reports (the first token is one row after it).

| model | tier | prompt | before tok/s | after tok/s | ratio | before TTFT s | after TTFT s |
|---|---|---|---|---|---|---|---|
| Llama 3.2 1B | avx512 | 512 | 305.5 | 338.4 | 1.11 | 1.676 | 1.513 |
| Llama 3.2 1B | avx512 | 4096 | 236.9 | 238.2 | 1.01 | 17.29 | 17.20 |
| Qwen 2.5 0.5B | avx512 | 512 | 488.0 | 594.0 | 1.22 | 1.049 | 0.862 |
| Qwen 2.5 0.5B | avx512 | 4096 | 397.9 | 394.4 | 0.99 | 10.30 | 10.39 |
| Gemma 3 1B | avx512 | 512 | 277.2 | 326.8 | 1.18 | 1.847 | 1.567 |
| Gemma 3 1B | avx512 | 4096 | 311.6 | 312.6 | 1.00 | 13.14 | 13.10 |
| Llama 3.2 1B | avx2 | 512 | 245.2 | 275.0 | 1.12 | 2.088 | 1.862 |
| Llama 3.2 1B | avx2 | 4096 | 211.7 | 214.3 | 1.01 | 19.35 | 19.12 |
| Qwen 2.5 0.5B | avx2 | 512 | 412.5 | 462.9 | 1.12 | 1.241 | 1.106 |
| Qwen 2.5 0.5B | avx2 | 4096 | 349.2 | 350.4 | 1.00 | 11.73 | 11.69 |
| Gemma 3 1B | avx2 | 512 | 235.5 | 268.7 | 1.14 | 2.174 | 1.905 |
| Gemma 3 1B | avx2 | 4096 | 254.3 | 254.3 | 1.00 | 16.11 | 16.11 |

The 512-token rows are the gain; the 4096 rows are flat because a prompt that long was already
read in 64-token pieces (`LONG_PROMPT_CHUNK`), so the wide chunk changes nothing there. The
Qwen 4096 row is 1 % down, inside its 3.5 tok/s spread. The noisiest row is Qwen on avx2 at
512: its narrow samples spread 64 tok/s (291.9, 412.5, 420.1) against 9 at the wide chunk
(461.8, 462.9, 479.1), so its ratio is 1.12 by median and 1.25 against the 371.2 the same
narrow setting measured in §8.1.

Decode alone is unchanged, as it must be (a decode step carries no prompt rows): 62.8 against
63.3 tok/s for the 1B and 98.9 against 97.0 for Qwen at 512/16 on avx512, 65.0/65.6 and
98.1/102.9 on avx2.

### 8.3 Four and ten users, before and after

`benchmarks/run.py --mode server --contexts 512 --tokens 64 --threads 12 --user-counts 4,10`,
three interleaved rounds per invocation, four invocations per variant, so twelve samples per
row; medians, with the observed range in brackets. "Per-user decode" is the harness's
decode-window aggregate divided by the user count: the rate one user gets while the others
decode.

| model | users | metric | before | after |
|---|---|---|---|---|
| Llama 3.2 1B | 4 | total tok/s | 30.85 [29.8, 41.7] | 30.93 [29.8, 31.3] |
| Llama 3.2 1B | 4 | per-user decode tok/s | 53.1 [45.3, 57.2] | 53.4 [49.7, 56.3] |
| Llama 3.2 1B | 4 | TTFT s | 5.23 [3.09, 5.45] | 5.21 [5.05, 5.42] |
| Llama 3.2 1B | 10 | total tok/s | 31.87 [31.5, 36.0] | 35.57 [31.8, 36.5] |
| Llama 3.2 1B | 10 | per-user decode tok/s | 52.8 [48.1, 54.6] | 54.4 [48.9, 67.4] |
| Llama 3.2 1B | 10 | TTFT s | 11.28 [11.21, 11.49] | 11.04 [9.09, 11.35] |
| Qwen 2.5 0.5B | 4 | total tok/s | 47.90 [46.3, 72.7] | 50.03 [48.2, 51.0] |
| Qwen 2.5 0.5B | 4 | per-user decode tok/s | 78.7 [73.1, 83.2] | 77.1 [74.8, 83.1] |
| Qwen 2.5 0.5B | 4 | TTFT s | 3.36 [1.53, 3.58] | 3.14 [3.11, 3.26] |
| Qwen 2.5 0.5B | 10 | total tok/s | 60.05 [49.0, 61.9] | 57.25 [50.8, 64.0] |
| Qwen 2.5 0.5B | 10 | per-user decode tok/s | 79.9 [75.6, 83.8] | 82.9 [72.6, 84.9] |
| Qwen 2.5 0.5B | 10 | TTFT s | 7.13 [4.87, 7.29] | 6.81 [4.53, 6.90] |

Two things to read. The quantity the change is meant to protect — a decoding user's rate while
somebody else's prompt is being read — is flat at four users (53.1 → 53.4, 78.7 → 77.1) and
slightly up at ten (52.8 → 54.4, 79.9 → 82.9): the chunk falls back while anyone decodes, so no
decoder gets a wider pass in front of it. And total throughput does not drop: flat at four
users, +11.6 % and −4.7 % at ten (both inside the spread of the "before" samples, whose upper
half is bimodal — WSL2 guest-clock jumps, which this harness has hit before, make single
invocations read up to 1.5× high; the medians are of twelve samples for that reason).

At ten users the median TTFT is 0.2 s (1B) and 0.3 s (Qwen) earlier: the first slots' prompts
are read wide while nobody decodes yet, and only the passes after the first decoder exist are
bounded by the narrow chunk.

### 8.4 Bits

Nothing allocates in the pass and no golden moved: the chunk is how a prompt is *read*.
`crates/peregrine-models/tests/prefill_chunk.rs` gains two tests:

- `a_prompt_read_in_two_widths_says_the_same` reads the same prompt narrow-then-wide and
  wide-then-narrow and compares the KV cache and the last position's logits with a uniform read;
- `the_engine_widens_the_pass_when_no_slot_is_decoding` runs the scenario the change is for — a
  short prompt that starts decoding half way through a long one and stops before it ends — and
  pins three things: the first pass carries the wide chunk, every pass that carried a decode row
  is bounded by the narrow chunk plus that row, and the pass widens again after the decoder
  stops. The tokens per slot are identical to the same run with one chunk.

Goldens pass unchanged on `x86_64-avx2-avx512-vnni-gfni-vbmi`, `x86_64-avx2` and `x86_64`
(scalar) for Llama 3.2 1B, Qwen 2.5 0.5B and Gemma 3 1B.

---

## 9. As built: the Q4_K kernel's instruction mix, measured (S5.1b)

- Status: measured; **no kernel change shipped**, because nothing correct was faster
- Date: 2026-11-12
- Host: AMD Ryzen AI 9 HX 370 (Zen 5, 12 cores / 24 threads), WSL2, one core (`taskset -c 4`)
- Raw measurements and the two probes: `benchmarks/results/desktop-zen5-2026-11-12-s5.1-q4k-kernel-2/`
- Fork read: ik_llama.cpp at `479074c`, already built at `~/.cache/peregrine-bench/ik_llama.cpp-479074c`

This item was "rebuild the Q4_K kernel with the fork's instruction mix, at 120 GMAC/s per core". The
instruction mix was read off both binaries, both kernels were measured, and the target turned out to
be above what this core does with a byte dot product at all. The kernel already runs at the fork's
rate, and at the machine's.

### 9.1 The two inner loops, instruction for instruction

Fork: `objdump -d` around `mul_mat_q4_k_r4_q8_k<8>` (`+0x6b0`, the 116-instruction loop at
`0x16951c0`). One iteration is one sub-block: 4 weight rows × 32 weights × 8 activation rows =
1024 MACs. Peregrine: `objdump -d` on `q4k8_gemm_zmm` in the `kqbench` binary (the loop at
`0x2cc90`); one iteration is a 4-window group of one super-block, and four of them are a
super-block: 8 weight rows × 256 weights × 8 activation rows = 16 384 MACs.

Per 1024 MACs, which is the fork's whole iteration and an eighth of Peregrine's super-block:

| per 1024 MACs | fork (256-bit) | Peregrine (512-bit) |
|---|---|---|
| `vpdpbusd` | 32 × ymm (32 byte-MACs each) | 16 × zmm (64 each) |
| multiplies, other | 8 `vpmulld` ymm (int32 scale) | 4 `vpmulld` zmm (int32 scale) |
| adds | 8 `vpaddd` ymm | 4 `vpaddd` zmm |
| activation broadcast | 32 `vpshufd` ymm from 8 loads (32 B) | 16 `vpbroadcastq` zmm from memory (8 B) |
| weight loads | 2 `vpandq` + 2 `vpsrlw` + 2 `vpandq` from memory, 128 B | 8 `vmovdqu64` zmm, 512 B, shared by 8 rows |
| activation bytes read | 256 | 128 |
| weight bytes read | 128 | 512 (amortised over 8 rows, so 512 per 8 rows) |
| nibble decode | 6 ops per 128 B | 4 ops per 64 B (48 per super-block) |
| loop-carried chains | 8 accumulators, one per activation row, 4 `vpdpbusd` deep; then one `vpmulld`+`vpaddd` per row | 16 accumulators (2 per row), 4 `vpdpbusd` deep; then 2 `vpmulld`+`vpaddd` per row |
| instructions in the loop | 116 | 215 per group, 860 per super-block |
| instructions per MAC | 0.113 | 0.052 |
| measured | 58–67 GMAC/s | 77–81 GMAC/s |

The multiply kinds are the same two on both paths: a VNNI byte dot product and an exact int32
multiply for the scale. Neither uses `vpmaddubsw` or `vpmaddwd` — those are the AVX2 kernels'
instructions, needed there because AVX2 has no VNNI; on this tier they would only add the int16
headroom problem back.

### 9.2 What the fork does that Peregrine does not

The fork forms the same MACs with **256-bit VNNI**: four weight rows per `vpdpbusd` instead of
eight, so it needs two instructions where Peregrine needs one, and it holds **one int32 accumulator
per activation row** (eight in all) where Peregrine holds two (sixteen). Its **activation broadcast
is a register shuffle**, `vpshufd` on the 32-byte `q8_K` load the row already needed, while
Peregrine's is a **`vpbroadcastq zmm` from memory, one per byte dot product**. Its scale multiply is
one `vpmulld` per (sub-block, activation row) on int32 partials — the same instruction Peregrine
uses — and its **min term is computed once per super-block** and shared by all eight activation
rows, where Peregrine computes a per-row min term with four `vpdpwssd`. Its weight block is four
rows interleaved per load (32 bytes); Peregrine's is eight (64 bytes).

So the fork wins on instruction count per MAC four times over and still runs *slower* per core than
Peregrine on this machine. That is the measurement, and §9.3 says why.

### 9.3 The core's byte-dot roof, and what both kernels are at

`peak.c` (in the results directory) is the roof probe: 16 named accumulators, one barrier per
iteration so nothing is hoisted, one instruction per accumulator per iteration. Measured on this
host, twice per run:

| probe | result |
|---|---|
| `vpdpbusd zmm`, 16 accumulators | 1.65–1.83 G instructions/s = **106–117 GMAC/s** |
| `vpdpbusd ymm`, 16 accumulators | 3.42–3.64 G instructions/s = **110–117 GMAC/s** |
| `vfmadd512`, 8 accumulators | 1.67–1.81 G instructions/s |

Two facts follow. A 512-bit byte dot product **costs twice a 256-bit one** (1 per 1.8 cycles
against 2 per cycle at the same clock), so VNNI width buys **nothing** here — the two columns are
the same GMAC/s, which is exactly what the two kernels show. And the roofs imply a clock near
1.7 GHz for a core that issues 2 × 256-bit or 1 × 512-bit vector op per cycle: `vfmadd512` at
1.7 G/s is one per cycle, and the earlier "455 GMAC/s / 3.55 GHz" figure in §1.2 is not
reproducible — nothing on this host reaches it, with any width.

Against that roof: Peregrine 77–81 GMAC/s (73 %), the fork 58–67 (55 %). Both are near the wall.
The 120 GMAC/s target is *above* the register-only roof's 106–117, i.e. unreachable by any kernel
of this shape on this core, and the "134–148 for ik" that set the target does not reproduce:
`ik-gemm-harness.c` runs the fork's own kernel through its own `ggml_mul_mat` (Q4_K_R4 weights,
Q8_K32 activations, one thread, 1024 output rows) and gets **58–67 GMAC/s** at 16, 64 and 128
activation rows against 4 000-token-equivalent work; its Q6_K kernel measures **52–69**.

Peregrine's own Q6_K ZMM kernel, on the same L1 tile, is 58–61 GMAC/s — the same wall.

> **Corrected in §10 — the roof stands, the conclusion does not.** Today's roof is 121-122
> GMAC/s (113.8 by a clock outside the VM), so the 106-117 measured here is the one number this
> section got right, and "near the wall" is true of the 4-bit kernel. But the fork's engine does
> not run that kernel at ub 512: at 32 activation rows or more its dispatcher requantises 32 rows
> of weights to 8 bits and runs an 8-bit kernel, which measures **91.3 GMAC/s** at 512 rows —
> 75 % of the corrected roof, not 55 %. That is the check this section was missing. See §10.3.

### 9.4 What was tried

Three kernels were built and measured in `kqbench`, and none is shipped:

1. **256-bit VNNI with the fork's shape** (4 weight rows per `vpdpbusd`, one broadcast serving two
   byte dots, register-shuffle broadcast from a 32-byte load, the same exact int32 scale and the
   same fold four lanes at a time): **65–68 GMAC/s**, i.e. 12 % *slower* than the 512-bit kernel.
   The extra instructions per MAC cost more than the cheaper broadcast saves.
2. **Halving the 512-bit broadcasts** (the same broadcast reused by both nibble half), which is
   not bit-correct but isolates the cost: **81.4 against 78.7** — 3 %. The broadcast is not the
   kernel's limit.
3. **Pre-broadened activation** (64 bytes per byte dot product, a plain load instead of
   `vpbroadcastq`): the load itself is cheap, but building the buffer costs one broadcast and one
   store per byte dot product, which is the work being removed; it cannot pay.

What the measurement says the remaining 27 % would cost: the kernel issues 860 instructions per
super-block against 256 `vpdpbusd` (32 per row × 8), i.e. two non-dot instructions per dot, and
the dot alone is 1 per 1.8 cycles. Cutting the epilogue in half would buy at most a fifth.

### 9.5 Where the prompt-time gap therefore is

Peregrine is at 0.52 of ik_llama.cpp on Qwen 2.5 0.5B (594.0 against 1 132.6 tok/s) and 0.64 on
Llama 3.2 1B (338.4 against 530.6) at 512 prompt tokens, twelve threads, today's defaults. Since
the Q4_K kernel is not behind — it is ahead per core — the gap has to be outside it: ik's pass
splits a prompt over 32-row tiles and its own flash attention (§2.4, §2.5), and its ubatch is 512
where Peregrine's default pass is 16 (64 wide). The next step for prompt speed is §6 item 4 and
item 5, not the Q4_K kernel.

> **Corrected in §10 — strike the conclusion of this paragraph.** The end-to-end numbers stand
> (today 0.65 and 0.55 at 512 tokens, twelve threads, on this branch). "The gap has to be outside
> the Q4_K kernel" is wrong: the gap is a kernel, just not the one this section measured — the
> fork requantises weights to 8 bits per 32-row block when the pass is wide, which is why its
> 4-bit kernel's rate says nothing about its engine. The pass size is the *enabler* of that
> kernel rather than a gain of its own: at equal pass width the two engines are level today
> (§10.5, §10.6).

### 9.6 Bits

No kernel changed, so no golden could move; the only code change is one line in
`kqbench` (`out` sized for 128 activation rows, which the width sweep already asked for and
which used to panic before the Q6_K rows). Goldens pass unchanged on the avx512, avx2 and
scalar tiers.

---

## 10. Round 3: the measurements reconciled

- Status: the record corrected; **no engine change**
- Date: 2026-11-12 (03:40–05:10 local, UTC+7)
- Host: AMD Ryzen AI 9 HX 370 (Zen 5, 12 cores / 24 threads, AVX-512 F/BW/VL/VNNI/VBMI/BF16),
  WSL2, 11.7 GiB
- Peregrine: this branch (`s5.1-reconcile` = `s5.1-adaptive-chunk` + §9's write-up),
  `cargo build --release -p peregrine`; ik_llama.cpp: `479074c`, the build PR #177 measured
  (not rebuilt: `isa.txt` shows its runtime instruction sets, `ldd`, its CMake flags and its
  26,526 `vpdpbusd`)
- Raw measurements, the harness, the two probes, the clock calibration and the table
  generators: `benchmarks/results/desktop-zen5-2026-11-12-s5.1-reconcile/`; the fork-side
  probe is `benchmarks/ik_kernel_probe.cpp` (it links the fork's library; no fork source is
  copied into this repository)
- Every ratio below is **Peregrine ÷ the fork**. Every rate is converted with the model's own
  multiply-accumulate count per prompt token, read from its tensor shapes: **0.9731 GMAC** for
  Llama 3.2 1B (16 layers: q 4.19 + k 1.05 + v 1.05 + o 4.19 + gate 16.78 + up 16.78 +
  down 16.78 MMAC per layer) and **0.3579 GMAC** for Qwen 2.5 0.5B (24 layers: 0.80 + 0.115 +
  0.115 + 0.80 + 4.36 × 3). The output head runs for the last token only and is excluded; it
  is 0.136 GMAC on Qwen, +0.07 % of a 512-token pass.

§1 and §9 disagreed about the fork by a factor of two and about the roof by a factor of four,
and only one side of each could be right. Both were wrong, for three separate reasons, and the
one check neither session made — that an engine's per-core GMAC/s times its matmul share must
be at or below its isolated kernel's rate — is what settles it.

### 10.0 What was wrong, and what is struck

**1. The clock this host gives a process moves, and both sessions timed with it.** §1.2's
455 GMAC/s needs 7.1 G `vpdpbusd` instructions/s; §9.3's 106-117 needs 1.7 G. Today, at the
same absolute wall-clock rate checked from outside the VM, the number is **113.8 GMAC/s**. A
3 s burn loop on one core reads CLOCK_MONOTONIC 3.000 s against 2.78 s of CPU time and 2.78 s
of CLOCK_MONOTONIC_RAW; 40 s of the same loop bracketed by two remote HTTP `Date` headers
reads MONOTONIC 40.02 s against a host 38 ± 2 s and 37.18 s of CPU time. **MONOTONIC ran 5-8 %
fast in this session.** Earlier in the *same* session it ran 1.6-1.8× fast (MONOTONIC 1.76 s
for 1.00 s of CPU time), which is the size of the error that separates §1.2's table from
today's, and it is exactly the hazard §1.1 named: a rate timed on a fast clock is understated,
a rate timed on a slow one inflated, and neither session recorded which it had.

**2. The 512-bit byte dot is one per cycle, not two.** §1.2 measured the same *instruction
rate* for ymm and zmm (7.1 G/s) and then multiplied one by 32 MACs and the other by 64. Today,
with sixteen named accumulators in inline asm: `vpdpbusd zmm` 1.89-1.91 G/s = **1.00 per
cycle**, `vpdpbusd ymm` 3.83-3.84 G/s = **2.01 per cycle** — the same GMAC/s, 121-123. The
"455" is 2 × the ymm figure from the same run, and the factor of two is that error.

**3. Probe and engine were different code paths.** §9's `58-67 GMAC/s` is a fair measurement
of the fork's **4-bit** kernel on 4.5-bit weights. At `-ub 512` the fork does not run it: with
32 activation rows or more, its dispatcher (`iqk_dequant_type`, which the probe now prints)
requantises 32 rows of weights to 8 bits once and runs an 8-bit kernel, and *that* is the code
the 78-80 GMAC/s engine number comes from. Measuring the 4-bit kernel and comparing it with an
engine running the 8-bit one is the error the review suspected.

Struck, in place, above: §1.2's roof table and its 3.55 GHz (corrected note under it), §1.6's
ik columns and the "17-18 % / 60-66 %" sentences, all of §5's account, §9.3's "both kernels
are at the wall", and §9.5's conclusion. What stands: §9.1's instruction table (it is a
disassembly, and the counts are right), §9.2, §8's and §9's end-to-end numbers to within 3 %,
and §9's finding that nothing correct was faster than Peregrine's Q4_K kernel — that part was
measured correctly and is confirmed in §10.3.

### 10.1 Method, and which core

One harness (`sweep.py`) for both engines: every point is one process, points are interleaved
one per round, five rounds, and then the whole series is measured again 70 minutes later.
Every row records the invocation's wall time *and* its CPU time (the two clocks of §10.0) and
the host's load average. Medians of five; the two runs are both printed and they agree within
1-4 % everywhere except the fork's `-ub 32` row (see §10.2).

One-thread rows are pinned with `taskset -c 4`. `lscpu -e` says CPUs 0-7 are cores 0-3 (the
four full cores) and CPUs 8-23 are cores 4-11 (the compact ones), with two SMT threads each.
The same fixed-work loop timed on each of the 24 CPUs reads the same CPU time on all of them
(1.0006 s on every one), which is consistent with §1.9's finding that the two clusters are
indistinguishable for this workload and with the alternative that this host's CPU accounting is
too coarse to separate them. Either way there is no evidence of a core worth pinning to, so
**CPU 4 (core 2, a full core) is used throughout** and no result here depends on the choice
(§1.9 measured 4 threads on cores 0-3 against 4 on the compact cores at 244.5 and 264.3
tok/s — 8 % in the compact cores' favour, the opposite direction).

Probes are timed with the minimum over 5-7 passes of `min(CLOCK_MONOTONIC_RAW, CPU time)`,
which is the least-descheduled pass in real seconds; the absolute value is checked once against
the host clock (`roof_abs.sh`, §10.4).

### 10.2 The ground truth

512 prompt tokens, `-p 512 -n 0` for the fork (`-rtr 1`, `-fa` default) and
`peregrine bench --prompt-tokens 512 --tokens 1` with `PEREGRINE_PREFILL_CHUNK` set. Two runs,
70 minutes apart, medians of five interleaved rounds each.

**Llama 3.2 1B Q4_K_M — one thread** (pk = pass width: the fork's `-ub`, Peregrine's chunk)

| pk | fork tok/s | fork run 2 | fork GMAC/s | Peregrine tok/s | run 2 | Peregrine GMAC/s | P ÷ fork |
|---|---|---|---|---|---|---|---|
| 8 | 53.0 | 53.4 | 51.6 | 50.8 | 50.7 | 49.4 | 0.96 |
| 16 | 55.5 | 55.5 | 54.0 | 51.5 | 51.1 | 50.1 | 0.93 |
| 32 | 34.6 | 40.5 | 33.6 | 52.0 | 51.6 | 50.6 | 1.50 |
| 64 | 51.2 | 50.4 | 49.8 | 51.9 | 51.3 | 50.5 | 1.01 |
| 128 | 65.1 | 64.9 | 63.3 | 49.6 | 49.8 | 48.2 | 0.76 |
| 256 | 74.8 | 76.6 | 72.8 | — (cap 128) | — | — | — |
| 512 | **80.4** | 79.3 | **78.3** | — | — | — | — |

**Qwen 2.5 0.5B Q4_K_M — one thread**

| pk | fork tok/s | fork run 2 | fork GMAC/s | Peregrine tok/s | run 2 | Peregrine GMAC/s | P ÷ fork |
|---|---|---|---|---|---|---|---|
| 8 | 109.9 | 108.2 | 39.3 | 101.0 | 100.6 | 36.1 | 0.92 |
| 16 | 116.4 | 117.4 | 41.7 | 100.7 | 100.8 | 36.0 | 0.86 |
| 32 | 130.3 | 127.4 | 46.7 | 101.7 | 102.5 | 36.4 | 0.78 |
| 64 | 153.1 | 155.1 | 54.8 | 102.9 | 101.9 | 36.8 | 0.67 |
| 128 | 175.1 | 172.4 | 62.7 | 100.8 | 100.5 | 36.1 | 0.58 |
| 256 | 184.1 | 181.8 | 65.9 | — | — | — | — |
| 512 | **180.1** | 173.5 | **64.5** | — | — | — | — |

**Llama 3.2 1B Q4_K_M — twelve threads**

| pk | fork tok/s | fork run 2 | fork GMAC/s | Peregrine tok/s | run 2 | Peregrine GMAC/s | P ÷ fork |
|---|---|---|---|---|---|---|---|
| 8 | 278.2 | 273.0 | 270.7 | 275.2 | 278.9 | 267.8 | 0.99 |
| 16 | 307.4 | 302.6 | 299.2 | 307.0 | 297.8 | 298.7 | 1.00 |
| 32 | 230.0 | 231.6 | 223.8 | 331.4 | 317.6 | 322.5 | 1.44 |
| 64 | 326.9 | 333.1 | 318.1 | **337.5** | 335.1 | **328.4** | 1.03 |
| 128 | 431.0 | 428.8 | 419.4 | 318.9 | 321.5 | 310.3 | 0.74 |
| 256 | 488.7 | 473.1 | 475.6 | — | — | — | — |
| 512 | **517.6** | 520.3 | **503.6** | — | — | — | 0.62¹ |

**Qwen 2.5 0.5B Q4_K_M — twelve threads**

| pk | fork tok/s | fork run 2 | fork GMAC/s | Peregrine tok/s | run 2 | Peregrine GMAC/s | P ÷ fork |
|---|---|---|---|---|---|---|---|
| 8 | 459.7 | 470.1 | 164.5 | 430.9 | 419.1 | 154.2 | 0.94 |
| 16 | 558.4 | 549.7 | 199.8 | 494.9 | 483.4 | 177.1 | 0.89 |
| 32 | 665.5 | 652.9 | 238.2 | 564.7 | 540.2 | 202.1 | 0.85 |
| 64 | 845.8 | 867.7 | 302.7 | 593.4 | 579.2 | 212.4 | 0.70 |
| 128 | 968.9 | 1003.3 | 346.8 | **620.3** | 596.0 | **222.0** | 0.64 |
| 256 | 1071.4 | 1059.0 | 383.5 | — | — | — | — |
| 512 | **1072.7** | 1061.9 | **383.9** | — | — | — | 0.58¹ |

¹ the fork's 512-row row against Peregrine's widest pass (128), which is the comparison a user
sees: 0.62 for the 1B and 0.56 for Qwen at one thread, 0.62 and 0.57 at twelve.

Two things in these tables are worth their own line. The **`-ub 32` cliff** in the fork: 34.6
and 40.5 tok/s, half of what `-ub 64` gives, on both models and both runs. That is the width at
which its dispatcher switches from the 4-bit kernel to the requantising one and the 32-row
blocks stop being amortised; `-ub 16` (the 4-bit kernel) is 1.4-1.6× faster than `-ub 32`. And
**Peregrine's chunk 128 is slower than its chunk 64** on the 1B at both thread counts (48.2
against 50.5 at one thread, 310.3 against 328.4 at twelve), which is §6 item 3 still standing.

At equal pass width the two engines are **level**: 0.93-1.03 at 16 and 64 rows, 0.96-0.99 at 8.
Everything the fork gains is at 128 rows and above.

### 10.3 The kernels in isolation, and the check against 10.2

**The fork.** `benchmarks/ik_kernel_probe.cpp` links `libggml.so` from the build PR #177
measured, quantises one real layer's weights (K = 2048, 8192 output rows — a gate or up matrix)
with ggml's own quantiser, repacks them with the fork's own `iqk_repack_tensor` (the call `-rtr`
makes at load, 11-17 ms for 9.4 MB), hands the activations in as F32 and runs the graph's
mul_mat node the way the engine does. **One call is 2048 × 8192 × ny multiply-accumulates** =
134.2 MMAC at ny 8, 1.074 GMAC at 64, 8.59 GMAC at 512. It also asks `iqk_dequant_type` which
of the fork's code paths that shape selects, and prints it:

| weights | ny | path the library takes | run 1 GMAC/s | run 2 |
|---|---|---|---|---|
| Q4_K_R4 | 8 | `q4_k_r4` (the 4-bit kernel) | 57.0 | 61.5 |
| Q4_K_R4 | 16 | `q4_k_r4` | 65.0 | 66.7 |
| Q4_K_R4 | 64 | `q8_k_r16` (32 weight rows requantised to 8 bits, 8-bit kernel) | 52.2 | 55.9 |
| Q4_K_R4 | **512** | `q8_k_r16` | **91.3** | **91.3** |
| Q6_K_R4 | 8 | `q6_k_r4` | 57.3 | 57.6 |
| Q6_K_R4 | 16 | `q6_k_r4` | 60.4 | 57.8 |
| Q6_K_R4 | 64 | `q8_k_r16` | 54.8 | 57.5 |
| Q6_K_R4 | 512 | `q8_k_r16` | 95.6 | 94.4 |

**Peregrine.** `kqbench`'s new layer mode (this branch, `KQBENCH_LAYER=1`) builds the same
matrix as 1024 eight-row tiles (9.4 MB, so it streams the way a whole model's layer does) and
calls the engine's own kernels once per tile: **one pass is the same 2048 × 8192 × ny MACs.**

| weights | 8 acts | 64 acts | 512 acts |
|---|---|---|---|
| Q4_K, `q4k8_gemm_zmm` | 74.3-75.5 | 74.2-75.0 | 71.8-72.7 |
| Q6_K, `q6k8_gemm_zmm` | 50.1-51.2 | 53.8-54.2 | 53.6-53.9 |
| Q5_0, `q5_08_gemm_zmm` | 49.0-51.5 | 50.8-51.8 | 48.8-49.3 |

(The same kernels on a single L1 tile, which is what `kqbench` always reported, read 81.1 / 60.4
/ 55.0 at 16-128 acts; the 8-10 % difference is the 9.4 MB stream, and that is the number to
compare with the fork, whose probe streams the same 9.4 MB.)

**The check.** An engine's end-to-end GMAC/s must not exceed its kernel's isolated rate times
the matmul share, and the share cannot exceed 1. Today, at one thread:

| engine, 1B, 1 thread | end to end | matmul share | required kernel ≥ | probe says | pass? |
|---|---|---|---|---|---|
| fork, ub 512 | 77.7 GMAC/s | 85.1 % (implied by the probe) | 89.8 at share 86.5 % | **91.3** | yes, by 1.7 % |
| Peregrine, chunk 64 | 50.3 | 86.5 % (measured, `PEREGRINE_PROF2`) | 58.2 | **74.2** | yes, by 27 % |
| Peregrine, Qwen chunk 64 | 36.5 | 82.7 % (measured) | 44.1 | **53.5** weighted | yes, by 21 % |

The fork's row is tight on purpose: its matmul share must be at least 85.1 % for its own kernel
to cover its own engine, and Peregrine's measured share is 86.5 %, so the two agree. If the
fork's share were 80 %, its kernel would have to be 97 GMAC/s, above what the probe and the
roof allow — so the probe is neither too low nor a different path from the engine's.

The §9 probe **fails** the same check: 58-67 against a required 89.8. That is the signature of
a probe measuring something else, and it was: it ran the 4-bit kernel for every width and
compared it with an engine that runs the requantising one.

### 10.4 The roof, counted by hand

`peak.c`, one core, minimum of seven timed passes per probe, and every probe's instruction
count per iteration is one instruction per *named* accumulator (an indexed array makes GCC
spill the accumulators, and the first version of this probe measured store-forwarding latency
until that was fixed — the previous sessions' `peak.c` used named accumulators, so this is not
their error).

The clock, three ways, all agreeing: a dependent `add` chain runs at **1.904 and 1.898 G add/s**
(unroll 8 and 16, so it is not a loop-overhead artefact) = 1.90 GHz at one cycle per add; a
dependent `imul` chain runs at 0.627-0.637 G/s = **2.96-3.01 cycles per imul**, the ratio the
architecture has; the TSC reads 1.997 GHz. So the core runs at **1.90 GHz**, 5 % under its TSC
nominal, and does not boost above it on this host.

| probe | instructions/iter | instr/s | MACs/instr | GMAC/s | per cycle |
|---|---|---|---|---|---|
| `vpdpbusd zmm`, 1 accumulator | 1 | 0.48 G | 64 | 30.8 | 0.25 (latency-bound: 4-cycle latency) |
| `vpdpbusd zmm`, 4 | 4 | 1.88 G | 64 | 120.0 | 0.99 |
| `vpdpbusd zmm`, 8 | 8 | 1.90 G | 64 | 121.4 | 1.00 |
| `vpdpbusd zmm`, 16 | 16 | 1.89-1.91 G | 64 | 121-122 | 1.00 |
| `vpdpbusd ymm`, 16 | 16 | 3.83-3.84 G | 32 | 122-123 | 2.01 |
| `vpdpbusd zmm`, 4, operand from an L1 buffer | 4 | 1.93-1.94 G | 64 | 124 | 1.01 |
| `vfmadd512`, 8 | 8 | 3.85-3.89 G | 16 | 62 | 2.02 |

By hand: 1.90 G instructions/s × 64 MACs = **121.6 GMAC/s**; 1 zmm byte dot per cycle, 2 ymm
ones per cycle, 2 zmm FMAs per cycle. **The byte-dot roof is 121-122 GMAC/s per core**, and an
L1 operand costs nothing.

Checked outside the VM: `roof_abs.c` executes a fixed 2.048 × 10¹² MACs (32 G `vpdpbusd`) with
`roof_abs.sh` reading the remote `Date` header before and after: **113.8 GMAC/s, twice**, over
an 18 s window whose ±2 s puts it at 102-128; the guest's own clocks read 111-120 for the same
run. So the roof is 121-122 GMAC/s and nothing on this host reaches 455 with any width.

Both kernels' placement against it: Peregrine's Q4_K is at **61 %** (74.2 of 122), the fork's
4-bit kernel at **47-55 %** (57-67), and the path the fork's engine actually runs at ub 512 at
**75 %** (91.3). §9.3's "both are near the wall" was true of two kernels, one of which the
engine does not use.

### 10.5 The account, again

**One thread, Llama 3.2 1B.** Measured: fork 79.9 tok/s = 77.7 GMAC/s, Peregrine 51.7 tok/s =
50.3 GMAC/s, P ÷ fork 0.65.

| part | fork | Peregrine | P ÷ fork | how |
|---|---|---|---|---|
| weight mix in the matmul | Q4_K 71.8 %, Q6_K 27.7 % | same tensors | — | `PEREGRINE_PROF2` by format |
| kernel, isolated, at each engine's own width | 91.3 (ub 512, `q8_k_r16`) | 74.2 (chunk 64, Q4_K) | 0.81 | §10.3 |
| kernel, isolated, weighted by the format mix | 91.3 | 68.3 (0.718 × 74.2 + 0.277 × 54.2) | 0.75 | §10.3 |
| matmul share of a pass | 85.1 % (the least that covers 77.7) | 86.5 % (measured) | ~1.0 | `PERF2` / arithmetic |
| in-situ kernel rate (end to end ÷ share) | 91.3 | 58.2 | 0.64 | — |
| in-situ efficiency (in situ ÷ isolated) | 100 % | 85 % | 0.85 | — |
| non-matmul: attention / quantise / rest | not measured | 7.7 % / 2.0 % / 3.8 % | — | `PERF2` |
| **end to end** | **77.7** | **50.3** | **0.65** | measured |

So the one-thread gap is **0.75 × 0.85 = 0.64**, and only the second factor is a Peregrine
inefficiency: its kernel runs 15 % slower inside the engine than on the same matrix in
isolation (the 9.4 MB layer probe already pays for the stream, so the remaining loss is the
per-tile output write and the per-layer cold start). The first factor is that the fork's kernel
is a different kernel at a different width.

**Twelve threads, Llama 3.2 1B.** Measured: fork 519 tok/s = 505 GMAC/s, Peregrine 336 tok/s =
327 GMAC/s, P ÷ fork 0.65. Per core: 42.1 against 27.3.

| part | fork | Peregrine | P ÷ fork |
|---|---|---|---|
| matmul share of a pass | 85.1 % (implied) | 82.1 % (measured; barrier wait 10.3 %, attention 7.7 %) | ~1.0 |
| in-situ kernel rate, per core | 49.5 | 33.2 | 0.67 |
| against the single-core isolated kernel | 54 % | 49 % | — |
| thread scaling, 1 → 12 | 6.49× | 6.50× | 1.00 |
| **end to end, GMAC/s** | **505** | **327** | **0.65** |

The 12-thread picture is the one-thread picture: both engines lose about half of their
single-core isolated kernel rate to memory when twelve cores share it, so the ratio stays at
0.65 and **thread scaling is not part of the gap** (6.49× against 6.50×; §1.7's "6.0× against
6.7×" does not reproduce on this branch).

**Is the fork's advantage the pass size?** Partly, and the numbers say which part:

| pk | 1B, 1 thread, P ÷ fork | 1B, 12 threads, P ÷ fork |
|---|---|---|
| 8 | 0.96 | 0.99 |
| 16 | 0.93 | 1.00 |
| 32 | 1.50 | 1.44 |
| 64 | 1.01 | 1.03 |
| 128 | 0.76 | 0.74 |

At 8, 16 and 64 rows the engines are **level**, and Peregrine's kernel on that same matrix is
1.2-1.4× the fork's in isolation (74.2 against 55.9 at 64 rows) — so the kernels are level in
situ and Peregrine's isolated advantage is spent on its in-situ loss. From 128 rows the fork
pulls away, and at 512 it is 0.62. Between 128 and 512 the fork's kernel changes: 55.9 at 64
rows (the 4-bit kernel, its requantise not yet amortised) to 91.3 at 512, while Peregrine's
goes 74.2 → 72.7 (flat, because its kernel already decodes each weight block once per eight
activation rows). **So the fork's 512-row pass is worth nothing by itself to Peregrine; it is
the *licence* for a kernel that requantises weights and is 27 % faster once the requantise is
spread over 512 rows.** Peregrine's own wide pass, at today's kernel, is flat at best and at
chunk 128 actually loses 4 % (one thread) and 5 % (twelve).

**Qwen 2.5 0.5B**, for completeness: one thread 36.5 against 63.3 GMAC/s (0.58), twelve 210 at
chunk 64 against 382 (0.55), and 218 at its widest against the same 382 (0.57). Its matmul at
one thread is 69.3 % Q5_0, 20.4 % Q6_K and 9.5 % Q4_K, and Peregrine's Q5_0 kernel is its
weakest (50.8 isolated against Q4_K's 74.2). Weighted isolated: 53.5 GMAC/s;
in situ 44.1, i.e. 83 % efficiency — the same story as the 1B with a weaker first term. The
fork's Q5_0 path was not probed, so its Qwen kernel rate is **unreconciled** (§10.8).

### 10.6 What to build, again

Ranked by gain over size, against the corrected record. Items 1 and 2 are one job and should be
judged together.

**1. A wide pass that requantises weights: the fork's actual trick (biggest, and the only one
that changes the kernel gap).**

- **What**: at a pass of 128-512 rows, convert each block of 32 weight rows from the 4-bit
  format to an 8-bit one *once*, into a scratch buffer of 64 KiB, and run an int8 tile kernel
  against the whole pass; for narrow passes keep the 4-bit kernel that is already better.
- **Evidence**: the fork's own dispatcher does exactly this above 32 activation rows
  (`iqk_dequant_type` → `q8_k_r16`), and it takes the fork from 55.9 GMAC/s at 64 rows to
  **91.3 at 512** (§10.3), while its 4-bit kernel stays at 57-67. Peregrine's kernel at the same
  shape is 71.8-74.2 and flat. Peregrine at one thread delivers 50.3 GMAC/s and the fork 77.7;
  with a kernel at the fork's 91.3 and Peregrine's measured 85 % in-situ efficiency the same
  engine would deliver 77.6 GMAC/s per core = 79.7 tok/s, **+54 % at one thread**. At twelve
  threads the ceiling is what the fork itself does with that kernel (505 GMAC/s), so the honest
  target is 519 tok/s from 336 today, **+55 %**, P ÷ fork 1.00.
- **Bits**: bit-identical is achievable and is the requirement: a 4-bit code with its per-32
  scale widened to an int8 code with the same per-32 scale is exact (Q4_K's codes are 0-15,
  Q6_K's -32…31, both exact in int8), and the min term must stay the integer term it already is
  (`s3.4.md` §4.4's argument). If the widening rounds, the goldens move and are re-recorded —
  so the kernel has to be written to widen, not to requantise.
- **Size**: 2-3 weeks (a conversion pass, an int8 tile kernel with the same fold order, the
  dispatch at a measured break-even, the bit-identity test).

**2. Make the wide pass pay, and raise the cap (needed for 1; no gain on its own).**

- **What**: `MAX_PREFILL_CHUNK` is 128 (`options.rs:95-98`); the fork's pass is 512. Fix the
  reason chunk 128 loses 4-5 % on the 1B (§6 item 3's scattered write-out), then raise the cap.
- **Evidence**: Peregrine's chunk 128 is 48.2 against 50.5 tok/s at one thread and 310.3
  against 328.4 at twelve, measured twice; its *kernel* is flat to 512 rows (74.2 → 71.8), so
  the loss is in the engine, not the kernel. Item 1 needs a 128-512-row pass to amortise the
  requantise (at 64 rows the fork's requantising path is 25 % *slower* than its 4-bit kernel).
- **Bits**: bit-identical (the chunk is scheduling; `crates/peregrine-models/tests/prefill_chunk.rs`).
- **Size**: 3-5 days on top of §6 item 3.

**3. The Q5_0 kernel (Qwen only).**

- **What**: bring `q5_08_gemm_zmm` (50.8 GMAC/s isolated) to Q4_K's rate (74.2).
- **Evidence**: Q5_0 is 69.3 % of Qwen's matmul time at one thread and 59.2 % at twelve, where
  it is 44.7 % of the whole pass; Q6_K is 12.2 % of the pass at 54.2 GMAC/s, Q4_K 9.7 % at
  74.2. A 1.45× Q5_0 kernel (50.8 → 74.2) takes 44.7 % of the pass down to 30.8 %:
  **+16 % on Qwen's prompt at twelve threads**, which is where its worst ratio is (0.55).
- **Bits**: bit-identical by the same argument; goldens do not move.

**4. Nothing in the Q4_K or Q6_K 4-bit kernels.** Reconfirmed, not merely repeated: Q4_K is
74.2 GMAC/s against the fork's 55.9-66.7, Q6_K 54.2 against 57.5-60.4, at every width either
engine can reach without the requantise. §9's "nothing correct was faster" stands for those
kernels. What was wrong was the conclusion drawn from it: the fork's engine is not running them.

**Not re-measured here**: §6 item 4 (a 512-bit attention score kernel; attention is 7.7 % of
the 1B's pass and 8.8 % of Qwen's at 512 tokens, unchanged) and item 5 (24 threads). Both stand
as written; neither is touched by the correction above.

### 10.7 Bits

No engine code changed in this round. The one source change is a measurement: `kqbench` gains
the `KQBENCH_LAYER` mode (off by default, so its existing output is unchanged) and the probe
`benchmarks/ik_kernel_probe.cpp` is new and links the fork's library. No golden can move, and
none was run; items 1-3 of §10.6 each carry the standard KL gate.

### 10.8 What is not reconciled

- **The fork's own wall-clock rates carry the host's clock error.** Both engines are timed with
  CLOCK_MONOTONIC, which ran 5-8 % fast in this session, so every tok/s above is 5-8 %
  *conservative* in real seconds and every GMAC/s with it. The correction is common to both
  engines, so every ratio stands; the absolute GMAC/s are 5-8 % low. Probes I timed use CPU
  time and MONOTONIC_RAW, and the roof is checked against the host clock.
- **The fork's Q5_0 path** (which is most of Qwen) was not probed, so the fork's Qwen kernel
  rate is inferred from its engine and Peregrine's measured share: 63.3 GMAC/s end to end ÷
  82.7 % = 76.5 GMAC/s per core in situ at one thread. That is below the 91.3 the same build
  gets on a Q4_K matrix at ub 512, which is consistent with a Q5_0 requantise being more work
  than a Q4_K one, but it is an inference and is marked as one.
- **The fork's matmul share inside its own pass** is still not directly measurable (no `perf`
  under WSL2); §10.3's lower bound of 85.1 % and Peregrine's measured 86.5 % are what make the
  one-thread account close.
- **The fork's `-ub 32` cliff** (34.6 against 51.2 tok/s at `-ub 64`, twice) is real but its
  cause is not pinned to a line of its code; it is at the boundary of the two paths.
- **Why the guest's clock scales** (1.6-1.8× early in this session, 1.05-1.08× later, both
  measured against the host and against CPU time) is not something a process inside WSL2 can
  see. It is reported as the hazard it is: absolute rates from different sessions are not
  comparable unless the clock was checked in each.

## 11. Round 4: is the fork's wide pass exact, and what an exact one can reach (S5.2)

- Status: the question settled — **the fork's wide pass is not exact, and an exact one buys
  nothing.** No engine change.
- Date: 2026-11-12 (06:20–08:10 local, UTC+7)
- Host: AMD Ryzen AI 9 HX 370 (Zen 5, 12 cores / 24 threads, AVX-512 F/BW/VL/VNNI/VBMI/BF16),
  WSL2, 11.7 GiB
- Peregrine: `s5.2-exactness` = `origin/main` (`dd6d64c`) + `kqbench`'s wide-pass mode and the
  exact wide kernel it measures; ik_llama.cpp `479074c` (the build of §10, not rebuilt);
  llama.cpp v0.9.0 (the build `docs/targets.md` quotes), both for Part 1 only
- Raw data, the corpus recipe, the end-to-end driver and the clock probe:
  `benchmarks/results/desktop-zen5-2026-11-12-s5.2-exactness/`
- Every ratio below is **Peregrine ÷ the fork**.

§10.6's item 1 proposed converting each block of weight rows to 8 bits once per wide pass, as
the fork does, and said it could be bit-identical "if the kernel widens, not requantises". This
round asks two questions in order: what the fork actually stores, and what an exact wide kernel
can reach. The first decides whether the fork's prompt-processing lead is bought with an
approximation; the second decides whether there is anything for Peregrine to build.

### 11.1 What the fork's 8-bit form stores

The dispatcher is `MulMat::is_dequant_better` (`ggml/src/iqk/iqk_mul_mat.cpp:252`): a repacked
Q4_K matrix (`GGML_TYPE_Q4_K_R4`) takes the 8-bit form for 32 activation rows or more
(`:281`), and a repacked Q6_K matrix for 32 or more (`:285`). `iqk_dequant_type` (`:607`) is
that function exposed, which is how §10.3's probe read it. Above that width, `iqk_mul_mat`
converts the tile's rows through `iqk_convert_repack` (`:665`, `:741`, `:932`, `:992-995`) into
`iqk_convert_kquants_q8X_r8` (`ggml/src/iqk/iqk_gemm_kquants.cpp:3263`), which for Q4_K is
`iqk_convert_q4_k_r4_q8_k_r16` (`:2400`) and for Q6_K `iqk_convert_q6_k_r4_q8_k_r16` (`:2724`).

What Q4_K's conversion stores, per row of the tile and per 256-weight super-block:

1. **The exact f32 weights.** The block's 32-weight sub-blocks are dequantised one at a time
   with the format's own arithmetic, in f32: `w = q·Ld·d + (−dmin·Lm)` with `q` the 4-bit code,
   `Ld`/`Lm` the 6-bit scale and min (`:2444-2463`). The integer products are exact (they are at
   most 945 and are converted to f32 without loss), so `w` is the value the 4-bit kernel
   computes to within two f32 roundings, ~1e-7 relative.
2. **One maximum.** `block_max` is the largest `|w|` over all 256 weights of that row's
   super-block (`:2455`), and
3. **one 8-bit grid per (row, 256 weights)**: `dq = block_max/127` (`:2464`), stored as an f32
   (`set_scale(y[i].d, k, dq)`, `:2466`; the `q8_k_r16` header is `float d[16]`, so this is not
   rounded again — the AVX2 `q8_k_r8` variant's header is f16 and does round it).
4. **One int8 code per weight**: `round(w/dq)` to nearest (`PackerQ80::convert`, `:16-26`, with
   `_MM_ROUND_NEAREST`), saturating packs that cannot saturate because `|w| ≤ block_max`, then
   biased to unsigned for `vpdpbusd` by an xor of `0x80` (`:2471-2474`).

So the fork replaces eight 6-bit scales and eight 6-bit mins per 256 weights with **one f32
scale per 256 weights and an int8 code per weight**, and it gets there by *rounding the weights
onto a new grid*. That is a requantisation, not a widening: `sc·q` reaches 945 and does not fit
int8, so nothing about the original grid survives except that `dq` is a real number.

**The error.** Each weight is off by at most half a step, `dq/2 = block_max/254`, i.e.
**0.394 % of the block's largest weight** (RMS `dq/√12` = 0.227 %, for uniformly distributed
fractional parts; a weight at half the block's maximum is off by at most 0.79 % of itself). This
is not only arithmetic: `requant_error.py` in the results directory repeats the conversion's own
steps on the Q4_K tensors of the two models (a 64th of the super-blocks, 12.98 M weights of the
1B and 0.82 M of Qwen), and finds the mean per-block error at **0.197 %** of the block's maximum
(blocks range 0.126-0.271 %), the worst single weight at **0.394 %** — exactly the bound — and
0.42 % of weights landing exactly on the grid. Q6_K is the same construction (`convert_to_q8_k_r8`,
`ggml/src/iqk/iqk_common.h:638-687`) with one more twist: it computes the maximum of `|q·sc|` in
int16 *before* multiplying by the block's `d`, and if that scaled maximum is below 127 it stores
the codes **exactly** with a scale of `1.0` (`:648-651`) — so Q6_K's conversion is exact for
blocks whose weights are small and rounds to `max/254` otherwise. Q4_K has no such case.

For comparison, the 4-bit format's own rounding is `0.5·d·sc` against a within-sub-block maximum
of about `15·d·sc`: ~3 % per weight. The requantisation adds about an eighth of that in RMS
terms, which is not much per weight and is what the next section measures end to end.

### 11.2 The measurement

40 chunks of 512 tokens of this repository's own prose (the recipe is in the results README; no
chat template was used anywhere), `llama-perplexity -c 512 -t 4`, the base run saving all
logits and the compared run reading them back with `--kl-divergence`.

| engine, comparison | PPL(Q) | PPL(base) | Mean KLD | RMS Δp | same top p |
|---|---|---|---|---|---|
| fork, `-ub 16` vs its `-ub 512` | 38.7359 | 38.7171 | **0.001087** | 0.634 % | **97.873 %** |
| llama.cpp v0.9.0, `-ub 16` vs its `-ub 512` | 38.7618 | 38.7551 | 0.003169 | 1.069 % | 96.559 % |
| llama.cpp v0.9.0 at 512 vs the fork at 512 | 38.7551 | 38.7171 | 0.003018 | 1.046 % | 96.520 % |
| llama.cpp v0.9.0 at 16 vs the fork at 16 | 38.7618 | 38.7359 | 0.002863 | 1.048 % | 96.716 % |

**Is `-ub 512` bit-different from `-ub 16`?** Yes, in both engines, at the first predicted
position: the two log-probability files differ at byte 81945, which is the fourth byte of the
first token's record (the header and the 512×40-token stream end at 81,940). Neither file is a
prefix of the other.

**Peregrine at chunk 16 against chunk 64 is identical**, which is the invariant the change is
about: `prefill_chunk.rs`'s `a_real_model_gives_the_same_tokens_at_any_chunk_size` reads a
200-token prompt in pieces of 16, 32, 64 and 128 on a real Qwen 2.5 0.5B and gets the same 16
greedy tokens every time, and `the_prompt_says_the_same_whatever_the_chunk_size` compares the
KV cache and the logits themselves; both pass on this branch. The fork's own 0.001087 is not
available to Peregrine by construction (ADR 0003).

### 11.3 Is the fork's prompt-processing lead bought with an approximation?

Partly, and the size is 0.0011 nats — **44 % of the distance between Q4_K_M and the unquantised
model** (`docs/targets.md`: 0.00248 mean KLD for this model on wikitext-2) — but pass-width
dependence is not unique to the requantisation and the honest reading is narrower than §10.6
item 1 implied. llama.cpp v0.9.0, which requantises nothing, moves its own logits *more* than the
fork does when `-ub` goes from 16 to 512 (0.003169 against 0.001087, top-1 96.6 % against
97.9 %), so part of every engine's width dependence is not weights at all (llama.cpp's attention
and activation quantisation are shape-dependent). What the fork's 0.001087 does establish is
that its wide path is a *different arithmetic* from its narrow one: a request's numbers would
depend on how the batch was packed, which is the thing ADR 0003 forbids regardless of size. And
the size is not negligible: for the two engines to be at 0.0029-0.0032 from each other at
*either* width, and for the fork's own width change to be 0.0011 of that, means a third of what
separates the two engines on real text is the fork's wide path rather than its kernels. A fork
user who wants the narrow path's numbers gets 0.62 of its prompt speed; a user who wants the wide
path's speed accepts 0.0011 nats. Peregrine chooses the first, and this round finds no exact way
to have the second.

### 11.4 What an exact wide pass can reach: measured

The exact version of the fork's conversion is not a requantisation at all: a Q4_K code is 0-15
and fits a byte, so the widening step is `and`/`shift` per window, copied into a scratch buffer
once per tile, and the per-32 scale multiply and the min term stay exactly as they are. That is
§10.6 item 1's "widen, not requantise", built as `kq4_8`'s two variants in
`crates/peregrine-kernels/src/kqsimd.rs`:

- **(a) widened codes, epilogue unchanged.** `q4k8_widen` copies a tile's 1024 nibble bytes into
  2048 bytes of codes once (`Q4K8_WIDE`), and `q4k8_gemm_zmm_wide` runs today's kernel body over
  the widened buffer: no `and`, no `shift` in the inner loop, the per-32 `vpmulld` on the int32
  partials and #199's `vpdpwssd` min term kept exactly. It is bit-identical to
  `q4k8_gemm_zmm` — asserted at every width from 4 to 40 by
  `the_widened_wide_pass_is_bit_identical_to_the_narrow_one`, and at 8-512 rows by
  `kqbench`'s own check before each timing — so it could be dispatched by pass width without
  moving a golden.
- **(b) the epilogue in its cheapest exact form.** That *is* today's kernel: one `vpmulld` per
  (sub-block, activation row) on int32 partials, the min term through `vpdpwssd` on paired sums,
  and the fold in the reference's order. §9's and #199's measurements (the scale multiply 18 %,
  the min term 10 %) were its budget. There is no cheaper exact arrangement of "one 6-bit
  multiply and one 6-bit min per 32 weights": the partials are 13-bit and the products 18-bit, so
  nothing can be merged into a wider lane, and a 16-bit stage would halve the products per
  instruction. **(a) against (b) is therefore the whole experiment.**

One real layer (2048 × 8192, 1024 eight-row tiles, 9.4 MB streamed, one thread), two runs half
an hour apart, `KQBENCH_LAYER=1 KQBENCH_WIDE=1`:

| activation rows | today (b), GMAC/s | widened (a), GMAC/s | (a) ÷ (b) | run 2 (a) |
|---|---|---|---|---|
| 8 | 73.0 | 61.8 | 0.85 | 64.5 |
| 64 | 75.1 | 73.8 | 0.98 | 71.7 |
| 128 | 73.4 | 73.5 | **1.00** | 73.7 |
| 256 | 71.3 | 70.6 | 0.99 | 72.2 |
| 512 | 71.7 | 72.6 | **1.01** | 70.5 |

The widening alone costs 46-59 GMAC/s of its own work, so it only pays from 64 rows up, and from
64 rows up it pays **nothing**: ±2 %, both signs, in both runs. The nibble split is free. That
is the answer to §10.6 item 1, and the reason is measurable in the instruction mix: per
super-block per eight activation rows the kernel issues 256 `vpdpbusd`, 128 `vpmulld`/`vpaddd`
for the scale, 32 for the min term, ~72 for the fold and ~48 for the split; removing 48 of 536
instructions (9 %) while doubling the bytes read per tile changes nothing, because the byte dot
and the int32 multiplier are the ports that are busy and the `and`/`shift` are not. So an exact
wide kernel's ceiling is today's 74-76 GMAC/s, not the fork's 91: what stands between it and the
roof (121) is the per-32 scale multiply and the min term, 28 % of the kernel by §9's count, and
those cannot be removed without the requantisation. **Nothing ships for Part 2**; the kernel
stays in the crate, documented and tested, so the measurement can be repeated.

### 11.5 The Q5_0 kernel: measured, and why it is where it is

§10.6 item 3 asked for `q5_08_gemm_zmm` (50.8 GMAC/s isolated) to be brought towards Q4_K's
74.2. The measurement this round first explains the gap, and then shows that the obvious fixes
are not fixes. One Q5_0 block is **32 weights**: one f16 scale, no min, 5-bit codes. The kernel
therefore runs its float fold — horizontal sum, the `−16·Σa` bias, the exact int32 sum, one fma —
once per block per activation row, i.e. **eight times more often per weight than Q4_K**, whose
fold is per 256. Per 2048 MACs the Q5_0 kernel issues ~162 instructions against Q4_K's ~72 per
same 2048, and the fold is 40 % of them. It measures 52.0 / 50.1 / 47.9 GMAC/s (8 / 64 / 512
activation rows, layer mode) against Q4_K's 74.0 / 71.8 / 70.6 on the same run.

Two exact variants were built and measured, and neither is faster:

1. **A cheaper decode and a prefetch.** The fifth-bit expansion through `vptestmb` plus one
   masked add of 16 instead of `vpcmpeqb`/`vpmovm2b`/`or` (two instructions fewer per group,
   bit-identical), and the Q8_0 kernel's prefetch distance. A/B on the same binary pair:
   +2.4 % at 64 activation rows, −1.9 % at 512, inside the ±3 % the same binary shows between
   runs. The kernel is not instruction-bound there.
2. **Sixteen output rows per fold.** A Q5_0 block's fold cannot be amortised within a tile, but
   two adjacent eight-row tiles are sixteen output rows, which fit one ZMM: the fold becomes one
   16-lane fma, and the activation broadcast that feeds both tiles is loaded once. It was built,
   asserted bit-identical to the eight-row kernel lane for lane, and measured **slower at every
   width**: 42.4 against 46.8 GMAC/s at 4 activation rows, 43.6 against 54.8 at 16, 42.4 against
   53.6 at 128 — the two weight streams and the doubled accumulator count cost more than the
   shared fold saves. With two activation rows per group (28 registers rather than 44) it was
   worse still, 29.8.

So Q5_0 stays at ~52 GMAC/s isolated (47-49 streamed) and **nothing ships for Part 3** either.
The gap to Q4_K is the format's 32-weight block, not the kernel: amortising the fold needs
sixteen output rows per tile, which the engine's eight-row store, the aarch64 kernel's identical
layout, the pack cache's bytes and `KQ_TILE_ROWS` all fix. That is a stage-5-sized change to the
store, not a kernel change, and it is not attempted here.

### 11.6 End to end, at the defaults

Nothing changed in the engine, so this is the §10.2 picture re-measured in this session's clock
regime (MONOTONIC ~10 % fast, i.e. every tok/s ~10 % conservative in real seconds; the ratio is
untouched). 512 prompt tokens, no decode, three rounds interleaved, medians: the fork at its
`-ub 512` default, Peregrine at its default chunk (the adaptive wide pass).

| model | pk | threads | fork tok/s | Peregrine tok/s | P ÷ fork |
|---|---|---|---|---|---|
| Llama 3.2 1B Q4_K_M | 64 | 1 | 78.8 | 50.4 | 0.639 |
| Llama 3.2 1B Q4_K_M | 64 | 12 | 527.9 | 326.2 | 0.618 |
| Qwen 2.5 0.5B Q4_K_M | 64 | 1 | 171.2 | 97.3 | 0.568 |
| Qwen 2.5 0.5B Q4_K_M | 64 | 12 | 1 095.2 | 574.3 | 0.524 |

These agree with §10.2 (0.65 and 0.62, 0.58 and 0.55-0.58) to within this host's noise; the
one-thread numbers are 4-9 % *worse* than §10.2's and the fork's are 2-6 % *better*, which is
the load average (2-5, other tenants) rather than a change in the engine. With no engine change
in this round, **Peregrine is at 0.64 of ik_llama.cpp on Llama 3.2 1B and 0.57 on Qwen 2.5 0.5B
on prompt processing at the default settings**, one thread, and 0.62 and 0.52 at twelve.

### 11.7 Bits

No engine code changed. The source changes are `kqsimd.rs`'s `Q4K8_WIDE`/`q4k8_widen`/
`q4k8_gemm_zmm_wide` (a kernel nobody dispatches), the `Q4Codes` split of the Q4_K group body
that lets the narrowed and the widened source share one implementation (the narrow path is
byte-for-byte the code it was), a test that the two agree bit for bit, and `kqbench`'s
`KQBENCH_WIDE` mode (off by default, so its existing output is unchanged). No golden can move
and none was run for Part A; the two structural Q5_0 variants were reverted after measuring.

### 11.8 What is not reconciled

- **Where llama.cpp's own pass-width difference comes from** (0.003169, larger than the fork's)
  is not pinned. It requantises no weights, so it is attention or activation quantisation, and
  §10.3's method (a probe on one matmul) cannot see it. The comparison in 11.3 uses it only as
  an upper bound on what is *not* the fork's requantisation.
- **The requantisation's effect on the matmul alone** was not isolated from attention: a probe
  that called the fork's 4-bit and 8-bit paths on the same matrix would change the activation
  quantisation too (`q8_2_x4` against `q8_k`), and the fork does not export its conversion
  entry point. The per-weight error in 11.1 is therefore arithmetic from the source, confirmed
  on sampled real weights (`requant_error.py`) rather than measured through the engine.
- **The Q5_0 kernel's 42 GMAC/s paired variant** is reported, not explained: no `perf` is
  available under WSL2, and the guess (two weight streams and two accumulators per activation
  row spilling) was not confirmed.
- **Decode was measured and is not judged here.** At 128 tokens after a 16-token prompt the two
  engines read 0.34 (1B, one thread) and 0.21 (1B, twelve), which is far below anything the
  prompt-path account predicts and outside this round's question; it is recorded in `e2e.txt`
  as a flag, not as a result.
