# Q5_0 takes the 8-bit form in the fast prompt mode on ARM (S5.4h)

[`docs/design/s5.4-fast-prompt.md`](../../../docs/design/s5.4-fast-prompt.md) §5h: S5.4f made
which formats take the 8-bit prompt form a per-tier table and gave Q5_0 the form, its build path
and an AVX2 builder, but left aarch64 at S5.4e's `[Q4_K, Q6_K]` because it was written on an
AVX2 machine where Q5_0 is level (0.99-1.03×) and 0.88× at Qwen 2.5's 896-wide rows. This round
measures Q5_0's form on the ARM host's two NEON caps, puts it in the aarch64 arm of the table
where the measurement says so, records the aarch64 Q5_0 fast goldens, and re-measures the mode
end to end on Qwen 2.5 0.5B (the file this is for, 165.1 MiB of Q5_0) with Llama 3.2 1B (no
Q5_0) as the guard.

- Date: 2026-11-12, 19:10-22:40 local (UTC+7).
- Host: DGX Spark (Lenovo ThinkStation PGX), aarch64, 20 cores in two clusters: 10 ×
  Cortex-X925 at 3.9 GHz (CPUs 5–9 and 15–19) and 10 × Cortex-A725 at 2.8 GHz, 121 GB unified
  memory, Linux `6.17.0-1008-nvidia`, governor `performance`, rustc 1.99.1. Kernel tier
  `aarch64-neon-i8mm`; the lower caps are the same binary under `PEREGRINE_ISA`.
- Peregrine: branch `s5.4h-fast-prompt-q5-arm`, stacked on `origin/s5.4f-form-where-it-wins`
  (PR #253, not yet merged), which is `origin/main` at `4d5fff2`'s merge base. The *before*
  build is that branch as it stands (`/var/tmp/pg-s54h-before-target/release/peregrine`), the
  *after* build is this branch; both were compiled from this host.
- Models, both regular files under `~/models`:

  | model | bytes | SHA-256 |
  |---|---|---|
  | `Qwen2.5-0.5B-Instruct-Q4_K_M.gguf` | 397 808 192 | `6eb923e7d26e9cea28811e1a8e852009b21242fb157b26149d3b188f3a8c8653` |
  | `Llama-3.2-1B-Instruct-Q4_K_M.gguf` | 807 694 464 | `6f85a640a97cf2bf5b8e764087b1e83da0fdb51d7c9fab7d0fece9385611df83` |

  Qwen 2.5 0.5B's multiplied weights are 28.1 MiB Q4_K (ffn_down, twelve layers, 4864 wide, a
  whole number of super-blocks) + 40.9 MiB Q6_K (ffn_down, the other twelve layers, 4864 wide) +
  **165.1 MiB Q5_0 (132 tensors, every one 896 wide)** + 1.4 MiB Q8_0 (attn_v, twelve layers,
  896 wide). Its 137.9 MiB token embedding is also Q8_0 but is gathered, never multiplied (the
  file ties its output head to it, so the one logits matmul is a single row per pass). Llama 3.2
  1B is 445.5 MiB Q4_K + 317.1 MiB Q6_K and **no Q5_0 at all**, which is why it is the guard.
- llama.cpp: v0.9.0, commit `cd959bae99ff4bed22bef2604f45241c09bbb8c8`, the harness cache build
  at `~/.cache/peregrine-bench/llama.cpp-v0.9.0`, CPU only (`CUDA_VISIBLE_DEVICES=""`).
- ik_llama.cpp: `master` at `479074ccac2ac94f06b6d24a9ce26f0f9e9aa6ae` (2026-11-11), built
  CPU-only, Release, `-march=armv9-a+dotprod+i8mm+bf16+sve2`, at its defaults.

## The host is shared

A co-tenant ran services on the same host for the whole window. The 1-minute load average was
**3.5–28.5** and is recorded beside every cell (`runs.jsonl` has one line per Peregrine cell and
its load; `llama.txt`/`ik.txt` carry a load per engine block; `load.txt`, `decode.txt` and
`sanity.txt` carry the rest). Consequences, stated plainly:

- the **one-thread cells are stable**: round-to-round ratios move by ≤1 % on every one-thread
  cell (Qwen 512/1 `after ÷ exact` is 2.71–2.77 over five rounds, 4 096/1 1.71–1.72 over three;
  Llama 512/1 1.00–1.01, 4 096/1 1.00–1.01), so they are read as measured;
- the **ten-thread cells are noisy**: one Qwen 512/10 round moved its exact arm 172.8–620.7
  tok/s for the same binary (a 3.6× swing) while the round's other arms moved with it. The
  medians are over 10 rounds (Qwen 512/10) or 5 (512-token cells) or 3 (all 4 096-token cells),
  every round interleaved (exact, fast-before, fast-after) so a drift inside a round hits every
  arm; the **within-round ratio** is the claim, not the absolute tok/s;
- every cell is marked **loaded** below (1-minute load above 4), including the one-thread ones.
  That is the host, not the change: the same marks apply to the before build.

## How the cells were measured

```
# Peregrine, one cell, one round (the two fast builds alternate inside it):
target/release/peregrine bench -m <model> --prompt-tokens <512|4096> --tokens 1 \
    --threads <1|10> [--fast-prompt] --json            # after build
/var/tmp/pg-s54h-before-target/release/peregrine bench ... --fast-prompt --json   # before
# threads 1 -> taskset -c 5; threads 10 -> taskset -c 5-9,15-19

# the other engines, at their defaults, one repetition per round:
CUDA_VISIBLE_DEVICES="" taskset -c <cpus> \
  ~/.cache/peregrine-bench/llama.cpp-v0.9.0/build/bin/llama-bench -m <model> -p <512|4096> -n 0 -t <1|10> -r 1
taskset -c <cpus> ~/.cache/peregrine-bench/ik_llama.cpp/build/bin/llama-bench \
  -m <model> -p <512|4096> -n 0 -t <1|10> -r 1

# the kernel table (S5.4h's new mode of the existing example), 3 runs per row:
PEREGRINE_ISA=<neon-dotprod|neon-i8mm> taskset -c 5-9,15-19 \
  target/release/examples/neonbench table 8 20 <1|10> both
```

`--tokens 1` makes the prefill the measurement (`prefill_tok_s`); a prompt pass with no slot
decoding is the wide one, so the fast build runs its own default wide chunk (512 on aarch64) and
the exact build runs today's (64). `pg-round.sh` runs one round's three Peregrine arms,
`eng-round.sh` one round of both engines, `eng-one.sh` a single engine when a pair does not fit
a foreground command; `summarize.py` rebuilds every table from `runs.jsonl`, `llama.txt` and
`ik.txt`.

## The kernel table: does Q5_0's form beat the exact kernel on ARM?

The rule S5.4f set: a format takes the form on a tier where the form is **at least 1.10× the
exact kernel from 64 activation rows up, on the shapes real files have**. The shapes here are a
whole 2048-weight row (8 super-blocks, `whole`) and Qwen 2.5's 896-wide row (28 Q5_0 blocks =
three and a half super-blocks, so the form pads a fourth, `row896`).

GMAC/s, medians of three runs of `neonbench table`; one row is eight output rows of a tile over
the stated super-blocks against the stated number of activation rows; each worker holds its own
tile (`neonbench-table-{1,2,3}.txt` have all three and the load averages, 4.4–6.6).

**`aarch64-neon-i8mm` (`smmla`), one worker:**

| format | shape | 8 rows | 64 rows | 128 rows | 512 rows |
|---|---|---:|---:|---:|---:|
| Q4_K exact | whole | 123.1 | 124.0 | 124.2 | 123.9 |
| Q4_K form | whole | 309.1 | 312.2 | 312.9 | 308.4 |
| Q6_K exact | whole | 75.6 | 76.0 | 76.1 | 75.6 |
| Q6_K form | whole | 309.7 | 312.5 | 312.8 | 307.0 |
| Q5_0 exact | whole | 55.3 | 55.3 | 55.2 | 55.0 |
| Q5_0 form | whole | 309.1 | 311.1 | 312.7 | 309.8 |
| Q5_0 exact | row896 | 54.4 | 54.8 | 54.8 | 54.6 |
| Q5_0 form | row896 | 262.6 | 266.3 | 267.7 | 268.3 |

Form ÷ exact at 64 rows: **Q4_K 2.52×, Q6_K 4.11×, Q5_0 5.63× (whole) and 4.86× (896-wide).**

**`aarch64-neon-dotprod` (`sdot`), one worker:**

| format | shape | 8 rows | 64 rows | 128 rows | 512 rows |
|---|---|---:|---:|---:|---:|
| Q4_K exact | whole | 78.1 | 77.9 | 78.0 | 77.5 |
| Q4_K form | whole | 193.0 | 186.7 | 200.0 | 189.7 |
| Q6_K exact | whole | 54.8 | 54.6 | 54.4 | 54.2 |
| Q6_K form | whole | 191.4 | 200.5 | 185.9 | 189.6 |
| Q5_0 exact | whole | 34.1 | 33.9 | 33.9 | 33.6 |
| Q5_0 form | whole | 191.0 | 186.5 | 186.8 | 190.5 |
| Q5_0 exact | row896 | 34.1 | 33.9 | 33.7 | 33.6 |
| Q5_0 form | row896 | 160.7 | 165.8 | 173.8 | 172.5 |

Form ÷ exact at 64 rows: **Q4_K 2.40×, Q6_K 3.67×, Q5_0 5.50× (whole) and 4.89× (896-wide).**

**Ten workers** (aggregate GMAC/s of ten independent copies; the engine splits output rows
across workers, so this is the rate a wide prompt pass asks for):

| tier | format, shape | 8 rows | 64 rows | 128 rows | 512 rows |
|---|---|---:|---:|---:|---:|
| i8mm | Q4_K exact / form | 917.3 / 3045.4 | 1225.0 / 3103.0 | 1227.9 / 3082.6 | 1221.4 / 3040.6 |
| i8mm | Q6_K exact / form | 604.6 / 3073.9 | 753.0 / 3095.7 | 753.3 / 3091.3 | 751.0 / 3057.0 |
| i8mm | Q5_0 exact / form, whole | 363.0 / 3068.2 | 548.0 / 3105.3 | 547.3 / 3097.5 | 540.1 / 3048.5 |
| i8mm | Q5_0 exact / form, row896 | 538.1 / 2569.2 | 542.9 / 2641.4 | 544.7 / 2659.7 | 539.6 / 2647.7 |
| dotprod | Q4_K exact / form | 593.0 / 1855.5 | 775.5 / 1857.6 | 774.6 / 1840.5 | 770.7 / 1829.2 |
| dotprod | Q6_K exact / form | 541.3 / 1853.4 | 543.8 / 1862.8 | 542.6 / 1864.0 | 542.7 / 1836.8 |
| dotprod | Q5_0 exact / form, whole | 338.1 / 1847.1 | 337.1 / 1861.3 | 337.7 / 1858.9 | 332.6 / 1836.7 |
| dotprod | Q5_0 exact / form, row896 | 333.7 / 1600.0 | 337.5 / 1616.2 | 336.3 / 1620.8 | 334.5 / 1618.1 |

Ten-worker form ÷ exact at 64 rows: Q4_K 2.53 / 2.40 (i8mm / dotprod), Q6_K 4.11 / 3.43,
Q5_0 5.67 / 5.52 (whole) and **4.87 / 4.79 (896-wide)**. The 8-row cells with ten workers are
the only ratios far from the others (Q4_K 3.32×, Q6_K 5.08×, Q5_0 8.45×): the exact kernels do
not saturate ten independent copies at eight activation rows (the form does, it is ALU-bound),
which is a property of the exact kernel's small-batch efficiency, not of the form.

## The decision, and the padded tail

**Q5_0 joins aarch64's form table where the multiply is a NEON body**, `neon-i8mm` and
`neon-dotprod`: 4.79–4.91× its exact kernel at Qwen's real 896-wide shape and 5.50–5.67× on
whole rows, from eight activation rows up, at one and ten workers, on both caps — far above the
1.10× bar that kept it off AVX2. The `neon` and `scalar` caps keep `[Q4_K, Q6_K]`: their
multiply *is* the scalar twin, which is no faster than the exact kernel for any format, so a
third format would add ~283 MiB of form on Qwen and buy nothing. That is what S5.4f's rule says,
and the tier line now reads `Q4_K, Q6_K and Q5_0 (NEON smmla)` on this host and `Q4_K and Q6_K
(scalar twin)` under `PEREGRINE_ISA=neon` and `=scalar`.

**The padded fourth super-block costs about 13 % and does not need a fix here.** The form of a
896-wide Q5_0 row holds 1024 codes — a quarter of the last super-block is zero weights meeting
zero activation codes — so the kernel walks 8 groups of 32 where 4 are real. Measured on the
same tier and width (i8mm, 64 rows, one worker): 5.63× on the whole 2048-weight row against
4.86× on the 896-wide one, i.e. the padded form is 13.7 % slower per useful weight, exactly the
1024/896 = 1.14 of traffic. **Skipping the four zero groups would be bit-identical** — the
integer sums are exact in any grouping and the epilogue is per super-block, which keeps its real
half — and the kernel test now proves it: it rewrites the padded groups' codes with a non-zero
pattern and gets the same bits, because every activation code they would meet is zero
(`a_q5_0_row_that_stops_inside_a_super_block_is_the_twins_bits`). It is not worth a shared
layout change: on ARM the form is 4.8× ahead there, and on AVX2 the whole-block ratio is only
1.00×, so removing the padding would bring its 896-wide cell from 0.88× to about 1.00× — still
below the 1.10× bar, and the x86 goldens and the AVX-512 SDE jobs would carry the risk for no
tier's table entry.

## End to end: fast ÷ exact before and after

`bench`, `--tokens 1`, the same two binaries alternating inside each round. "before" is
S5.4f's build (Q5_0 exact), "after" is this one. Medians of interleaved rounds; the load range
of the cell's rounds is beside it, and every cell is **loaded**.

| model | cell | exact | fast before | fast after | after ÷ exact | before ÷ exact | load |
|---|---|---:|---:|---:|---:|---:|---|
| Qwen 0.5B | 512/1 | 144.7 | 162.5 | **394.4** | **2.73** | 1.12 | 4.9–5.3 |
| Qwen 0.5B | 512/10 | 321.1 | 475.7 | **811.3** | **2.53** | 1.48 | 5.5–6.5 |
| Qwen 0.5B | 4096/1 | 94.6 | 102.8 | **162.5** | **1.72** | 1.09 | 4.6–7.0 |
| Qwen 0.5B | 4096/10 | 224.7 | 366.5 | **520.0** | **2.31** | 1.63 | 7.5–12.7 |
| Llama 1B | 512/1 | 94.6 | 202.3 | **202.8** | **2.14** | 2.14 | 7.8–11.5 |
| Llama 1B | 512/10 | 284.4 | 593.4 | **730.5** | **2.57** | 2.09 | 7.7–9.9 |
| Llama 1B | 4096/1 | 63.4 | 99.2 | **99.5** | **1.57** | 1.56 | 6.2–7.9 |
| Llama 1B | 4096/10 | 177.5 | 448.8 | **422.2** | **2.38** | 2.53 | 11.6–14.0 |

Within-round ratios (the noise-resistant reading; `summarize.py` prints the medians above):

- **Qwen 512/1**: before ÷ exact 1.12–1.14, after ÷ exact 2.71–2.77 over five rounds;
  after ÷ before 2.42–2.43 in every round.
- **Qwen 4096/1**: before 1.09 every round, after 1.71–1.72, after ÷ before 1.58 every round.
- **Qwen 512/10** (ten rounds): after ÷ exact 2.37, 3.07, 3.19, 4.48, 3.76, 3.03, 1.31, 4.39,
  3.23, 2.45; before ÷ exact 1.29, 2.49, 1.32, 1.33, 2.24, 1.88, 0.80, 2.43, 2.01, 1.01.
- **Qwen 4096/10**: after ÷ exact 2.73, 2.15, 2.01; before 1.76, 1.63, 1.31.
- **Llama** is level in every one-thread round (512/1 after ÷ before 1.00–1.01 twice and 1.00
  three times; 4 096/1 1.00–1.01) and within the ten-thread noise (512/10 after ÷ before
  1.02, 1.14, 1.30, 0.63, 1.49; 4 096/10 0.99, 0.81, 1.10). The 1B has no Q5_0 tensor, so the
  path it runs is byte-for-byte the before build's.

**Acceptance.** On Qwen 0.5B the fast mode improves on **every** cell — 1.12→2.73, 1.48→2.53,
1.09→1.72, 1.63→2.31 (medians), and the one-thread cells by 1.58× (4 096 tokens) and 2.42× (512) in every round — and
no cell is slower than before in any round where the two fast builds alternate. The 1B guard is
level (one-thread ratios 1.00–1.01).

## Fast ÷ llama.cpp and fast ÷ ik_llama.cpp

Ratios from the raw medians of the table above and the engines' 3-round medians
(`llama.txt`, `ik.txt`; both engines at their defaults, one repetition per round).

| model | cell | llama.cpp | ik | fast ÷ llama.cpp | fast ÷ ik |
|---|---|---:|---:|---:|---:|
| Qwen 0.5B | 512/1 | 61.7 | 306.9 | **6.39** | **1.29** |
| Qwen 0.5B | 512/10 | 310.5 | 574.2 | **2.61** | **1.41** |
| Qwen 0.5B | 4096/1 | 50.5 | 188.1 | **3.22** | 0.86 |
| Qwen 0.5B | 4096/10 | 259.9 | 568.9 | **2.00** | 0.91 |
| Llama 1B | 512/1 | 30.1 | 118.1 | **6.74** | **1.72** |
| Llama 1B | 512/10 | 177.2 | 711.2 | **4.12** | **1.03** |
| Llama 1B | 4096/1 | 25.9 | 86.3 | **3.84** | **1.15** |
| Llama 1B | 4096/10 | 173.1 | 574.1 | **2.44** | 0.74 |

Below 1.0, named: **Qwen ÷ ik at 4 096 tokens, 0.86× (one thread) and 0.91× (ten)**. It is not
an exact-weights leftover: after this step 99.4 % of Qwen's *multiplied* bytes take the form
(234.1 of 235.5 MiB), and the Q8_0 that stays exact is 1.4 MiB of attn_v. It is the form's own
bytes at a long prompt: the resident form of those matrices is 385.9 MiB against the exact
235.5 MiB (**1.64×**), and ik_llama.cpp streams the file's own 0.69 bytes per Q5_0 weight
instead. At 512 tokens the same cell is 1.29× (one thread) and 1.41× (ten); the 4 096-token
pass is where the extra bytes bite. **Llama 1B ÷ ik at 4 096 tokens/10 threads, 0.74×** is the
same cell S5.4e and S5.4f had below 1.0 (ik's own repacking and threaded prefill win there); it
is unchanged by this step.

## What the form costs at load

`peregrine inspect` with `PEREGRINE_FAST_PROMPT=1` for the size; `/usr/bin/time -v` on an 8-token
bench (load plus a tiny pass), three warm runs each, ten threads (`form-size.txt`, `load.txt`;
`%M` is KiB, shown here in MiB):

| model | form before | form after | wall before (median) | wall after (median) | peak RSS before | peak RSS after |
|---|---:|---:|---:|---:|---:|---:|
| Qwen 2.5 0.5B | 102.9 MiB | **385.9 MiB** | 0.77 s | 0.86 s | 789.6 MiB | 1 079.6 MiB |
| Llama 3.2 1B | 957.0 MiB | 957.0 MiB | 1.05 s | 1.03 s | 1 897.5 MiB | 1 897.5 MiB |

Qwen's form grows by **+283 MiB** for 165.1 MiB of Q5_0 weights: a padded 896-wide row is 1.16
bytes per weight against the file's 0.6875, so the padding is 1.69× the exact Q5_0 bytes. Load
costs +0.09 s and +290 MiB peak; the 1B is unchanged (no Q5_0). This is the price of the form
being resident; S5.4a's per-pass conversion remains the alternative it did not take.

## Decode after a fast prompt

Decode rows are exact in both modes; what could move is the cost of the extra resident bytes.
One thread (core 5), 32 decode tokens after a 512-token prompt, modes alternated each round,
three rounds (`decode.txt`):

| model | exact (tok/s) | fast (tok/s) | fast ÷ exact |
|---|---:|---:|---:|
| Qwen 2.5 0.5B | 43.79, 44.75, 45.01 (median 44.75) | 44.57, 44.30, 44.56 (44.56) | **0.996×** |
| Llama 3.2 1B | 26.16, 26.11, 25.96 (26.11) | 25.80, 26.02, 25.96 (25.96) | **0.994×** |

Level within 0.5 % on both models.

## Sanity: 64 greedy tokens on Qwen 0.5B, fast against exact

`peregrine run -m <Qwen> --max-tokens 64 --temperature 0 --threads 10 [--fast-prompt]`
(`sanity.txt`):

- *"Write one sentence about the sea."* — both 40 tokens; they first differ at the third word
  ("with its vast" against "a vast"). The mode re-rounds the prompt, so a divergence this early
  is what the trade costs; the sentence stays on topic.
- *"List three colours, comma separated."* — **byte-identical**: "Red, Yellow, Blue".

The accuracy report that puts a number on the first case is S5.4c's and is not this round's.

## What is left exact on Qwen 2.5 0.5B, and what that means

| weights | size | takes the form? |
|---|---:|---|
| Q4_K (ffn_down, 12 layers, 4864 wide) | 28.1 MiB | yes (S5.4e/f) |
| Q6_K (ffn_down, 12 layers, 4864 wide) | 40.9 MiB | yes (S5.4e/f) |
| Q5_0 (132 tensors, 896 wide) | 165.1 MiB | **yes, from this step** |
| Q8_0 attn_v (12 layers, 896 wide) | 1.4 MiB | no — an 8-bit format with its own kernel |
| Q8_0 token embedding (gathered, and the tied one-row logits head) | 137.9 MiB | n/a to prompt passes |
| F32 | 0.3 MiB | n/a |

**99.4 % of the bytes a prompt pass multiplies (234.1 of 235.5 MiB) now take the form.** The
Q8_0 that remains exact is one 0.12 MiB matrix per layer. The 139.3 MiB of "Q8_0" that S5.4e's
account counted as exact is almost all the token embedding, which is gathered; it is why the
file's byte count overstates what a prompt pass multiplies, and why the 4 096-token cells' gap
to ik_llama.cpp is the form's 1.64× bytes and not an untouched format.

## What was not done

- The padded-tail skip (§ above): measured, proven bit-identical by test, not implemented —
  it would be a shared layout/dispatch change for a 13 % gain on one shape that already runs
  at 4.8×, and it would not lift AVX2's 0.88× over the 1.10× bar.
- Q8_0's form.
- The accuracy report (S5.4c).
- Nothing on Q5_0's goldens: `record-goldens.yml` on `macos-15` recorded
  `aarch64-neon-dotprod-macos/llama-Q5_0-fast.golden`, and it is byte-identical to the Linux
  `aarch64-neon-dotprod` one (the run's own record-then-verify pair passed; its recorded-files
  list held only that file, so no existing macOS golden changed).
