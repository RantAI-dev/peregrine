# Bluefin (H2): llama.cpp at its faster flash-attention setting

The bluefin re-run after **#100** found that llama.cpp v0.9.0's default `-fa auto` resolves to
flash attention *on* on the CPU backend, and that on this host that path is much slower than
`-fa off` for single-user decode at long context (13.0 vs 29.1 tok/s on Llama 3.2 1B at 4096).
Every ratio we published from the #97 bluefin baseline compared against that default.

This folder re-measures the #97 suite so that each row is compared against **llama.cpp's faster
flash-attention setting for that row**: `-fa off` for single-user decode, and the default
`-fa auto` where it is faster (prefill and multi-user throughput). The default number is kept
beside it as a secondary note. No ratio below is taken against the slower setting.

- Date: 2026-11-08. New runs 13:50-16:01 local (06:50-09:01 UTC); the default-setting numbers
  are the #97 baseline from the same day, 10:25-12:40 local (03:25-05:40 UTC).
- Host: AMD Ryzen 3 PRO 4350G (Zen 2), 4 cores / 8 threads, AVX2/FMA, 13.5 GiB RAM,
  8 GiB swap. Benchmark threads: 4 (one per physical core). Governor `powersave`, unchanged.
- Peregrine: the engine source is identical to `origin/main` `9bcf524` (verified: no diff under
  `crates/`, `Cargo.toml` or `Cargo.lock` between `origin/main` and the worktree). The worktree
  HEAD was the #100 branch merge `2a227d2`, built `release` with `cargo build --release -p
  peregrine`, `rustc 1.99.1 (58cf8f9c3 2026-10-13)`.
- llama.cpp: v0.9.0 (`cd959ba`), prebuilt at `/var/tmp/llama.cpp/build/bin` (CMake Release,
  `GGML_NATIVE=ON`). Same binary as #97 and #100.
- Models, both from `~/models`, SHA-256 verified against the #97 manifest:
  - `Llama-3.2-1B-Instruct-Q4_K_M.gguf`, 807694464 bytes,
    `6f85a640a97cf2bf5b8e764087b1e83da0fdb51d7c9fab7d0fece9385611df83`;
  - `Qwen2.5-0.5B-Instruct-Q4_K_M.gguf`, 397808192 bytes,
    `6eb923e7d26e9cea28811e1a8e852009b21242fb157b26149d3b188f3a8c8653`.
- Settings: contexts 512 and 4096, 3 interleaved rounds per engine, 256 greedy decode tokens per
  request, server mode, prefix-cache defeat by per-request nonce, prompts calibrated per engine
  and model (every measured prompt within 5% of target).

## The policy

For every published row:

1. measure llama.cpp with flash attention at its default (`auto`, which is on here) and with
   `-fa off`;
2. compare Peregrine against whichever of the two is **faster on that row**;
3. show the default-setting number only as a secondary note.

The harness records the setting in every manifest (`settings.llama_flash_attn`,
`engine.flash_attn`), so a run cannot be misread. `--llama-fa` defaults to `auto` so older
invocations reproduce their original configuration.

## Method and command splitting

Each round was its own `-r 1` invocation, as in #97, to keep every command under the 10-minute
cap. At 512 tokens one command ran both engines; at 4096 tokens each engine ran separately, and
the Llama 3.2 1B 4096 rounds were split by user count (`--user-counts 1,4` and `10`). Samples
were then merged into one manifest per engine and model with the harness's own medians and
half-range spreads, exactly the shape a single `-r 3` invocation writes; the raw per-round
manifests are not committed but the merged JSONs carry every individual sample.

The only change to llama.cpp's command line is the flash-attention flag:

```sh
LLAMA_SERVER_BIN=/var/tmp/llama.cpp/build/bin/llama-server \
python3 benchmarks/run.py -m ~/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
  -e llama_cpp -c 4096 -r 1 -n 256 -t 4 --llama-fa off \
  --user-counts 1,4 -o /tmp/bench-fa/raw
```

## Quiet host

The host was quiet apart from its usual background (`localsearch-3` about 2% CPU, `gnome-shell`
and session processes about 1% total, `kswapd0` about 2-3%). The load average immediately before
each command ranged from 1.08 to 3.99 (1 min), the high end being the benchmark's own four
worker threads still winding down from the previous command. Swap used was flat at
3859-3864 MiB of 8191 MiB across the whole session; it did not grow during any run. No process
was stopped or killed for these runs.

## Results

`Peregrine ÷ llama.cpp` is against the faster setting named in the "faster" column.
Medians, tok/s unless noted; spreads are in [`report.md`](report.md) and the JSON manifests.

### Llama 3.2 1B Instruct (Q4_K_M)

| Context | Metric | Peregrine | llama.cpp default | llama.cpp `-fa off` | Faster | P ÷ faster |
|---|---|---|---|---|---|---|
| 512 | Single decode | 46.06 | 39.51 | **42.01** | off | **1.10×** |
| 512 | Prompt processing | 197.34 | **188.27** | 173.57 | default | **1.05×** |
| 512 | 4 users total | 53.82 | **39.55** | 34.22 | default | **1.36×** |
| 512 | 10 users total | 55.18 | **37.36** | 23.88 | default | **1.48×** |
| 4096 | Single decode | 36.85 | 13.02 | **29.25** | off | **1.26×** |
| 4096 | Prompt processing | 144.59 | **141.23** | 122.19 | default | **1.02×** |
| 4096 | 4 users total | 7.65 | **5.36** | 3.10 | default | **1.43×** |
| 4096 | 10 users total | 7.73 | **5.07** | not completed | default | **1.52×** |

### Qwen 2.5 0.5B Instruct (Q4_K_M)

| Context | Metric | Peregrine | llama.cpp default | llama.cpp `-fa off` | Faster | P ÷ faster |
|---|---|---|---|---|---|---|
| 512 | Single decode | 52.87 | 74.92 | **77.41** | off | **0.68×** |
| 512 | Prompt processing | 263.19 | **302.91** | 277.11 | default | **0.87×** |
| 512 | 4 users total | 70.66 | **65.24** | 55.03 | default | **1.08×** |
| 512 | 10 users total | 71.72 | **68.69** | 42.52 | default | **1.04×** |
| 4096 | Single decode | 45.20 | 41.04 | **51.97** | off | **0.87×** |
| 4096 | Prompt processing | 198.23 | **235.98** | 192.91 | default | **0.84×** |
| 4096 | 4 users total | 9.97 | **10.88** | 5.02 | default | **0.92×** |
| 4096 | 10 users total | 9.81 | **10.13** | not completed | default | **0.97×** |

### Time to first token at 10 users (s, lower is better)

| Model | Context | Peregrine | llama.cpp default | llama.cpp `-fa off` | Faster | P ÷ faster |
|---|---|---|---|---|---|---|
| Llama 3.2 1B | 512 | 18.96 | **22.02** | 35.30 | default | **0.86×** (P faster) |
| Llama 3.2 1B | 4096 | 192.54 | **185.22** | not completed | default | **1.04×** (P slower) |
| Qwen 2.5 0.5B | 512 | 14.76 | **13.65** | 22.35 | default | **1.08×** (P slower) |
| Qwen 2.5 0.5B | 4096 | 151.42 | **114.25** | not completed | default | **1.33×** (P slower) |

## What the numbers say

- **The bluefin 4096 single-user win on Llama 3.2 1B is 1.26×, not 2.82×.** Against llama.cpp's
  default 13.02 tok/s it is 2.83×, but that default is the broken-on-this-host flash-attention
  path. With `-fa off`, llama.cpp does 29.25 tok/s and Peregrine's 36.85 is 1.26×.
- **The 512 single-user win on Llama 3.2 1B shrinks from 1.15× to 1.10×.** `-fa off` is also
  faster at 512 (42.01 vs 39.51), so the faster setting is used there too.
- **The Qwen 0.5B 4096 single-user win disappears.** It was 1.10× against the default (45.2 vs
  41.0); with `-fa off` llama.cpp decodes at 51.97 tok/s and Peregrine's 45.20 is **0.87×**, a
  loss. At 512 the Qwen loss widens slightly, 0.70× to 0.68×.
- **The 10-user claim survives.** llama.cpp's flash-attention path is *faster* for multi-user
  throughput, not slower: at 512 tokens `-fa off` drops 10-user throughput from 37.36 to 23.88
  tok/s on the 1B, and from 68.69 to 42.52 on the 0.5B. The faster setting is therefore the
  default, and the bluefin numbers are 1.48× (1B, 512), 1.43× (1B, 512, 4 users), 1.52× (1B,
  4096, 10 users), 1.04× (0.5B, 512, 10 users) and 1.08× (0.5B, 512, 4 users). The claim was not
  inflated by the flash-attention default.
- **Prompt processing is unchanged in practice.** The default is faster than `-fa off` here too,
  so the like-for-like ratios are the #97 ones: 1.05× and 1.02× on the 1B, 0.87× and 0.84× on
  the 0.5B.

### The 10-user, 4096, `-fa off` rows could not be measured

With `-fa off`, the 10-user phase at 4096 tokens does not finish inside the 10-minute command
cap on this 4-core host, even with prompt calibration skipped: a dedicated attempt was still
running at 9.5 minutes and was stopped. The default-setting phase completes in about 8.4 minutes
of measured wall time on the 1B (5.07 tok/s aggregate, 2560 tokens; the whole split command was
9m23 in #97) and about 4.2 minutes on the 0.5B (10.13 tok/s). The `-fa off`
setting is already about 2× slower at 4 users at 4096 (3.10 vs 5.36 on the 1B; 5.02 vs 10.88 on
the 0.5B) and about 1.6× slower at 10 users at 512, so the default is unambiguously the faster
setting for these two rows. They are marked "not completed" rather than estimated; the ratios
use the default.

## Files

- `2026-11-08-064755-llama.cpp-Qwen2.5-0.5B-Instruct-Q4_K_M.json` and the three others: merged
  per-engine, per-model manifests with every raw sample, the machine record and the engine
  builds, with `settings.llama_flash_attn` and `engine.flash_attn` recording the setting.
- `report.md`: the standard harness report, **against `-fa off` only**. Its ratio column is not
  the published one where the default is faster; use the tables above.
