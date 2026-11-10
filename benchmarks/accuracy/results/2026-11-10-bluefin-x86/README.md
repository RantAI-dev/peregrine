# Task-accuracy run: bluefin (x86, AVX2), 2026-11-10

The S3.3 release-blocker run on x86 hardware, for the two gate models the desktop
carries: Qwen2.5-0.5B-Instruct and gemma-3-1b-it, both Q4_K_M. Full item sets
(GSM8K 250, MMLU 500, HumanEval+ 163), greedy, thinking off.

`report.md` is the harness's own report (accuracy, Wilson intervals, paired
differences, exact McNemar p, gate verdict). This file records the provenance and
reads the verdict.

## Provenance

| item | value |
|---|---|
| Peregrine build | `origin/main` at `1c93345ac8d12711df023123b4411a7a4cfef8d3` (`peregrine 0.1.0`), `cargo build --release -p peregrine` |
| Harness | `origin/acc-gate` at `d38a5f0` (PR #136); `origin/main` did not yet contain `benchmarks/accuracy` when the run started |
| Host | bluefin, AMD Ryzen 3 PRO 4350G (4 cores / 8 threads), 13 GB RAM, kernel 7.2.7-200.fc44, no GPU |
| ISA | `x86_64-avx2` (no AVX-512) |
| Qwen2.5-0.5B-Instruct Q4_K_M | sha256 `6eb923e7d26e9cea28811e1a8e852009b21242fb157b26149d3b188f3a8c8653` |
| gemma-3-1b-it Q4_K_M | sha256 `12bf0fff8815d5f73a3c9b586bd8fee8e7b248c935de70dec367679873d0f29d` |
| Datasets | pinned revisions from `data.py`; task hashes identical to the DGX Spark run (`gsm8k` `3482e5e4…`, `mmlu` `447f9a23…`, `humaneval` `53dd3a4b…`) |
| BF16 reference | vLLM runs from `../2026-11-09-dgx-spark/full/{qwen25-05b,gemma3-1b}-vllm` |
| llama.cpp baseline | this host, `llama-server` 0.9.0-dev (`cd959ba`), same GGUF files |

Engines as invoked:

- **Peregrine**: `peregrine serve -m <model> --port 8299` with the harness's
  settings — 8 threads (the default here), ctx 8192, 4 slots, `--concurrency 8`,
  greedy, `enable_thinking: false`, `reasoning_effort: "none"`. The two models
  have no thinking mode, so that last field is a no-op; llama.cpp did not receive
  it.
- **llama.cpp**: `llama-server -m <model> -c 32768 -t 8 -np 8 --jinja` (the DGX
  command line with this host's default thread count), `--concurrency 8`.
  gemma-3-1b MMLU and HumanEval+ were captured with `-c 16384` instead: the host
  is shared, and its memory pressure made the larger per-slot KV cache thrash.
  Per-slot context stayed at 2048 tokens, well above the longest prompt plus the
  512-token cap, and greedy decoding does not depend on the KV allocation size.

Model files are recorded by basename; no host paths are kept in these artifacts.

## Gate verdict

Per model/task: paired against llama.cpp on the same items (`|Δ| <= 4 pp`, or a
non-significant exact McNemar test) and against the BF16 reference (`<= 10 pp`
beyond llama.cpp's own distance from it).

| model | task | Peregrine | llama.cpp | Δ llama.cpp (pp) | paired 95% CI | McNemar p | Δ BF16 (pp) | verdict |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | GSM8K | 42.4% (106/250) | 43.6% (109/250) | -1.2 | [-5.1, +2.7] | 0.690 | -0.4 | **pass** |
| Qwen2.5-0.5B | MMLU | 33.6% (168/500) | 32.0% (160/500) | +1.6 | [-0.6, +3.8] | 0.215 | -2.6 | **pass** |
| Qwen2.5-0.5B | HumanEval+ | 32.5% (53/163) | 31.3% (51/163) | +1.2 | [-2.6, +5.0] | 0.754 | +7.4 | **pass** |
| gemma-3-1b | GSM8K | 45.2% (113/250) | 47.2% (118/250) | -2.0 | [-6.4, +2.4] | 0.473 | -8.8 | **pass** |
| gemma-3-1b | MMLU | 39.8% (199/500) | 39.6% (198/500) | +0.2 | [-2.1, +2.5] | 1.000 | +0.8 | **pass** |
| gemma-3-1b | HumanEval+ | 34.4% (56/163) | 32.5% (53/163) | +1.8 | [-2.1, +5.8] | 0.549 | -4.3 | **pass** |

All six checks pass; no FAIL analysis is required. The widest gap is gemma-3-1b
on GSM8K (-8.8 pp against BF16), where llama.cpp on the same GGUF is itself
-6.8 pp from BF16 on the same items, i.e. within the protocol's reference-relative
budget; the two engines disagree on 13 items in Peregrine's favour and 18 in
llama.cpp's, all ordinary mid-computation arithmetic divergences (different
intermediate numbers, not formatting, stop-sequence or tokenizer drift — both
engines emit `#### <number>` and the same GGUF bytes are used).

## Decoding invariance (Qwen2.5-0.5B, x86)

520 prompts (the fixed GSM8K/MMLU/HumanEval+ mix), 48 new tokens each, decoded
alone, batched (group 8), and with `--speculate`:

- **520 / 520 byte-identical** across all three phases (`invariance/qwen25-05b/compare.md`).
- Batched throughput 3.13 prompts/s vs 1.77 alone; speculation matches alone's
  throughput on this model.

## Notes

- The host is shared and was under memory pressure (13 GB, swap full) during part
  of the run; a few llama.cpp requests ran tens of seconds per token. That affects
  `mean s/item` in `report.md`, not the answers: greedy decoding is unchanged, and
  every task's captures completed with 0 server errors and 0 empty replies.
- llama.cpp is not bit-stable across hosts in practice: the same GGUF scored
  40.4% (DGX, `-t 16`) vs 43.6% (here, `-t 8`) on Qwen2.5-0.5B GSM8K. That is why
  the gate compares against a same-host llama.cpp run; the DGX llama.cpp runs stay
  in the repository as the reference's companion, not as this run's baseline.
- Reproduce: `python3 benchmarks/accuracy/run.py report --runs <these runs plus
  the two DGX *-vllm dirs> --reference vllm --baseline llama_cpp
  --llama-tolerance-pp 4 --reference-budget-pp 10`.
