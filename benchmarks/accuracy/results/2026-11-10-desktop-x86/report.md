# Task-accuracy gate

Generated 2026-11-09T21:52:09+00:00 from `full, reference`.

Greedy decoding, fixed prompts (`prompts.py`), the same item set per task across engines.
Confidence intervals are Wilson 95%; paired differences use the discordant pairs only.

| run | engine | task | accuracy | n | 95% CI | mean s/item |
|---|---|---|---|---|---|---|
| llama32-1b-llama | llama_cpp | GSM8K | 44.0% | 250 | 38.0-50.2% | 15.77 |
| llama32-1b-llama | llama_cpp | HumanEval+ | 31.9% | 163 | 25.2-39.4% | 16.73 |
| llama32-1b-llama | llama_cpp | MMLU | 30.4% | 500 | 26.5-34.6% | 3.75 |
| llama32-1b-peregrine | peregrine | GSM8K | 46.8% | 250 | 40.7-53.0% | 15.07 |
| llama32-1b-peregrine | peregrine | HumanEval+ | 33.1% | 163 | 26.4-40.7% | 14.63 |
| llama32-1b-peregrine | peregrine | MMLU | 29.4% | 500 | 25.6-33.5% | 2.77 |
| qwen3-17b-llama | llama_cpp | GSM8K | 82.4% | 250 | 77.2-86.6% | 30.54 |
| qwen3-17b-llama | llama_cpp | HumanEval+ | 63.8% | 163 | 56.2-70.8% | 25.89 |
| qwen3-17b-llama | llama_cpp | MMLU | 42.6% | 500 | 38.3-47.0% | 7.80 |
| qwen3-17b-peregrine | peregrine | GSM8K | 80.4% | 250 | 75.0-84.8% | 30.19 |
| qwen3-17b-peregrine | peregrine | HumanEval+ | 64.4% | 163 | 56.8-71.4% | 24.07 |
| qwen3-17b-peregrine | peregrine | MMLU | 43.4% | 500 | 39.1-47.8% | 5.92 |
| llama32-1b-vllm | vllm | GSM8K | 48.8% | 250 | 42.7-55.0% | 2.33 |
| llama32-1b-vllm | vllm | HumanEval+ | 33.1% | 163 | 26.4-40.7% | 1.89 |
| llama32-1b-vllm | vllm | MMLU | 30.4% | 500 | 26.5-34.6% | 0.16 |
| qwen3-17b-vllm | vllm | GSM8K | 80.4% | 250 | 75.0-84.8% | 4.64 |
| qwen3-17b-vllm | vllm | HumanEval+ | 62.6% | 163 | 54.9-69.6% | 3.68 |
| qwen3-17b-vllm | vllm | MMLU | 47.0% | 500 | 42.7-51.4% | 0.38 |

## Paired differences

`Δ llama.cpp` is the accuracy difference on the items both runs answered, with the 95% CI of the paired difference and the exact McNemar p-value. `Δ ref` is the same against the full-precision reference. `n` is the number of paired items.

| run | task | n | Δ ref | Δ llama.cpp | paired 95% CI (pp) | McNemar p | gate |
|---|---|---|---|---|---|---|---|
| llama32-1b-peregrine | GSM8K | 250 | -2.0 | +2.8 | [-1.8, +7.4] | 0.311 | pass |
| llama32-1b-peregrine | HumanEval+ | 163 | +0.0 | +1.2 | [-1.2, +3.6] | 0.625 | pass |
| llama32-1b-peregrine | MMLU | 500 | -1.0 | -1.0 | [-3.2, +1.2] | 0.473 | pass |
| qwen3-17b-peregrine | GSM8K | 250 | +0.0 | -2.0 | [-6.5, +2.5] | 0.487 | pass |
| qwen3-17b-peregrine | HumanEval+ | 163 | +1.8 | +0.6 | [-3.4, +4.6] | 1.000 | pass |
| qwen3-17b-peregrine | MMLU | 500 | -3.6 | +0.8 | [-1.2, +2.8] | 0.557 | pass |

## Gate verdict

All 6 checks pass: within ±4 pp of llama.cpp (paired CI or non-significant McNemar) and no more than 10 points below the full-precision reference.


## Host, build and method

- **Host:** desktop, AMD Ryzen AI 9 HX 370 (24 threads, 11 GB RAM), WSL2, x86_64.
- **Peregrine:** `origin/main` `1c93345ac8d12711df023123b4411a7a4cfef8d3`, release profile. `main` does not
  contain `benchmarks/accuracy`, so the harness came from `origin/acc-gate` (`d38a5f0`); the two differ in no
  crate (`git diff origin/main origin/acc-gate -- crates/` is empty), so the binary is a main build. Kernel tier
  `x86_64-avx2-avx512-vnni-gfni-vbmi` (AVX-512), default threads (24), default `--slots 4`.
- **llama.cpp:** v0.9.0-dev, commit `cd959ba`, CPU build, `llama-server -c 16384 -t 12 -np 8 --jinja`.
  The first attempt used `-c 32768` (as on the 128 GB DGX) and swapped on this 11 GB host; 16384 is ample for
  512-token replies.
- **Protocol:** greedy, one user turn, thinking off (`chat_template_kwargs.enable_thinking=false`), the full sets
  (GSM8K 250, MMLU 500, HumanEval+ 163) with the same seed and therefore the same item ids as the 2026-11-09
  reference. MMLU is 0-shot: `prompts.py` ignores the shots parameter, and `mmlu_shots: 5` is recorded only to
  match the reference manifest.
- **HumanEval sandbox:** 30 s per item. The dataset's own canonical solution for HumanEval/139 needs 16.5 s here
  (6.5 s on the DGX), so the default 10 s would fail a canonical solution on this host. The sandbox interpreter
  preloads numpy before the audit hook is installed: CPython 3.12's numpy import calls `ctypes.PyDLL(None)`, whose
  `ctypes.dlopen` event the hook denies on this host. The hook still guards the candidate program (no sockets, no
  subprocesses); the change only makes the pinned HumanEval+ tests runnable.
- **Model files:** Llama-3.2-1B-Instruct-Q4_K_M sha256 `6f85a640…`; Qwen3-1.7B-Q4_K_M sha256 `b139949c…`.
  The brief names bartowski as the source of the Qwen3 file, but `b139949c…` is the unsloth upload; bartowski's
  Q4_K_M hashes to `72c5c3cb…`. The stated hash was treated as authoritative and the file was downloaded from
  the repository whose LFS object matches it.
- The committed run directories keep the scored files (replies, verdicts, usage, manifest) and drop the
  duplicate unscored captures and item lists, as in the 2026-11-09 results. Recorded paths are relative or
  neutralised (`<external>/…`, `<sandbox>`).

## Gate verdict per model and task

The harness verdict above follows the committed definition (Peregrine no more than 10 points behind
*llama.cpp's own distance* from BF16). The brief states an absolute bound (at most 10 points below BF16 on the
same items). Both readings pass, six of six:

| model | task | Peregrine | llama.cpp | Δ vs llama.cpp (paired) | McNemar p | BF16 ref | Δ vs BF16 | verdict |
|---|---|---|---|---|---|---|---|---|
| Llama-3.2-1B | GSM8K | 46.8 | 44.0 | +2.8 | 0.311 | 48.8 | -2.0 | pass |
| Llama-3.2-1B | MMLU | 29.4 | 30.4 | -1.0 | 0.473 | 30.4 | -1.0 | pass |
| Llama-3.2-1B | HumanEval+ | 33.1 | 31.9 | +1.2 | 0.625 | 33.1 | +0.0 | pass |
| Qwen3-1.7B | GSM8K | 80.4 | 82.4 | -2.0 | 0.487 | 80.4 | +0.0 | pass |
| Qwen3-1.7B | MMLU | 43.4 | 42.6 | +0.8 | 0.557 | 47.0 | -3.6 | pass |
| Qwen3-1.7B | HumanEval+ | 64.4 | 63.8 | +0.6 | 1.000 | 62.6 | +1.8 | pass |

Largest absolute distance from llama.cpp is 2.8 points, every exact McNemar p ≥ 0.31, and the largest distance
from BF16 is 3.6 points. No task fails.

## Qwen3-1.7B HumanEval+: what the Spark subset showed

The 2026-11-09 notes list Peregrine at 47.5 and llama.cpp at 56.4 on this model/task. Those two numbers are not
from the same item set: Peregrine's 47.5 is 19/40, llama.cpp's 56.4 is 92/163. On the same 40 items the Spark
llama.cpp run scored 20/40 (50.0), a paired difference of -2.5 points — the value in that report's own table.

Re-scoring the exact same 40 items on this host gives Peregrine 21/40 (52.5) and llama.cpp 21/40 (52.5). Two
items that failed on the ARM host pass here (HumanEval/37, /40), one llama.cpp item flips the other way
(/59), and one Peregrine item passes where this host's llama.cpp fails (/69). At n = 40 a near-zero paired
difference has a 95 % interval of roughly ±13 points, so the ARM numbers are consistent with sampling noise in
both engines; the -8.9 point framing was a subset-vs-full comparison, not a measured regression.

The full x86 set makes the same point with 4× the items: 105/163 (64.4) for Peregrine against 104/163 (63.8) for
llama.cpp, 11 discordant items (6 Peregrine-only, 5 llama.cpp-only), exact McNemar p = 1.000.

**Item-level diagnosis (all 11 discordant items).** No structural difference was found:

- every item produced a fenced Python block on both engines, so code-block extraction is not the cause;
- `finish_reason` distribution is identical on both sides (161 `stop`, 2 `length`), so max_tokens truncation is
  not the cause and no reply lost its answer to the 512-token cap;
- neither engine returned `reasoning_content` (thinking off on both), so reasoning-tag handling is not the cause;
- both engines read the same GGUF, hence the same tokenizer and chat template, and their outputs agree
  byte-for-byte from the first token up to a divergence point inside the generated body (common prefix 10-819
  characters of 163-1063). Example, HumanEval/39:
  `if is_prime(a) and is_fibonacci(a)` (Peregrine) against `if is_fibonacci(a) and is_prime(a)` (llama.cpp).
- 60 of 163 replies are byte-identical; the divergences are greedy choices between near-tied next tokens that
  then lead to different but often both-correct programs.

The cause is therefore numerical near-ties in greedy decoding (different activation quantisation and summation
order in the two engines), not the chat template, stop sequences, tokenizer, `max_tokens`, or reasoning tags.
There is no defect to fix and no gate failure on the full set.

## Invariance, Llama-3.2-1B, AVX-512 build

520 fixed prompts (208 GSM8K, 156 MMLU, 156 HumanEval+, `max_tokens 48`) through `peregrine serve` with the
default 24 threads and 4 slots: decoded one at a time (433 s), batched eight at a time (298 s), and alone again
against a server started with `--speculate`. **520/520 replies are byte-identical in all three phases**
(`invariance/llama32-1b/compare.md`).
