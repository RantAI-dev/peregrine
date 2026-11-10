# 2026-11-10 — desktop x86, AVX-512 (S3.3 release run)

Full task-accuracy gate on the x86 desktop, run to close the release blocker that the 2026-11-09
DGX Spark subset run left open.

- **Peregrine:** `origin/main` `1c93345ac8d12711df023123b4411a7a4cfef8d3`, release build, AVX-512
  kernel tier, default threads (24), default slots (4). The harness itself comes from `origin/acc-gate`
  (`d38a5f0`) because `main` does not contain `benchmarks/accuracy` yet; its crate tree is byte-identical
  to `main`.
- **Models:** Llama-3.2-1B-Instruct-Q4_K_M (`6f85a640…`) and Qwen3-1.7B-Q4_K_M (`b139949c…`).
- **llama.cpp:** v0.9.0-dev `cd959ba`, `llama-server -c 16384 -t 12 -np 8 --jinja`.
- **Items:** full sets — GSM8K 250, MMLU 500, HumanEval+ 163 — greedy, thinking off.

```
report.md                       gate verdict, paired differences, host notes and analysis
full/<model>-peregrine/         Peregrine runs (scored replies, manifest, summary)
full/<model>-llama/             same-host llama.cpp runs, same items
reference/<model>-vllm/         copies of the 2026-11-09 BF16 (vLLM) reference runs
invariance/llama32-1b/          520 prompts alone / batched / with speculation
```

Result: six of six model×task checks pass (|Δ| ≤ 2.8 points vs llama.cpp, all McNemar p ≥ 0.31,
|Δ| ≤ 3.6 points vs BF16); 520/520 invariance replies byte-identical. See `report.md`.

Reproduce (one task per invocation; the harness resumes):

```sh
python3 benchmarks/accuracy/run.py run --engine peregrine --url http://127.0.0.1:8299 \
    --model-id qwen3 --model-name qwen3-17b --model-file /tmp/gold-models/Qwen3-1.7B-Q4_K_M.gguf \
    --tasks mmlu --limit mmlu=500 --concurrency 8 --out <dir>
python3 benchmarks/accuracy/run.py score --run <dir> \
    --python /tmp/opencode/venv2/bin/python-sandbox --exec-timeout 30
python3 benchmarks/accuracy/run.py report --runs full reference \
    --reference vllm --baseline llama_cpp \
    --llama-tolerance-pp 4 --reference-budget-pp 10 --out report.md
```

Notes: the HumanEval sandbox needs a 30 s per-item timeout here (the canonical HumanEval/139 solution
takes 16.5 s on this host) and a numpy preload before the audit hook (CPython 3.12's numpy import raises
the `ctypes.dlopen` event the hook denies). The scored files keep the replies and verdicts with host paths
neutralised (`<external>/…`, `<sandbox>`); the duplicate raw captures and item lists are not committed,
matching the 2026-11-09 layout.
