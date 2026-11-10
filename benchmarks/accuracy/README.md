# Task-accuracy gate (roadmap S3.3)

Compares Peregrine's answers with a full-precision reference (BF16 via vLLM) and with
llama.cpp on the same GGUF, on three public tasks, and checks that decoding is invariant to
batching and speculation. It is a runner for any OpenAI-compatible chat server, so the same
prompt bytes and the same items go to `peregrine serve`, `llama-server` and `vllm serve`.

## What is measured

| Task | Items | Metric | Prompt |
|---|---|---|---|
| GSM8K (`openai/grade-school-math`, test) | 250 | numeric match against the `####` answer | 0-shot, "solve step by step, final line `#### <number>`" |
| MMLU (`cais/mmlu`, `all`/test) | 500 | letter match | 0-shot, one question with choices and "reply with only the letter" |
| HumanEval+ (`evalplus/humanevalplus`) | 163 | pass@1, greedy, sandboxed execution | 0-shot, complete the function in one code block |

The three tasks use one user turn and greedy decoding (`temperature: 0`, no sampling
extensions), so every engine sees the same bytes. MMLU is 0-shot on purpose: a five-shot
`Answer: X` transcript made the 1B chat models answer the whole list or ignore the format,
and the single-question form scored higher for every gate model in the protocol check.
HumanEval/32 is excluded from the pinned HumanEval+ revision: its plus test unpacks the
candidate's float return value, so not even the dataset's canonical solution passes it
(`run.py selftest` reports 163/163 without it).

Datasets are fetched at fixed revisions, hashed, and cached outside the repository
(`PG_ACC_CACHE`, default `/var/tmp/pg/work/datasets`). The fetch step needs `pyarrow`
(MMLU is distributed as parquet); the runner, scorers and report are standard library only.

## Running

```sh
# 1. datasets (needs pyarrow; see requirements.txt)
python3 benchmarks/accuracy/run.py fetch --cache /var/tmp/pg/work/datasets

# 2. one server at a time, run the tasks, then score
python3 benchmarks/accuracy/run.py run --engine peregrine \
    --url http://127.0.0.1:8299 --model-name qwen25-05b \
    --model-file /var/tmp/pg/models/acc-gate/qwen25-05b-q4km.gguf \
    --tasks gsm8k,mmlu,humaneval --limit gsm8k=150,mmlu=300,humaneval=100 \
    --concurrency 8 --out results/2026-11-09-dgx-spark/subset/qwen25-05b-peregrine

python3 benchmarks/accuracy/run.py score \
    --run results/2026-11-09-dgx-spark/subset/qwen25-05b-peregrine \
    --python /path/to/venv/bin/python      # interpreter for the HumanEval sandbox (numpy)

# 3. merge scored runs into the report and the gate verdict
python3 benchmarks/accuracy/run.py report \
    --runs results/2026-11-09-dgx-spark/full results/2026-11-09-dgx-spark/subset \
    --reference vllm --baseline llama_cpp \
    --llama-tolerance-pp 4 --reference-budget-pp 10 \
    --out results/2026-11-09-dgx-spark/report.md
```

Runs are resumable: `<task>.jsonl` is checkpointed (atomically) as items complete, and
existing items are skipped, so an invocation that hits a command-time limit on a slow or
shared host continues where it stopped. `--limit` takes `task=n` pairs; the selection is a
seeded shuffle (seed 20261109), so the same limits always yield the same items, and a
smaller limit is a prefix of a larger one.

`run.py selftest` checks the scorers against the datasets' own answers: the canonical
HumanEval+ solutions must pass the sandbox (163/163) and the GSM8K/MMLU extractors must read
their reference values.

### Invariance

```sh
# same prompt set, decoded alone, batched, and with speculation
python3 benchmarks/accuracy/run.py invariance --url http://127.0.0.1:8299 \
    --phase alone --prompts 520 --max-tokens 48 \
    --out results/2026-11-09-dgx-spark/invariance/qwen25-05b
python3 benchmarks/accuracy/run.py invariance --url http://127.0.0.1:8299 \
    --phase batched --group 8 --prompts 520 --max-tokens 48 \
    --out results/2026-11-09-dgx-spark/invariance/qwen25-05b
# restart the server with --speculate, then --phase speculation
python3 benchmarks/accuracy/run.py compare \
    --phases results/.../alone.jsonl results/.../batched.jsonl results/.../speculation.jsonl \
    --out results/.../compare.md
```

The prompt set is a fixed mix of GSM8K, MMLU and HumanEval+ prompts, written once to
`prompts.jsonl` and reused by every phase. `compare` hashes each reply and reports any
prompt whose bytes differ.

## Scoring and the sandbox

- **GSM8K**: the last number in the reply, compared with the reference answer (1e-4
  relative tolerance).
- **MMLU**: the letter after an explicit "answer is/answer:", else the first standalone
  A–D.
- **HumanEval+**: the first fenced Python block is run against the dataset's own tests in a
  fresh subprocess: `-I` interpreter, empty environment, fresh directory, CPU/address-space/
  file-size limits, wall-clock timeout, process group killed on timeout, and an audit hook
  that denies socket and subprocess events. The host does not allow unprivileged network
  namespaces, so the audit hook is the no-network guard; test code is trusted (pinned
  dataset), generated code is not.

## Report

`report.md` has the per-run accuracies with Wilson 95% intervals, the paired difference
against llama.cpp and the BF16 reference on the *same items* (exact McNemar p-value and the
paired difference CI), and the gate verdict. The paired statistics matter here: with
n ≤ 300 items, unpaired point differences of a few points are inside the noise.

## Layout

```
benchmarks/accuracy/
├── run.py           # subcommands: fetch, run, score, selftest, report, invariance, compare
├── data.py          # pinned downloads, normalisation, deterministic selection
├── prompts.py       # the three prompt templates (single source of truth)
├── client.py        # minimal OpenAI-compatible chat client (urllib)
├── scorers.py       # GSM8K / MMLU / HumanEval+ scoring
├── sandbox.py       # subprocess sandbox for generated code
├── requirements.txt # pyarrow (fetch), numpy (HumanEval+ tests)
└── results/         # committed raw outputs, scores, manifests, report
```

Each run directory holds `manifest.json` (engine, model file hashed and path-neutralised,
server command, dataset revisions and hashes, host), `<task>.scored.jsonl` (the reply, its
hash, usage, timings and the per-item verdict) and `summary.json`. The committed results keep
the scored files (which carry the replies) and drop the duplicate unscored captures.
No absolute home-directory paths are recorded: model paths are stored relative to
`/var/tmp/pg` or as basenames.
