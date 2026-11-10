# OpenAI compatibility suite

`compat.py` checks a running Peregrine server (or any OpenAI-compatible server) against the parts of
the OpenAI API Peregrine claims. It is Python 3 with the standard library only — no pip install, no
`openai` package. It starts `peregrine serve` itself unless `--base-url` is given, and prints one
line per check. (`peregrine serve` is the same server as the `peregrine-server` binary; the facade
build is the one that includes every weight format.)

```sh
cargo build --release -p peregrine

# SmolLM2-360M-Instruct Q4_K_M: the model CI uses.
python3 tests/compat/compat.py \
  --model "$HOME/models/SmolLM2-360M-Instruct.Q4_K_M.gguf" \
  --peregrine-bin target/release/peregrine

# A thinking model: also checks reasoning_content vs content, and streaming.
python3 tests/compat/compat.py \
  --model "$HOME/models/Qwen3-1.7B-Q4_K_M.gguf" \
  --peregrine-bin target/release/peregrine --thinking
```

Exit status is 0 when nothing failed. A `DEVIATION` is a documented difference from the OpenAI
specification (or from llama-server) and does not fail the run; `SKIP` means the check needs
something the server or model does not have. `--json-out results.json` writes the same results for a
report. Add `COMPAT_VERBOSE=1` to print a traceback on failure and the server log at the end.

Options that matter:

| Flag | Default | |
|---|---|---|
| `--base-url URL` | — | Use a server that is already running (no process is started) |
| `--model PATH` | — | GGUF file for the server the suite starts |
| `--peregrine-bin PATH` | `./target/release/peregrine` | Binary to run |
| `--threads N` | `min(8, CPUs)` | Server threads |
| `--ctx N` | `2048` | Server context |
| `--slots N` | `4` | Parallel slots (the concurrency check sends 8) |
| `--thinking` | off | The model has a reasoning channel; run the reasoning checks |
| `--engine` | `peregrine` | `peregrine` or `llama-server` (labels the report; checks what the server returns) |
| `--timeout S` | `180` | Per-request timeout |

## Test models

| Model | Upload | Bytes | SHA-256 |
|---|---|---|---|
| SmolLM2-360M-Instruct Q4_K_M | <https://huggingface.co/mradermacher/SmolLM2-360M-Instruct-GGUF/resolve/main/SmolLM2-360M-Instruct.Q4_K_M.gguf> | 270590976 | `ade7aee1e5bd7c3b8f1fedafc61a1a0be31e8afa64e006607cd202323e2ccce0` |

The same upload is the A1 oracle fixture in `docs/dev/testing.md`, so its hash is already pinned
there. CI caches it under `~/.cache/peregrine-compat` and checks the hash before every run.
The thinking check uses any Qwen 3 GGUF (the suite only needs a non-empty `reasoning_content`);
Qwen3-1.7B Q4_K_M is what the recorded comparison used. It is skipped in CI because it is a second,
larger download.

## Comparing with llama-server

Build or install `llama-server`, then start it on the same file and point the suite at it:

```sh
llama-server -m "$HOME/models/SmolLM2-360M-Instruct.Q4_K_M.gguf" \
  --host 127.0.0.1 --port 8311 -t 6 -c 2048 -np 4 --jinja --metrics &
python3 tests/compat/compat.py --base-url http://127.0.0.1:8311 --engine llama-server
```

Use `-np` at least as large as the `n` in the tests (3, and 8 for the concurrency check queues
requests; `-np 4` is enough). The observed differences are recorded in
[`docs/compat.md`](../../docs/compat.md#against-llama-server); the short version is that Peregrine
has no `/v1/completions`, while llama-server does not reproduce a `seed` on its first request after
start and its `reasoning_format: "none"` leaves `<think>` in the content.

## Adding a check

A check is a function that returns a detail string, or raises:

- `Fail` — a broken contract (the run exits non-zero);
- `Deviation` — a known difference from the OpenAI spec;
- `Skip` — the check does not apply (no thinking model, model stopped too early for a stop string).

Register it in `main()` with `run("name", check_function)`. Use `need(condition, message)` for
assertions, `chat()` for chat requests, and `stream_chat()` for SSE (it parses `data:` lines and
`[DONE]`). Add the difference to `docs/compat.md` if the check records a deviation.
