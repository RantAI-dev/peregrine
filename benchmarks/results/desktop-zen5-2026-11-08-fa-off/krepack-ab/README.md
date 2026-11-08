# PEREGRINE_KREPACK=1 vs the default, Llama 3.2 1B, single user (A/B)

Five rounds, alternating the two arms (default, repacked, repacked, default, default,
repacked, ... — see the timestamps in `raw/`), one invocation per arm per round, 512 and
4096 tokens, 12 threads, 256 greedy decode tokens, the same harness settings as the
baseline folder above. The repacked arm ran with `--opt KREPACK=1` (the CLI form of
`PEREGRINE_KREPACK=1`), which the manifest records in `settings.engine_args`.

The question: whether to turn the repacked Q4_K kernels on by default for the AVX-512 tier,
where they are currently off. Decode must not get slower.

| Round | Local time | Arm | Decode 512 tok/s | Decode 4096 tok/s | Prompt 512 tok/s | Prompt 4096 tok/s |
|---|---|---|---|---|---|---|
| 1 | 20:47 | default | 65.50 | 47.31 | 178.63 | 135.41 |
| 1 | 20:49 | `KREPACK=1` | 60.17 | 46.75 | 233.39 | 155.61 |
| 2 | 20:50 | `KREPACK=1` | 61.71 | 46.22 | 230.55 | 157.66 |
| 2 | 20:52 | default | 64.99 | 46.77 | 189.93 | 130.94 |
| 3 | 20:54 | default | 64.00 | 46.70 | 185.48 | 129.21 |
| 3 | 20:56 | `KREPACK=1` | 65.00 | 46.25 | 236.84 | 154.67 |
| 4 | 20:58 | `KREPACK=1` | 63.24 | 46.82 | 234.70 | 157.26 |
| 4 | 20:59 | default | 63.09 | 46.56 | 178.22 | 129.21 |
| 5 | 21:01 | default | 62.64 | 47.87 | 186.18 | 130.56 |
| 5 | 21:03 | `KREPACK=1` | 63.97 | 48.90 | 233.88 | 156.05 |

Medians of the five rounds:

| Test | Default | `KREPACK=1` | Change |
|---|---|---|---|
| Decode 512 tok/s | 64.00 | 63.24 | −1.2% |
| Decode 4096 tok/s | 46.77 | 46.75 | 0.0% |
| Prompt 512 tok/s | 185.48 | 233.88 | +26.1% |
| Prompt 4096 tok/s | 130.56 | 156.05 | +19.5% |

Decode at 4096 is unchanged. Decode at 512 is 0.76 tok/s lower, with the two arms' ranges
overlapping (default 62.64–65.50, repacked 60.17–65.00). Prompt processing gains a fifth to a
quarter. This agrees with the 2026-11-08 baseline's KREPACK pass (decode −0.7% at 512 and
−2.1% at 4096, prompt +21.9% and +15.3%), which used three non-alternating rounds.

The raw manifests are in [`raw/`](raw/); each is a one-round `--runs 1` run set with both
contexts.
