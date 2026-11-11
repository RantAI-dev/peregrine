# Desktop, S5.1: the wide pass

Raw measurements behind `docs/design/s5.1-wide-pass.md`.

- Host: AMD Ryzen AI 9 HX 370 (Zen 5, 12 cores / 24 threads), WSL2, 11.7 GiB. The load average was
  6–28 through these series (the agent's own UI), so the before/after ratios are the quantity to
  read, not the absolute tok/s.
- Baseline: the S5.1 branch before this change (`s5.1-q6k-zmm`).
- Models: Qwen 2.5 0.5B Q4_K_M and Llama 3.2 1B Q4_K_M; 512-token prompt.

| file | what it is |
|---|---|
| `chunk-12t.json` | `peregrine bench --prompt-tokens 512 --tokens 0` at `PEREGRINE_PREFILL_CHUNK` 16/32/64/128, twelve threads, 3 interleaved rounds of the two binaries; every sample |
| `chunk-1t.json` | the same at one thread, 1 round (the series is long at one thread) |
