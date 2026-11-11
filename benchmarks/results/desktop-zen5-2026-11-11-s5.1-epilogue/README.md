# Desktop, S5.1: the AVX-512 k-quant kernels, before and after

Raw measurements behind the S5.1 kernel work. The study they follow is
`docs/design/s5-prefill-avx512.md` (PR #194, not on `main` when this ran); the note beside this
folder is `docs/design/s5.1-kq-epilogue.md`.

- Host: AMD Ryzen AI 9 HX 370 (Zen 5, 12 cores / 24 threads, AVX-512 F/BW/VL/VNNI/VBMI), WSL2,
  11.7 GiB. `uptime` is at the top of every file it applies to; the host was not otherwise
  idle for all of them (the agent's own UI), so every series **interleaves** the two binaries or
  the two kernels and reports the median of the rounds. Between-series comparisons are not
  trustworthy on this host.
- Peregrine: `origin/main` `ab54f4e` (old) against the S5.1 branch (new), `cargo build --release -p peregrine`.
- Models: Qwen 2.5 0.5B Instruct Q4_K_M (491,400,032 B) and Llama 3.2 1B Instruct Q4_K_M
  (807,694,464 B).
- Prompt 512 tokens.

| file | what it is |
|---|---|
| `kernels-l1.txt` | `kqbench`, 5 interleaved rounds old/new, `taskset -c 2`, one 8-row tile of 16 super-blocks (18 KiB, L1-resident): zmm vs avx2 MAC/s per core at 1, 4, 16, 64, 128 activation rows, plus the one-row GEMV |
| `kernels-dram.txt` | the same, 3 interleaved rounds, one tile of 32 768 super-blocks (36 MiB, DRAM): the working set the engine actually has at chunk 16, where each pass streams the whole matrix |
| `prefill-decode-12t.json` | `peregrine bench`, 7 interleaved rounds old/new, 12 threads, prompt 512, 4 decode tokens; every raw sample |
| `prefill-decode-1t.json` | the same at one thread, 3 rounds |

Commands:

```
cargo run --release -p peregrine-bench --example kqbench -- <n_sb> <reps>
peregrine bench --model <file> --prompt-tokens 512 --tokens N --threads T --no-profile --json
```

`kqbench`'s GMAC/s is `2048 / ns` per (tile, activation row, super-block) for Q4_K and
`256 / ns` per (tile, activation row, block) for Q5_0.
