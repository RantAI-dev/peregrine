# Desktop, S5.1 Q6_K: the 512-bit tile kernel

Raw measurements behind `docs/design/s5.1-q6k-zmm.md`.

- Host: AMD Ryzen AI 9 HX 370 (Zen 5, 12 cores / 24 threads, AVX-512 F/BW/VL/VNNI/VBMI), WSL2,
  11.7 GiB. `uptime` is at the top of each file. The host was not otherwise idle (the agent's own
  UI), so the end-to-end series interleaves the two binaries and reports the median of the rounds.
- Baseline: the S5.1 branch before this change (`s5.1-kq-epilogue`).
- Models: Qwen 2.5 0.5B Instruct Q4_K_M and Llama 3.2 1B Instruct Q4_K_M. Prompt 512 tokens.

| file | what it is |
|---|---|
| `kernels-l1.txt` | `kqbench`, 3 rounds, `taskset -c 2`, one 8-row tile of 16 super-blocks (18 KiB, L1): the Q6_K per-row path, the AVX2 eight-row tile and the new `q6k8_gemm_zmm`, at 1, 4, 16, 64 and 128 activation rows |
| `kernels-dram.txt` | the same on a tile of 32 768 super-blocks (55 MiB, DRAM) |
| `prefill-decode-12t.json` | `peregrine bench`, 5 interleaved rounds of the two binaries, 12 threads, prompt 512, 4 decode tokens |

`kqbench`'s Q6_K figure is `2048 / ns` per (tile, activation row, super-block).
