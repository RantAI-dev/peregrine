# The other engines, same cells (S5.4f §c)

`llama.cpp` build `cd959ba` (v0.9.0-dev), `/var/tmp/llama.cpp/build/bin/llama-bench`.
`ik_llama.cpp` build `883789c` (`cmake -B build -DCMAKE_BUILD_TYPE=Release`, built on this
host), `/var/tmp/ik-llama-s54/build/bin/llama-bench`. Prompt tokens/s, `-n 0`, `-r 5`
(`-r 3` for the 4096-token one-thread rows), `taskset -c 0-3` at 4 threads and `-c 0` at 1.

```
taskset -c 0-3 <engine>/llama-bench -m <model> -p 512  -n 0 -t 4 -r 5
taskset -c 0   <engine>/llama-bench -m <model> -p 4096 -n 0 -t 1 -r 3
```

| model | prompt | threads | llama.cpp pp | ik_llama.cpp pp |
|---|---|---|---|---|
| Llama 3.2 1B Instruct Q4_K_M | 512 | 1 | 51.07 ± 0.08 | 57.09 ± 0.10 |
| Llama 3.2 1B Instruct Q4_K_M | 512 | 4 | 194.02 ± 0.40 | 221.42 ± 0.32 |
| Llama 3.2 1B Instruct Q4_K_M | 4096 | 1 | 38.14 ± 0.03 | 44.62 ± 0.10 |
| Llama 3.2 1B Instruct Q4_K_M | 4096 | 4 | 143.16 ± 0.83 | 171.51 ± 0.55 |
| Qwen 2.5 0.5B Instruct Q4_K_M | 512 | 1 | 81.59 ± 0.20 | 127.93 ± 0.35 |
| Qwen 2.5 0.5B Instruct Q4_K_M | 512 | 4 | 305.11 ± 1.72 | 487.63 ± 2.38 |
| Qwen 2.5 0.5B Instruct Q4_K_M | 4096 | 1 | 63.58 ± 0.12 | 92.67 ± 0.24 |
| Qwen 2.5 0.5B Instruct Q4_K_M | 4096 | 4 | 239.70 ± 0.59 | 354.36 ± 0.28 |
