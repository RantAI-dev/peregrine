# The other engines' raw lines (S5.4d)

Copied from the runs' stdout; the commands are in `README.md` §c.

llama.cpp v0.9.0-dev `cd959ba`, `/var/tmp/llama.cpp/build/bin/llama-bench`:

```
| llama 1B Q4_K - Medium         | 762.81 MiB |     1.24 B | CPU        |       1 |           pp512 |         51.09 ± 0.03 |
| llama 1B Q4_K - Medium         | 762.81 MiB |     1.24 B | CPU        |       4 |           pp512 |        192.09 ± 1.53 |
| qwen2 1B Q4_K - Medium         | 373.71 MiB |   494.03 M | CPU        |       1 |           pp512 |         81.83 ± 0.15 |
| qwen2 1B Q4_K - Medium         | 373.71 MiB |   494.03 M | CPU        |       4 |           pp512 |        305.53 ± 1.16 |
| llama 1B Q4_K - Medium         | 762.81 MiB |     1.24 B | CPU        |       4 |          pp4096 |        143.03 ± 0.49 |
| qwen2 1B Q4_K - Medium         | 373.71 MiB |   494.03 M | CPU        |       4 |          pp4096 |        237.18 ± 2.24 |
| llama 1B Q4_K - Medium         | 762.81 MiB |     1.24 B | CPU        |       1 |          pp4096 |         38.31 ± 0.04 |
| qwen2 1B Q4_K - Medium         | 373.71 MiB |   494.03 M | CPU        |       1 |          pp4096 |         63.71 ± 0.07 |
```

ik_llama.cpp `883789c`, `/var/tmp/ik-llama-s54/build/bin/llama-bench`, at its defaults (its
own repacked sizes: 968.30 MiB for the 1B, 511.65 MiB for the Qwen):

```
| llama ?B Q4_K - Medium         | 968.30 MiB |     1.50 B | CPU        |       1 |         pp512 |     57.28 ± 0.09 |
| llama ?B Q4_K - Medium         | 968.30 MiB |     1.50 B | CPU        |       4 |         pp512 |    220.14 ± 0.32 |
| qwen2 1B Q4_K - Medium         | 511.65 MiB |   630.17 M | CPU        |       1 |         pp512 |    127.94 ± 0.50 |
| qwen2 1B Q4_K - Medium         | 511.65 MiB |   630.17 M | CPU        |       4 |         pp512 |    488.96 ± 2.45 |
| llama ?B Q4_K - Medium         | 968.30 MiB |     1.50 B | CPU        |       4 |        pp4096 |    171.60 ± 0.59 |
| qwen2 1B Q4_K - Medium         | 511.65 MiB |   630.17 M | CPU        |       4 |        pp4096 |    353.31 ± 0.28 |
| llama ?B Q4_K - Medium         | 968.30 MiB |     1.50 B | CPU        |       1 |        pp4096 |     44.61 ± 0.13 |
| qwen2 1B Q4_K - Medium         | 511.65 MiB |   630.17 M | CPU        |       1 |        pp4096 |     92.85 ± 0.13 |
```

The 4096/1 rows are `-r 3` (each repetition is 90 s or more); every other row is `-r 5`.
