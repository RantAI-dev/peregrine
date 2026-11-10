# S4.2 embedding baselines — generated tables


**minilm** — texts per s

| engine · precision | cos | L16 b1 | L16 b8 | L16 b32 | L16 b128 | L128 b1 | L128 b8 | L128 b32 | L128 b128 | L512 b1 | L512 b8 | L512 b32 | L512 b128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fastembed fp32 | 1.0000 | 240 | 506 | 643 | 600 | 84.60 | 80.01 | 73.54 | 74.96 | 15.46 | 13.57 | 13.26 | 12.51 |
| fastembed int8 | 0.9890 | 267 | 437 | 636 | 645 | 86.97 | 90.06 | 86.06 | 85.71 | 14.82 | 13.92 | 13.52 | 13.64 |
| llama.cpp f16 | 1.0000 | 246 | 269 | 299 | 310 | 56.59 | 60.34 | 60.62 | 60.81 | 13.13 | 13.27 | 13.30 | 13.27 |
| llama.cpp q8_0 | 0.9996 | 262 | 316 | 365 | 365 | 63.52 | 69.91 | 70.48 | 70.43 | 14.75 | 14.83 | 14.90 | 14.95 |
| OpenVINO fp32 | 1.0000 | 199 | 537 | 655 | 653 | 78.04 | 98.54 | 95.93 | 96.62 | 20.70 | 20.84 | 21.20 | 20.50 |
| TEI fp32 | 1.0000 | 153 | 223 | 228 | 247 | 35.77 | 36.39 | 37.75 | 37.45 | 4.26 | 4.46 | 4.10 | 4.01 |


**minilm** — tokens per s

| engine · precision | cos | L16 b1 | L16 b8 | L16 b32 | L16 b128 | L128 b1 | L128 b8 | L128 b32 | L128 b128 | L512 b1 | L512 b8 | L512 b32 | L512 b128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fastembed fp32 | 1.0000 | 3,836 | 8,090 | 10,289 | 9,595 | 10,829 | 10,241 | 9,413 | 9,594 | 7,910 | 6,942 | 6,780 | 6,397 |
| fastembed int8 | 0.9890 | 4,276 | 6,988 | 10,177 | 10,319 | 11,132 | 11,528 | 11,016 | 10,971 | 7,581 | 7,121 | 6,914 | 6,977 |
| llama.cpp f16 | 1.0000 | 3,943 | 4,309 | 4,790 | 4,968 | 7,244 | 7,723 | 7,760 | 7,784 | 6,718 | 6,788 | 6,801 | 6,788 |
| llama.cpp q8_0 | 0.9996 | 4,197 | 5,050 | 5,843 | 5,847 | 8,130 | 8,949 | 9,022 | 9,015 | 7,545 | 7,588 | 7,622 | 7,647 |
| OpenVINO fp32 | 1.0000 | 3,187 | 8,586 | 10,484 | 10,454 | 9,989 | 12,613 | 12,279 | 12,368 | 10,591 | 10,663 | 10,844 | 10,488 |
| TEI fp32 | 1.0000 | 2,455 | 3,562 | 3,647 | 3,944 | 4,579 | 4,658 | 4,832 | 4,793 | 2,177 | 2,279 | 2,096 | 2,052 |


## single-text latency (batch 1)


| model | engine · precision | p50 1×16 (ms) | p99 1×16 (ms) | p50 10×1×16 (ms) | p99 10×1×16 (ms) |
|---|---|---|---|---|---|
| minilm | fastembed fp32 | 3.9 | 4.7 | 20.5 | 39.4 |
| minilm | fastembed int8 | 3.3 | 4.4 | 19.3 | 36.4 |
| minilm | llama.cpp f16 | 3.9 | 4.3 | 19.8 | 39.7 |
| minilm | llama.cpp q8_0 | 3.4 | 5.0 | 17.4 | 38.6 |
| minilm | OpenVINO fp32 | 4.9 | 5.3 | 23.5 | 42.6 |
| minilm | TEI fp32 | 6.1 | 7.6 | 34.8 | 54.3 |


## fastest competitor per cell


| model | cell | best competitor | texts/s | tokens/s | worst-cell note |
|---|---|---|---|---|---|
| minilm | L16 b1 | fastembed int8 | 267 | 4,276 | 3 rounds |
| minilm | L16 b8 | OpenVINO fp32 | 537 | 8,586 | 3 rounds |
| minilm | L16 b32 | OpenVINO fp32 | 655 | 10,484 | 3 rounds |
| minilm | L16 b128 | OpenVINO fp32 | 653 | 10,454 | 3 rounds |
| minilm | L128 b1 | fastembed int8 | 87.0 | 11,132 | 3 rounds |
| minilm | L128 b8 | OpenVINO fp32 | 98.5 | 12,613 | 3 rounds |
| minilm | L128 b32 | OpenVINO fp32 | 95.9 | 12,279 | 3 rounds |
| minilm | L128 b128 | OpenVINO fp32 | 96.6 | 12,368 | 3 rounds |
| minilm | L512 b1 | OpenVINO fp32 | 20.7 | 10,591 | 3 rounds |
| minilm | L512 b8 | OpenVINO fp32 | 20.8 | 10,663 | 3 rounds |
| minilm | L512 b32 | OpenVINO fp32 | 21.2 | 10,844 | 3 rounds |
| minilm | L512 b128 | OpenVINO fp32 | 20.5 | 10,488 | 3 rounds |
