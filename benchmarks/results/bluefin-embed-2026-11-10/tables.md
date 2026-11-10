# S4.2 embedding baselines — generated tables


**minilm** — texts per s

| engine · precision | cos | L16 b1 | L16 b8 | L16 b32 | L16 b128 | L128 b1 | L128 b8 | L128 b32 | L128 b128 | L512 b1 | L512 b8 | L512 b32 | L512 b128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fastembed fp32 | 1.0000 | 240 | 501 | 643 | 607 | 82.05 | 79.37 | 75.83 | 76.00 | 15.72 | 13.91 | 13.52 | 12.55 |
| fastembed int8 | 0.9890 | 267 | 448 | 651 | 645 | 87.57 | 90.78 | 84.37 | 81.48 | 14.82 | 13.92 | 13.52 | 13.59 |
| llama.cpp f16 | 1.0000 | 246 | 269 | 299 | 311 | 56.59 | 60.29 | 60.08 | 60.96 | 13.18 | 13.27 | 13.32 | 13.28 |
| llama.cpp q8_0 | 0.9996 | 244 | 358 | 370 | 370 | 66.11 | 70.08 | 70.41 | 70.43 | 14.74 | 14.83 | 14.86 | 14.92 |
| OpenVINO fp32 | 1.0000 | 202 | 537 | 656 | 656 | 78.04 | 99.06 | 96.28 | 97.28 | 20.70 | 20.95 | 21.20 | 20.56 |
| TEI fp32 | 1.0000 | 153 | 223 | 243 | 254 | 35.92 | 34.38 | 37.75 | 37.11 | 4.08 | 4.46 | 4.06 | 4.01 |


**minilm** — tokens per s

| engine · precision | cos | L16 b1 | L16 b8 | L16 b32 | L16 b128 | L128 b1 | L128 b8 | L128 b32 | L128 b128 | L512 b1 | L512 b8 | L512 b32 | L512 b128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fastembed fp32 | 1.0000 | 3,836 | 8,014 | 10,289 | 9,704 | 10,502 | 10,159 | 9,706 | 9,728 | 8,040 | 7,118 | 6,914 | 6,421 |
| fastembed int8 | 0.9890 | 4,276 | 7,169 | 10,414 | 10,319 | 11,210 | 11,620 | 10,799 | 10,430 | 7,581 | 7,121 | 6,914 | 6,950 |
| llama.cpp f16 | 1.0000 | 3,930 | 4,309 | 4,790 | 4,972 | 7,244 | 7,717 | 7,690 | 7,803 | 6,742 | 6,788 | 6,812 | 6,792 |
| llama.cpp q8_0 | 0.9996 | 3,904 | 5,731 | 5,912 | 5,927 | 8,462 | 8,971 | 9,012 | 9,015 | 7,542 | 7,588 | 7,600 | 7,634 |
| OpenVINO fp32 | 1.0000 | 3,228 | 8,587 | 10,488 | 10,500 | 9,989 | 12,680 | 12,323 | 12,452 | 10,591 | 10,717 | 10,844 | 10,519 |
| TEI fp32 | 1.0000 | 2,455 | 3,562 | 3,888 | 4,071 | 4,598 | 4,401 | 4,832 | 4,750 | 2,088 | 2,279 | 2,076 | 2,052 |


## single-text latency (batch 1)


| model | engine · precision | p50 1×16 (ms) | p99 1×16 (ms) | p50 10×1×16 (ms) | p99 10×1×16 (ms) |
|---|---|---|---|---|---|
| minilm | fastembed fp32 | 3.9 | 4.7 | 21.2 | 40.1 |
| minilm | fastembed int8 | 3.3 | 4.4 | 19.2 | 36.4 |
| minilm | llama.cpp f16 | 3.9 | 4.5 | 22.1 | 39.9 |
| minilm | llama.cpp q8_0 | 3.5 | 5.1 | 15.3 | 27.3 |
| minilm | OpenVINO fp32 | 4.9 | 5.2 | 23.7 | 42.5 |
| minilm | TEI fp32 | 6.1 | 7.7 | 31.2 | 53.2 |


## fastest competitor per cell


| model | cell | best competitor | texts/s | tokens/s | worst-cell note |
|---|---|---|---|---|---|
| minilm | L16 b1 | fastembed int8 | 267 | 4,276 | 1 rounds |
| minilm | L16 b8 | OpenVINO fp32 | 537 | 8,587 | 1 rounds |
| minilm | L16 b32 | OpenVINO fp32 | 656 | 10,488 | 1 rounds |
| minilm | L16 b128 | OpenVINO fp32 | 656 | 10,500 | 1 rounds |
| minilm | L128 b1 | fastembed int8 | 87.6 | 11,210 | 1 rounds |
| minilm | L128 b8 | OpenVINO fp32 | 99.1 | 12,680 | 1 rounds |
| minilm | L128 b32 | OpenVINO fp32 | 96.3 | 12,323 | 1 rounds |
| minilm | L128 b128 | OpenVINO fp32 | 97.3 | 12,452 | 1 rounds |
| minilm | L512 b1 | OpenVINO fp32 | 20.7 | 10,591 | 1 rounds |
| minilm | L512 b8 | OpenVINO fp32 | 20.9 | 10,717 | 1 rounds |
| minilm | L512 b32 | OpenVINO fp32 | 21.2 | 10,844 | 1 rounds |
| minilm | L512 b128 | OpenVINO fp32 | 20.6 | 10,519 | 1 rounds |
