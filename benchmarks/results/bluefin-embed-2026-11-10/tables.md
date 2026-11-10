# S4.2 embedding baselines — generated tables


**bge-base** — texts per s

| engine · precision | cos | L16 b1 | L16 b8 | L16 b32 | L16 b128 | L128 b1 | L128 b8 | L128 b32 | L128 b128 | L512 b1 | L512 b8 | L512 b32 | L512 b128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fastembed fp32 | 1.0000 | 66.12 | 123 | 124 | 104 | 16.33 | 14.47 | 13.02 | 12.94 | 3.32 | 2.88 | 2.88 | skip |
| fastembed int8 | 0.9795 | 90.02 | 106 | 124 | 123 | 16.30 | 16.40 | 15.96 | 15.93 | 3.53 | 3.44 | 3.47 | skip |
| llama.cpp f32 | 1.0000 | 60.64 | 102 | 104 | 106 | 13.29 | 13.36 | 13.32 | 13.31 | 3.16 | 3.00 | 2.98 | skip |
| llama.cpp q8_0 | 0.9998 | 63.40 | 82.46 | 82.90 | 84.09 | 10.69 | 9.33 | 9.42 | 9.21 | 2.44 | 2.16 | 2.16 | skip |
| OpenVINO fp32 | 1.0000 | 60.08 | 111 | 117 | 116 | 14.45 | 13.69 | 15.21 | 14.76 | 3.35 | 3.55 | 3.54 | skip |
| TEI fp32 | 1.0000 | 26.44 | 27.97 | 29.14 | 29.10 | 5.80 | 5.67 | 5.86 | 5.80 | 1.08 | 1.03 | 1.00 | skip |

**bge-small** — texts per s

| engine · precision | cos | L16 b1 | L16 b8 | L16 b32 | L16 b128 | L128 b1 | L128 b8 | L128 b32 | L128 b128 | L512 b1 | L512 b8 | L512 b32 | L512 b128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fastembed fp32 | 1.0000 | 174 | 267 | 368 | 342 | 46.95 | 48.23 | 38.76 | 38.94 | 8.20 | 7.27 | 7.17 | 6.98 |
| fastembed int8 | 0.9796 | 206 | 310 | 383 | 389 | 47.02 | 48.36 | 46.88 | 47.15 | 8.37 | 8.32 | 8.31 | 8.47 |
| llama.cpp f32 | 1.0000 | 150 | 313 | 321 | 331 | 42.08 | 40.76 | 40.94 | 41.08 | 9.20 | 8.34 | 8.24 | 8.31 |
| llama.cpp q8_0 | 0.9999 | 166 | 257 | 264 | 271 | 34.75 | 31.92 | 32.29 | 32.16 | 7.47 | 6.81 | 6.71 | 6.75 |
| OpenVINO fp32 | 1.0000 | 142 | 344 | 400 | 385 | 45.47 | 53.22 | 50.86 | 50.87 | 11.11 | 10.79 | 10.98 | 10.78 |
| TEI fp32 | 1.0000 | 88.58 | 102 | 127 | 127 | 18.16 | 18.04 | 18.84 | 19.39 | 2.19 | 2.18 | 2.12 | 2.04 |

**minilm** — texts per s

| engine · precision | cos | L16 b1 | L16 b8 | L16 b32 | L16 b128 | L128 b1 | L128 b8 | L128 b32 | L128 b128 | L512 b1 | L512 b8 | L512 b32 | L512 b128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fastembed fp32 | 1.0000 | 240 | 506 | 643 | 600 | 84.60 | 80.01 | 73.54 | 74.96 | 15.46 | 13.57 | 13.26 | 12.51 |
| fastembed int8 | 0.9890 | 267 | 437 | 636 | 645 | 86.97 | 90.06 | 86.06 | 85.71 | 14.82 | 13.92 | 13.52 | 13.64 |
| llama.cpp f16 | 1.0000 | 246 | 269 | 299 | 310 | 56.59 | 60.34 | 60.62 | 60.81 | 13.13 | 13.27 | 13.30 | 13.27 |
| llama.cpp q8_0 | 0.9996 | 262 | 316 | 365 | 365 | 63.52 | 69.91 | 70.48 | 70.43 | 14.75 | 14.83 | 14.90 | 14.95 |
| OpenVINO fp32 | 1.0000 | 199 | 537 | 655 | 653 | 78.04 | 98.54 | 95.93 | 96.62 | 20.70 | 20.84 | 21.20 | 20.50 |
| TEI fp32 | 1.0000 | 153 | 223 | 228 | 247 | 35.77 | 36.39 | 37.75 | 37.45 | 4.26 | 4.46 | 4.10 | 4.01 |

**qwen3-embed-0.6b** — texts per s

| engine · precision | cos | L16 b1 | L16 b8 | L16 b32 | L16 b128 | L128 b1 | L128 b8 | L128 b32 | L128 b128 | L512 b1 | L512 b8 | L512 b32 | L512 b128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| llama.cpp q8_0 | 0.9993 | 15.71 | 15.87 | 15.97 | 15.82 | 2.19 | 2.22 | 2.20 | 2.19 | 0.50 | 0.50 | 0.49 | skip |


**bge-base** — tokens per s

| engine · precision | cos | L16 b1 | L16 b8 | L16 b32 | L16 b128 | L128 b1 | L128 b8 | L128 b32 | L128 b128 | L512 b1 | L512 b8 | L512 b32 | L512 b128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fastembed fp32 | 1.0000 | 1,058 | 1,971 | 1,988 | 1,667 | 2,090 | 1,853 | 1,667 | 1,656 | 1,696 | 1,472 | 1,474 | skip |
| fastembed int8 | 0.9795 | 1,440 | 1,701 | 1,991 | 1,963 | 2,087 | 2,099 | 2,043 | 2,039 | 1,804 | 1,758 | 1,774 | skip |
| llama.cpp f32 | 1.0000 | 970 | 1,634 | 1,668 | 1,688 | 1,700 | 1,710 | 1,705 | 1,704 | 1,616 | 1,534 | 1,524 | skip |
| llama.cpp q8_0 | 0.9998 | 1,014 | 1,319 | 1,326 | 1,346 | 1,369 | 1,194 | 1,205 | 1,180 | 1,247 | 1,105 | 1,105 | skip |
| OpenVINO fp32 | 1.0000 | 961 | 1,780 | 1,872 | 1,852 | 1,849 | 1,753 | 1,948 | 1,890 | 1,710 | 1,814 | 1,813 | skip |
| TEI fp32 | 1.0000 | 423 | 448 | 466 | 466 | 743 | 725 | 750 | 743 | 550 | 527 | 514 | skip |

**bge-small** — tokens per s

| engine · precision | cos | L16 b1 | L16 b8 | L16 b32 | L16 b128 | L128 b1 | L128 b8 | L128 b32 | L128 b128 | L512 b1 | L512 b8 | L512 b32 | L512 b128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fastembed fp32 | 1.0000 | 2,784 | 4,266 | 5,888 | 5,476 | 6,009 | 6,173 | 4,961 | 4,984 | 4,193 | 3,720 | 3,666 | 3,570 |
| fastembed int8 | 0.9796 | 3,296 | 4,962 | 6,126 | 6,220 | 6,018 | 6,191 | 6,001 | 6,035 | 4,282 | 4,259 | 4,251 | 4,332 |
| llama.cpp f32 | 1.0000 | 2,402 | 5,001 | 5,134 | 5,295 | 5,386 | 5,217 | 5,240 | 5,258 | 4,703 | 4,266 | 4,218 | 4,250 |
| llama.cpp q8_0 | 0.9999 | 2,659 | 4,106 | 4,231 | 4,335 | 4,448 | 4,085 | 4,133 | 4,117 | 3,822 | 3,484 | 3,431 | 3,454 |
| OpenVINO fp32 | 1.0000 | 2,275 | 5,508 | 6,398 | 6,155 | 5,820 | 6,812 | 6,510 | 6,511 | 5,681 | 5,519 | 5,616 | 5,517 |
| TEI fp32 | 1.0000 | 1,417 | 1,636 | 2,026 | 2,028 | 2,325 | 2,309 | 2,412 | 2,481 | 1,120 | 1,116 | 1,084 | 1,043 |

**minilm** — tokens per s

| engine · precision | cos | L16 b1 | L16 b8 | L16 b32 | L16 b128 | L128 b1 | L128 b8 | L128 b32 | L128 b128 | L512 b1 | L512 b8 | L512 b32 | L512 b128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fastembed fp32 | 1.0000 | 3,836 | 8,090 | 10,289 | 9,595 | 10,829 | 10,241 | 9,413 | 9,594 | 7,910 | 6,942 | 6,780 | 6,397 |
| fastembed int8 | 0.9890 | 4,276 | 6,988 | 10,177 | 10,319 | 11,132 | 11,528 | 11,016 | 10,971 | 7,581 | 7,121 | 6,914 | 6,977 |
| llama.cpp f16 | 1.0000 | 3,943 | 4,309 | 4,790 | 4,968 | 7,244 | 7,723 | 7,760 | 7,784 | 6,718 | 6,788 | 6,801 | 6,788 |
| llama.cpp q8_0 | 0.9996 | 4,197 | 5,050 | 5,843 | 5,847 | 8,130 | 8,949 | 9,022 | 9,015 | 7,545 | 7,588 | 7,622 | 7,647 |
| OpenVINO fp32 | 1.0000 | 3,187 | 8,586 | 10,484 | 10,454 | 9,989 | 12,613 | 12,279 | 12,368 | 10,591 | 10,663 | 10,844 | 10,488 |
| TEI fp32 | 1.0000 | 2,455 | 3,562 | 3,647 | 3,944 | 4,579 | 4,658 | 4,832 | 4,793 | 2,177 | 2,279 | 2,096 | 2,052 |

**qwen3-embed-0.6b** — tokens per s

| engine · precision | cos | L16 b1 | L16 b8 | L16 b32 | L16 b128 | L128 b1 | L128 b8 | L128 b32 | L128 b128 | L512 b1 | L512 b8 | L512 b32 | L512 b128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| llama.cpp q8_0 | 0.9993 | 251 | 254 | 256 | 253 | 281 | 284 | 281 | 281 | 255 | 254 | 252 | skip |


## single-text latency (batch 1)


| model | engine · precision | p50 1×16 (ms) | p99 1×16 (ms) | p50 10×1×16 (ms) | p99 10×1×16 (ms) |
|---|---|---|---|---|---|
| bge-base | fastembed fp32 | 14.8 | 18.9 | 85.3 | 151.4 |
| bge-base | fastembed int8 | 10.5 | 16.3 | 60.3 | 109.0 |
| bge-base | llama.cpp f32 | 16.4 | 17.1 | 99.8 | 110.0 |
| bge-base | llama.cpp q8_0 | 15.4 | 22.1 | 133.8 | 153.2 |
| bge-base | OpenVINO fp32 | 16.5 | 18.1 | 87.9 | 159.3 |
| bge-base | TEI fp32 | 36.0 | 44.7 | 241.6 | 397.4 |
| bge-small | fastembed fp32 | 5.6 | 6.1 | 33.0 | 63.4 |
| bge-small | fastembed int8 | 4.5 | 6.3 | 28.5 | 55.4 |
| bge-small | llama.cpp f32 | 6.5 | 7.9 | 32.7 | 42.0 |
| bge-small | llama.cpp q8_0 | 5.8 | 8.5 | 38.2 | 55.5 |
| bge-small | OpenVINO fp32 | 7.0 | 7.2 | 35.4 | 63.6 |
| bge-small | TEI fp32 | 10.8 | 13.5 | 60.5 | 99.1 |
| minilm | fastembed fp32 | 3.9 | 4.7 | 20.5 | 39.4 |
| minilm | fastembed int8 | 3.3 | 4.4 | 19.3 | 36.4 |
| minilm | llama.cpp f16 | 3.9 | 4.3 | 19.8 | 39.7 |
| minilm | llama.cpp q8_0 | 3.4 | 5.0 | 17.4 | 38.6 |
| minilm | OpenVINO fp32 | 4.9 | 5.3 | 23.5 | 42.6 |
| minilm | TEI fp32 | 6.1 | 7.6 | 34.8 | 54.3 |
| qwen3-embed-0.6b | llama.cpp q8_0 | 63.1 | 65.2 | 622.4 | 648.8 |


## fastest competitor per cell


| model | cell | best competitor | texts/s | tokens/s | worst-cell note |
|---|---|---|---|---|---|
| bge-base | L16 b1 | fastembed int8 | 90.0 | 1,440 | 3 rounds |
| bge-base | L16 b8 | fastembed fp32 | 123 | 1,971 | 3 rounds |
| bge-base | L16 b32 | fastembed int8 | 124 | 1,991 | 3 rounds |
| bge-base | L16 b128 | fastembed int8 | 123 | 1,963 | 3 rounds |
| bge-base | L128 b1 | fastembed fp32 | 16.3 | 2,090 | 3 rounds |
| bge-base | L128 b8 | fastembed int8 | 16.4 | 2,099 | 3 rounds |
| bge-base | L128 b32 | fastembed int8 | 16.0 | 2,043 | 3 rounds |
| bge-base | L128 b128 | fastembed int8 | 15.9 | 2,039 | 3 rounds |
| bge-base | L512 b1 | fastembed int8 | 3.5 | 1,804 | 3 rounds |
| bge-base | L512 b8 | OpenVINO fp32 | 3.5 | 1,814 | 3 rounds |
| bge-base | L512 b32 | OpenVINO fp32 | 3.5 | 1,813 | 3 rounds |
| bge-small | L16 b1 | fastembed int8 | 206 | 3,296 | 3 rounds |
| bge-small | L16 b8 | OpenVINO fp32 | 344 | 5,508 | 3 rounds |
| bge-small | L16 b32 | OpenVINO fp32 | 400 | 6,398 | 3 rounds |
| bge-small | L16 b128 | fastembed int8 | 389 | 6,220 | 3 rounds |
| bge-small | L128 b1 | fastembed int8 | 47.0 | 6,018 | 3 rounds |
| bge-small | L128 b8 | OpenVINO fp32 | 53.2 | 6,812 | 3 rounds |
| bge-small | L128 b32 | OpenVINO fp32 | 50.9 | 6,510 | 3 rounds |
| bge-small | L128 b128 | OpenVINO fp32 | 50.9 | 6,511 | 3 rounds |
| bge-small | L512 b1 | OpenVINO fp32 | 11.1 | 5,681 | 3 rounds |
| bge-small | L512 b8 | OpenVINO fp32 | 10.8 | 5,519 | 3 rounds |
| bge-small | L512 b32 | OpenVINO fp32 | 11.0 | 5,616 | 3 rounds |
| bge-small | L512 b128 | OpenVINO fp32 | 10.8 | 5,517 | 3 rounds |
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
| qwen3-embed-0.6b | L16 b1 | llama.cpp q8_0 | 15.7 | 251 | 3 rounds |
| qwen3-embed-0.6b | L16 b8 | llama.cpp q8_0 | 15.9 | 254 | 3 rounds |
| qwen3-embed-0.6b | L16 b32 | llama.cpp q8_0 | 16.0 | 256 | 3 rounds |
| qwen3-embed-0.6b | L16 b128 | llama.cpp q8_0 | 15.8 | 253 | 3 rounds |
| qwen3-embed-0.6b | L128 b1 | llama.cpp q8_0 | 2.2 | 281 | 3 rounds |
| qwen3-embed-0.6b | L128 b8 | llama.cpp q8_0 | 2.2 | 284 | 3 rounds |
| qwen3-embed-0.6b | L128 b32 | llama.cpp q8_0 | 2.2 | 281 | 3 rounds |
| qwen3-embed-0.6b | L128 b128 | llama.cpp q8_0 | 2.2 | 281 | 3 rounds |
| qwen3-embed-0.6b | L512 b1 | llama.cpp q8_0 | 0.5 | 255 | 3 rounds |
| qwen3-embed-0.6b | L512 b8 | llama.cpp q8_0 | 0.5 | 254 | 3 rounds |
| qwen3-embed-0.6b | L512 b32 | llama.cpp q8_0 | 0.5 | 252 | 3 rounds |
