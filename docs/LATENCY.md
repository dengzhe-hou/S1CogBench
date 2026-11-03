# Controlled latency

These are the accepted per-mode timings of the controlled latency study (P1). Times are milliseconds on one A100 40GB device at batch size one, with 20 warm-up calls per mode. N counts timed requests, and Tokens gives median input length. Ratio is the median per-request end-to-end ratio to Llama on matching source identities, using the original analyzer. Approved 200-request variants retain their prefix sizes. Forward and end-to-end timers are distinct from API service time ([API_TIME_COST.md](API_TIME_COST.md)).

The protocol is in [SUPPLEMENTARY_CONTROLS.md](SUPPLEMENTARY_CONTROLS.md). Laya TileLang is the row the paper labels `Laya fast`. Its two largest P95 values, in Recent probes and SHJ, are consistent with compiling kernels for new input shapes during timed calls.

## Horizon

| Configuration | N | Tokens | E2E P50 | E2E P95 | Forward P50 | Ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Laya FP32 | 1000 | 246 | 25.50 | 26.16 | 23.42 | 0.69 |
| Laya BF16 | 1000 | 246 | 26.55 | 26.87 | 24.27 | 0.71 |
| Laya TileLang | 200 | 246 | 6.45 | 6.79 | 4.51 | 0.17 |
| Intern FLA | 1000 | 401 | 52.06 | 52.74 | 49.86 | 1.41 |
| Intern Torch | 200 | 401 | 174.58 | 176.55 | 172.34 | 4.70 |
| Qwen FLA | 1000 | 284 | 51.66 | 52.11 | 50.14 | 1.40 |
| Qwen Torch | 200 | 284 | 162.25 | 166.82 | 160.73 | 4.40 |
| Llama | 1000 | 294 | 36.97 | 37.60 | 34.51 | 1.00 |
| Minitaur | 1000 | 262 | 35.65 | 36.07 | 34.25 | 0.96 |

## Recent probes

| Configuration | N | Tokens | E2E P50 | E2E P95 | Forward P50 | Ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Laya FP32 | 1000 | 1478 | 101.75 | 195.16 | 95.53 | 0.80 |
| Laya BF16 | 1000 | 1478 | 32.29 | 56.06 | 26.25 | 0.27 |
| Laya TileLang | 200 | 1442 | 20.67 | 50394.59 | 13.19 | 0.16 |
| Intern FLA | 1000 | 1610 | 101.26 | 172.44 | 96.88 | 0.81 |
| Intern Torch | 200 | 1574 | 293.82 | 487.63 | 289.61 | 2.44 |
| Qwen FLA | 1000 | 1492 | 92.69 | 166.58 | 89.14 | 0.74 |
| Qwen Torch | 200 | 1456 | 270.82 | 447.72 | 267.41 | 2.28 |
| Llama | 1000 | 1471 | 124.47 | 221.16 | 117.21 | 1.00 |
| Minitaur | 1000 | 1439 | 120.49 | 215.20 | 116.62 | 0.97 |

## Multiple attribute choice

| Configuration | N | Tokens | E2E P50 | E2E P95 | Forward P50 | Ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Laya FP32 | 576 | 1463 | 98.01 | 188.62 | 91.96 | 0.68 |
| Laya BF16 | 576 | 1463 | 32.22 | 55.02 | 26.04 | 0.23 |
| Laya TileLang | 200 | 1463 | 17.11 | 28.86 | 10.68 | 0.12 |
| Intern FLA | 576 | 1901 | 114.98 | 192.90 | 110.29 | 0.79 |
| Intern Torch | 200 | 1901 | 340.66 | 562.20 | 336.05 | 2.33 |
| Qwen FLA | 576 | 1784 | 108.54 | 186.25 | 104.59 | 0.74 |
| Qwen Torch | 200 | 1784 | 329.26 | 548.31 | 325.28 | 2.19 |
| Llama | 576 | 1746 | 152.73 | 254.24 | 144.21 | 1.00 |
| Minitaur | 576 | 1714 | 148.16 | 247.82 | 143.72 | 0.97 |

## RLWM history

| Configuration | N | Tokens | E2E P50 | E2E P95 | Forward P50 | Ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Laya FP32 | 1000 | 416 | 36.07 | 69.44 | 33.18 | 0.67 |
| Laya BF16 | 1000 | 416 | 26.70 | 28.99 | 23.91 | 0.51 |
| Laya TileLang | 200 | 413 | 8.02 | 13.78 | 5.27 | 0.15 |
| Intern FLA | 1000 | 615 | 53.50 | 89.06 | 51.01 | 1.03 |
| Intern Torch | 200 | 612 | 190.30 | 257.46 | 187.64 | 3.64 |
| Qwen FLA | 1000 | 497 | 52.06 | 75.59 | 50.19 | 1.01 |
| Qwen Torch | 200 | 494 | 178.99 | 235.06 | 177.13 | 3.43 |
| Llama | 1000 | 484 | 51.84 | 97.05 | 47.77 | 1.00 |
| Minitaur | 1000 | 452 | 49.19 | 91.98 | 47.40 | 0.95 |

## SHJ

| Configuration | N | Tokens | E2E P50 | E2E P95 | Forward P50 | Ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Laya FP32 | 1000 | 3591 | 274.44 | 742.54 | 258.22 | 0.95 |
| Laya BF16 | 1000 | 3591 | 82.18 | 200.76 | 67.93 | 0.29 |
| Laya TileLang | 200 | 3491 | 59.83 | 45873.71 | 35.22 | 0.16 |
| Intern FLA | 1000 | 3726 | 219.40 | 437.76 | 209.93 | 0.77 |
| Intern Torch | 200 | 3626 | 655.74 | 1222.52 | 633.40 | 2.21 |
| Qwen FLA | 1000 | 3606 | 211.88 | 427.38 | 203.95 | 0.72 |
| Qwen Torch | 200 | 3506 | 595.82 | 1188.75 | 588.33 | 2.12 |
| Llama | 1000 | 3448 | 274.02 | 611.22 | 256.62 | 1.00 |
| Minitaur | 1000 | 3416 | 265.50 | 596.42 | 256.04 | 0.97 |

## WCST

| Configuration | N | Tokens | E2E P50 | E2E P95 | Forward P50 | Ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Laya FP32 | 1000 | 3194 | 241.30 | 539.01 | 223.66 | 0.93 |
| Laya BF16 | 1000 | 3194 | 66.75 | 146.91 | 53.30 | 0.28 |
| Laya TileLang | 200 | 2983 | 35.13 | 79.07 | 21.10 | 0.14 |
| Intern FLA | 1000 | 3254 | 192.55 | 349.16 | 184.20 | 0.75 |
| Intern Torch | 200 | 3049 | 546.22 | 1068.31 | 538.07 | 2.19 |
| Qwen FLA | 1000 | 3132 | 186.11 | 335.21 | 177.87 | 0.71 |
| Qwen Torch | 200 | 2927 | 521.41 | 1064.39 | 514.01 | 2.09 |
| Llama | 1000 | 2981 | 259.85 | 477.91 | 234.55 | 1.00 |
| Minitaur | 1000 | 2949 | 242.95 | 448.44 | 234.97 | 0.93 |
