# P2-plus + P3 Update Summary

## C4-16384 static main-result update

| Method | Seeds | Val Loss | Test PPL | Delta Val ? | Delta PPL ? | actual_k | KV access | tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 5 | 4.0745 ? 0.0189 | 59.193 ? 1.117 | +0.0000 | +0.000 | 16.0 | 1024.0 | 11940 |
| mono_inc | 5 | 4.0671 ? 0.0275 | 58.790 ? 1.649 | +0.0074 | +0.402 | 16.0 | 1024.0 | 11890 |
| mono_dec | 3 | 4.0820 ? 0.0124 | 59.607 ? 0.773 | -0.0075 | -0.414 | 16.0 | 1024.0 | 12100 |

Note: baseline and mono_inc use 5 seeds; mono_dec remains a 3-seed reverse-direction control.

## P2-plus selected attention probability behavior

| Method | Low far mass (1-6) | High far mass (19-24) | Low entropy norm | High entropy norm |
|---|---:|---:|---:|---:|
| baseline | 0.141 ? 0.021 | 0.102 ? 0.028 | 0.323 | 0.496 |
| mono_inc | 0.131 ? 0.019 | 0.115 ? 0.037 | 0.329 | 0.494 |
| mono_dec | 0.165 ? 0.005 | 0.079 ? 0.023 | 0.321 | 0.518 |

Interpretation: P2-plus supports a probability-level behavioral diagnostic: mono_inc assigns slightly more selected-attention mass to far context in high layers, while mono_dec suppresses high-layer far-context mass and shifts broader access toward lower layers. This remains diagnostic evidence, not a causal proof.
