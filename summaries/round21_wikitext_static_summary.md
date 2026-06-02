# Round21 WikiText-103 Static Budget Support

Decision status: `mono_dec_stronger_or_mixed`

mono_dec is competitive or stronger than mono_inc; monotonic direction is not fully settled on WikiText-103.

| Method | Seeds | Val Loss | Test PPL | Delta Val ↑ | Delta PPL ↑ | actual_k | KV access | tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 42,123,3407 | 4.1668 +/- 0.0327 | 62.715 +/- 2.137 | +0.0000 | +0.000 | 16.000 | 1024.00 | 14469.7 +/- 44.0 |
| mono_inc | 42,123,3407 | 4.1889 +/- 0.0083 | 64.075 +/- 0.561 | -0.0221 | -1.360 | 16.000 | 1024.00 | 14306.7 +/- 39.9 |
| mono_dec | 42,123,3407 | 4.1630 +/- 0.0175 | 62.378 +/- 1.146 | +0.0037 | +0.337 | 16.000 | 1024.00 | 14328.4 +/- 7.1 |

Notes:
- Delta is defined as baseline - method; positive values indicate improvement over uniform baseline.
- All three static schedules are compute-matched at average actual_k = 16.0 and estimated selected KV access = 1024.0.
- WikiText-103 is additional cross-corpus directional support, not proof of broad generalization.
