# R14 Static Long-Training Summary

| method | count | seeds | val@8192 | val@16384 | final val loss | test PPL | actual_k | KV access | tok/s | dVal final vs baseline | dTest PPL vs baseline |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 3 | 42,123,3407 | 4.2604 +/- 0.0163 | 4.0680 +/- 0.0125 | 4.0680 +/- 0.0125 | 58.824 +/- 0.767 | 16.000 +/- 0.000 | 1024.00 +/- 0.00 | 11949.5 +/- 50.3 | 0.0000 | 0.000 |
| mono_inc | 3 | 42,123,3407 | 4.2499 +/- 0.0210 | 4.0577 +/- 0.0226 | 4.0577 +/- 0.0226 | 58.239 +/- 1.328 | 16.000 +/- 0.000 | 1024.00 +/- 0.00 | 11941.9 +/- 90.4 | -0.0103 | -0.585 |
| mono_dec | 3 | 42,123,3407 | 4.2740 +/- 0.0121 | 4.0820 +/- 0.0101 | 4.0820 +/- 0.0101 | 59.607 +/- 0.631 | 16.000 +/- 0.000 | 1024.00 +/- 0.00 | 12099.8 +/- 38.7 | 0.0140 | 0.783 |

Decision: mono_inc retains the matched-compute long-training signal and remains the supported static prior.
