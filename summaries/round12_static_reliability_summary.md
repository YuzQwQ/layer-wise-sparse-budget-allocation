# R12 Static Reliability Summary

## Five-seed baseline vs mono_inc

| method | count | seeds | final val loss | test ppl | actual_k | KV access | tok/s |
|---|---:|---|---:|---:|---:|---:|---:|
| baseline | 5 | 42,123,2024,2025,3407 | 4.4104 +/- 0.0059 | 82.839 +/- 0.481 | 16.000 +/- 0.000 | 1024.00 +/- 0.00 | 11123.0 +/- 745.2 |
| mono_inc | 5 | 42,123,2024,2025,3407 | 4.4074 +/- 0.0208 | 82.608 +/- 1.753 | 16.000 +/- 0.000 | 1024.00 +/- 0.00 | 10990.6 +/- 808.0 |

Deltas are mono_inc minus baseline; negative loss/PPL deltas are improvements.

- final validation loss improves by 0.0030 if the delta is negative.
- final test PPL delta: -0.231.
- estimated KV access delta: 0.00.

## Static schedule comparison

| method | count | seeds | final val loss | test ppl | actual_k | KV access | tok/s |
|---|---:|---|---:|---:|---:|---:|---:|
| baseline | 3 | 42,123,3407 | 4.4135 +/- 0.0025 | 83.097 +/- 0.209 | 16.000 +/- 0.000 | 1024.00 +/- 0.00 | 11729.2 +/- 76.5 |
| mono_inc | 3 | 42,123,3407 | 4.4111 +/- 0.0183 | 82.921 +/- 1.535 | 16.000 +/- 0.000 | 1024.00 +/- 0.00 | 11649.0 +/- 63.1 |
| mono_dec | 3 | 42,123,3407 | 4.4106 +/- 0.0190 | 82.858 +/- 1.561 | 16.000 +/- 0.000 | 1024.00 +/- 0.00 | 10079.5 +/- 112.3 |
| u_shape | 3 | 42,123,3407 | 4.4266 +/- 0.0058 | 84.170 +/- 0.475 | 16.000 +/- 0.000 | 1024.00 +/- 0.00 | 10046.9 +/- 13.1 |
