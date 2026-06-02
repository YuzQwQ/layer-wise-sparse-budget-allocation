# R13 Static Direction Summary

## C4 5-seed static direction

| method | count | seeds | final val loss | test PPL | actual_k | KV access | tok/s | dVal vs baseline | dVal vs mono_inc |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 5 | 42,123,2024,2025,3407 | 4.4104 +/- 0.0059 | 82.839 +/- 0.481 | 16.000 +/- 0.000 | 1024.00 +/- 0.00 | 11123.0 +/- 745.2 | 0.0000 | 0.0030 |
| mono_inc | 5 | 42,123,2024,2025,3407 | 4.4074 +/- 0.0208 | 82.608 +/- 1.753 | 16.000 +/- 0.000 | 1024.00 +/- 0.00 | 10990.6 +/- 808.0 | -0.0030 | 0.0000 |
| mono_dec | 5 | 42,123,2024,2025,3407 | 4.4181 +/- 0.0174 | 83.487 +/- 1.438 | 16.000 +/- 0.000 | 1024.00 +/- 0.00 | 10041.3 +/- 100.1 | 0.0077 | 0.0107 |

Decision: mono_inc remains the supported static direction; mono_dec does not retain a matched-compute improvement over baseline.

## OpenWebText direction check

| method | count | seeds | final val loss | test PPL | actual_k | KV access | tok/s | dVal vs baseline | dVal vs mono_inc |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 3 | 42,123,3407 | 4.8680 +/- 0.0057 | 133.587 +/- 0.649 | 16.000 +/- 0.000 | 1024.00 +/- 0.00 | 12030.4 +/- 35.7 | 0.0000 | 0.0158 |
| mono_inc | 3 | 42,123,3407 | 4.8522 +/- 0.0079 | 131.466 +/- 1.250 | 16.000 +/- 0.000 | 1024.00 +/- 0.00 | 12047.1 +/- 191.0 | -0.0158 | 0.0000 |
| mono_dec | 3 | 42,123,3407 | 4.8590 +/- 0.0179 | 132.330 +/- 2.260 | 16.000 +/- 0.000 | 1024.00 +/- 0.00 | 12008.3 +/- 48.8 | -0.0090 | 0.0068 |

Decision: OpenWebText favors mono_inc over mono_dec, suggesting possible corpus sensitivity.
