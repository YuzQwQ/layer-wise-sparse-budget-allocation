| Sparse unit / platform | Dataset | Best schedule | Delta Val | Interpretation |
|---|---|---|---:|---|
| NSA block | C4 | mono_inc | +0.007 | C4-16384 long-training main comparison |
| NSA block | OpenWebText | mono_inc | +0.016 | cross-corpus support |
| NSA block | WikiText-103 | mono_dec | +0.004 | mixed direction on third corpus |
| H2O token | C4 | uniform | +0.000 | uniform strong default |
| H2O token | OpenWebText | uniform ~= mono_dec | +0.000 / -0.001 | near-tie; mono_dec PPL slightly better |
| H2O token | WikiText-103 | uniform | +0.000 | uniform strongest |
| Sliding window | C4 | mono_dec | +0.300 | mono_dec wins 3/3 windows |
| Sliding window | OpenWebText | mono_dec | +0.327 | mono_dec wins 3/3 windows |
| Sliding window | WikiText-103 | mono_dec | +0.241 | single official split diagnostic |
