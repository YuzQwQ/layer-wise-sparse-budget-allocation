# R26 H2O-style Evaluation-window Robustness Diagnostic

This is inference-time cross-operator diagnostic evidence, not an H2O benchmark or reproduction.

## Decisions

- **C4**: `uniform_strong_default_with_window_robustness`
- **OpenWebText**: `uniform_strong_default_with_window_robustness`

## Mean across deterministic windows

| Dataset | Method | Windows | Val Loss mean+/-std (lower) | Test PPL mean+/-std (lower) | Delta Val mean+/-std (higher) | Delta PPL mean+/-std (higher) | Val wins | PPL wins | avg HH | avg retained | tok/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C4 | mono_dec | 3 | 3.1648+/-0.0158 | 24.302+/-0.655 | -0.0046+/-0.0002 | -0.091+/-0.002 | 0/3 | 0/3 | 128 | 256 | 629.2 |
| C4 | mono_inc | 3 | 3.1712+/-0.0166 | 24.507+/-0.676 | -0.0110+/-0.0008 | -0.296+/-0.034 | 0/3 | 0/3 | 128 | 256 | 638.7 |
| C4 | uniform | 3 | 3.1602+/-0.0158 | 24.211+/-0.655 | +0.0000+/-0.0000 | +0.000+/-0.000 | 0/3 | 0/3 | 128 | 256 | 631.1 |
| OpenWebText | mono_dec | 3 | 2.8609+/-0.0032 | 18.103+/-0.654 | -0.0007+/-0.0022 | +0.028+/-0.010 | 1/3 | 3/3 | 128 | 256 | 638.3 |
| OpenWebText | mono_inc | 3 | 2.8847+/-0.0061 | 18.610+/-0.703 | -0.0245+/-0.0010 | -0.479+/-0.045 | 0/3 | 0/3 | 128 | 256 | 638.5 |
| OpenWebText | uniform | 3 | 2.8602+/-0.0052 | 18.131+/-0.662 | +0.0000+/-0.0000 | +0.000+/-0.000 | 0/3 | 0/3 | 128 | 256 | 637.9 |

## Per-window results

| Dataset | Window | Val skip | Test skip | Method | Val Loss (lower) | Test PPL (lower) | Delta Val (higher) | Delta PPL (higher) | tokens |
|---|---|---:|---:|---|---:|---:|---:|---:|---|
| C4 | A | 0 | 8000000 | uniform | 3.1668 | 24.723 | +0.0000 | +0.000 | 393216/393216 |
| C4 | A | 0 | 8000000 | mono_inc | 3.1781 | 25.059 | -0.0113 | -0.336 | 393216/393216 |
| C4 | A | 0 | 8000000 | mono_dec | 3.1716 | 24.815 | -0.0048 | -0.092 | 393216/393216 |
| C4 | B | 1000000 | 9000000 | uniform | 3.1422 | 24.437 | +0.0000 | +0.000 | 393216/393216 |
| C4 | B | 1000000 | 9000000 | mono_inc | 3.1523 | 24.711 | -0.0101 | -0.273 | 393216/393216 |
| C4 | B | 1000000 | 9000000 | mono_dec | 3.1467 | 24.525 | -0.0046 | -0.088 | 393216/393216 |
| C4 | C | 2000000 | 10000000 | uniform | 3.1716 | 23.473 | +0.0000 | +0.000 | 393216/393216 |
| C4 | C | 2000000 | 10000000 | mono_inc | 3.1831 | 23.753 | -0.0116 | -0.279 | 393216/393216 |
| C4 | C | 2000000 | 10000000 | mono_dec | 3.1760 | 23.565 | -0.0045 | -0.091 | 393216/393216 |
| OpenWebText | A | 0 | 8000000 | uniform | 2.8634 | 17.729 | +0.0000 | +0.000 | 393216/393216 |
| OpenWebText | A | 0 | 8000000 | mono_inc | 2.8880 | 18.203 | -0.0246 | -0.474 | 393216/393216 |
| OpenWebText | A | 0 | 8000000 | mono_dec | 2.8621 | 17.700 | +0.0013 | +0.030 | 393216/393216 |
| OpenWebText | B | 1000000 | 9000000 | uniform | 2.8542 | 18.895 | +0.0000 | +0.000 | 393216/393216 |
| OpenWebText | B | 1000000 | 9000000 | mono_inc | 2.8776 | 19.421 | -0.0234 | -0.526 | 393216/393216 |
| OpenWebText | B | 1000000 | 9000000 | mono_dec | 2.8573 | 18.858 | -0.0031 | +0.037 | 393216/393216 |
| OpenWebText | C | 2000000 | 10000000 | uniform | 2.8629 | 17.769 | +0.0000 | +0.000 | 393216/393216 |
| OpenWebText | C | 2000000 | 10000000 | mono_inc | 2.8883 | 18.205 | -0.0254 | -0.436 | 393216/393216 |
| OpenWebText | C | 2000000 | 10000000 | mono_dec | 2.8634 | 17.752 | -0.0005 | +0.017 | 393216/393216 |

Notes:
- Delta is defined as uniform - method; positive values indicate improvement over uniform.
- H2O-style and sliding-window diagnostics do not involve training seeds; this table reports deterministic evaluation-window robustness.
- Results should be interpreted as operator/window sensitivity evidence, not universal optimality.
