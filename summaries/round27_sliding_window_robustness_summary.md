# R27 Sliding-window Evaluation-window Robustness Diagnostic

This is inference-time cross-operator diagnostic evidence, not a sliding-window benchmark or StreamingLLM reproduction.

## Decisions

- **C4**: `preferred_direction_differs_under_window_level_attention`
- **OpenWebText**: `preferred_direction_differs_under_window_level_attention`

## Mean across deterministic windows

| Dataset | Method | Windows | Val Loss mean+/-std (lower) | Test PPL mean+/-std (lower) | Delta Val mean+/-std (higher) | Delta PPL mean+/-std (higher) | Val wins | PPL wins | avg window | tok/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C4 | mono_dec | 3 | 6.8179+/-0.0171 | 915.755+/-43.676 | +0.2996+/-0.0049 | +322.525+/-19.878 | 3/3 | 3/3 | 512 | 118910.9 |
| C4 | mono_inc | 3 | 7.1305+/-0.0205 | 1274.035+/-64.535 | -0.0129+/-0.0098 | -35.755+/-4.159 | 0/3 | 0/3 | 512 | 116874.0 |
| C4 | uniform | 3 | 7.1176+/-0.0142 | 1238.280+/-63.553 | +0.0000+/-0.0000 | +0.000+/-0.000 | 0/3 | 0/3 | 512 | 119339.7 |
| OpenWebText | mono_dec | 3 | 6.6440+/-0.0138 | 790.463+/-25.880 | +0.3273+/-0.0112 | +301.840+/-11.120 | 3/3 | 3/3 | 512 | 117830.8 |
| OpenWebText | mono_inc | 3 | 7.0511+/-0.0158 | 1176.987+/-52.014 | -0.0799+/-0.0108 | -84.684+/-16.157 | 0/3 | 0/3 | 512 | 117914.6 |
| OpenWebText | uniform | 3 | 6.9713+/-0.0249 | 1092.303+/-36.356 | +0.0000+/-0.0000 | +0.000+/-0.000 | 0/3 | 0/3 | 512 | 119528.2 |

## Per-window results

| Dataset | Window | Val skip | Test skip | Method | Val Loss (lower) | Test PPL (lower) | Loss after active | Delta Val (higher) | Delta PPL (higher) | tokens |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---|
| C4 | A | 0 | 8000000 | uniform | 7.1291 | 1268.047 | 8.4227 | +0.0000 | +0.000 | 393216/393216 |
| C4 | A | 0 | 8000000 | mono_inc | 7.1527 | 1299.904 | 8.6735 | -0.0236 | -31.857 | 393216/393216 |
| C4 | A | 0 | 8000000 | mono_dec | 6.8279 | 936.326 | 8.5628 | +0.3012 | +331.720 | 393216/393216 |
| C4 | B | 1000000 | 9000000 | uniform | 7.1017 | 1281.487 | 8.3910 | +0.0000 | +0.000 | 393216/393216 |
| C4 | B | 1000000 | 9000000 | mono_inc | 7.1124 | 1321.621 | 8.6134 | -0.0107 | -40.134 | 393216/393216 |
| C4 | B | 1000000 | 9000000 | mono_dec | 6.7982 | 945.346 | 8.5270 | +0.3035 | +336.141 | 393216/393216 |
| C4 | C | 2000000 | 10000000 | uniform | 7.1218 | 1165.306 | 8.4178 | +0.0000 | +0.000 | 393216/393216 |
| C4 | C | 2000000 | 10000000 | mono_inc | 7.1263 | 1200.579 | 8.6439 | -0.0044 | -35.273 | 393216/393216 |
| C4 | C | 2000000 | 10000000 | mono_dec | 6.8277 | 865.592 | 8.5660 | +0.2941 | +299.714 | 393216/393216 |
| OpenWebText | A | 0 | 8000000 | uniform | 6.9515 | 1078.854 | 8.2926 | +0.0000 | +0.000 | 393216/393216 |
| OpenWebText | A | 0 | 8000000 | mono_inc | 7.0353 | 1163.031 | 8.5952 | -0.0838 | -84.177 | 393216/393216 |
| OpenWebText | A | 0 | 8000000 | mono_dec | 6.6322 | 784.398 | 8.4499 | +0.3192 | +294.457 | 393216/393216 |
| OpenWebText | B | 1000000 | 9000000 | uniform | 6.9992 | 1133.467 | 8.3613 | +0.0000 | +0.000 | 393216/393216 |
| OpenWebText | B | 1000000 | 9000000 | mono_inc | 7.0669 | 1234.555 | 8.6573 | -0.0677 | -101.088 | 393216/393216 |
| OpenWebText | B | 1000000 | 9000000 | mono_dec | 6.6591 | 818.837 | 8.5265 | +0.3401 | +314.629 | 393216/393216 |
| OpenWebText | C | 2000000 | 10000000 | uniform | 6.9631 | 1064.588 | 8.3096 | +0.0000 | +0.000 | 393216/393216 |
| OpenWebText | C | 2000000 | 10000000 | mono_inc | 7.0512 | 1133.374 | 8.6251 | -0.0881 | -68.787 | 393216/393216 |
| OpenWebText | C | 2000000 | 10000000 | mono_dec | 6.6407 | 768.155 | 8.4881 | +0.3224 | +296.433 | 393216/393216 |

Notes:
- Delta is defined as uniform - method; positive values indicate improvement over uniform.
- Sliding-window diagnostics do not involve training seeds; this table reports deterministic evaluation-window robustness.
- Results should be interpreted as operator/window sensitivity evidence, not universal optimality.
