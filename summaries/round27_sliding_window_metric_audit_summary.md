# R27 Sliding-window Metric Consistency Audit

This audit recomputes all-token loss, after512, and after768 from the same logits/labels/forward path.

## Mean across windows

| Dataset | Method | Windows | Val all | Val after512 | Val after768 | Delta all | Delta after512 | Delta after768 | Tokens all/512/768 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| C4 | uniform | 3 | 7.1176 | 8.4105 | 8.6069 | +0.0000 | +0.0000 | +0.0000 | 393024/294912/245760 |
| C4 | mono_inc | 3 | 7.1305 | 8.2239 | 8.6436 | -0.0129 | +0.1866 | -0.0368 | 393024/294912/245760 |
| C4 | mono_dec | 3 | 6.8179 | 7.9696 | 8.5519 | +0.2996 | +0.4409 | +0.0549 | 393024/294912/245760 |
| OpenWebText | uniform | 3 | 6.9713 | 8.3212 | 8.5540 | +0.0000 | +0.0000 | +0.0000 | 393024/294912/245760 |
| OpenWebText | mono_inc | 3 | 7.0511 | 8.1993 | 8.6258 | -0.0799 | +0.1219 | -0.0718 | 393024/294912/245760 |
| OpenWebText | mono_dec | 3 | 6.6440 | 7.8404 | 8.4882 | +0.3273 | +0.4808 | +0.0658 | 393024/294912/245760 |

## Ranking audit

- **C4 / loss**: winners per window = `mono_dec,mono_dec,mono_dec`, counts = `{'uniform': 0, 'mono_inc': 0, 'mono_dec': 3}`
- **C4 / loss_after_512**: winners per window = `mono_dec,mono_dec,mono_dec`, counts = `{'uniform': 0, 'mono_inc': 0, 'mono_dec': 3}`
- **C4 / loss_after_768**: winners per window = `mono_dec,mono_dec,mono_dec`, counts = `{'uniform': 0, 'mono_inc': 0, 'mono_dec': 3}`
- **OpenWebText / loss**: winners per window = `mono_dec,mono_dec,mono_dec`, counts = `{'uniform': 0, 'mono_inc': 0, 'mono_dec': 3}`
- **OpenWebText / loss_after_512**: winners per window = `mono_dec,mono_dec,mono_dec`, counts = `{'uniform': 0, 'mono_inc': 0, 'mono_dec': 3}`
- **OpenWebText / loss_after_768**: winners per window = `mono_dec,mono_dec,mono_dec`, counts = `{'uniform': 0, 'mono_inc': 0, 'mono_dec': 3}`

## Per-window deltas

| Dataset | Window | Method | Delta all | Delta after512 | Delta after768 |
|---|---|---|---:|---:|---:|
| C4 | A | uniform | +0.0000 | +0.0000 | +0.0000 |
| C4 | A | mono_inc | -0.0236 | +0.1741 | -0.0540 |
| C4 | A | mono_dec | +0.3012 | +0.4439 | +0.0567 |
| C4 | B | uniform | +0.0000 | +0.0000 | +0.0000 |
| C4 | B | mono_inc | -0.0107 | +0.1896 | -0.0344 |
| C4 | B | mono_dec | +0.3035 | +0.4451 | +0.0520 |
| C4 | C | uniform | +0.0000 | +0.0000 | +0.0000 |
| C4 | C | mono_inc | -0.0044 | +0.1960 | -0.0218 |
| C4 | C | mono_dec | +0.2941 | +0.4338 | +0.0562 |
| OpenWebText | A | uniform | +0.0000 | +0.0000 | +0.0000 |
| OpenWebText | A | mono_inc | -0.0838 | +0.1189 | -0.0815 |
| OpenWebText | A | mono_dec | +0.3192 | +0.4700 | +0.0637 |
| OpenWebText | B | uniform | +0.0000 | +0.0000 | +0.0000 |
| OpenWebText | B | mono_inc | -0.0677 | +0.1355 | -0.0594 |
| OpenWebText | B | mono_dec | +0.3401 | +0.4991 | +0.0714 |
| OpenWebText | C | uniform | +0.0000 | +0.0000 | +0.0000 |
| OpenWebText | C | mono_inc | -0.0881 | +0.1112 | -0.0746 |
| OpenWebText | C | mono_dec | +0.3224 | +0.4733 | +0.0624 |

Notes:
- Delta is uniform - method; positive values indicate improvement over uniform.
- The previous R27 `loss_after_window_active` was not comparable across methods because uniform used 512 excluded tokens while mono_inc/mono_dec used 768.
- This audit fixes the cutoff across all methods at 512 and 768.
