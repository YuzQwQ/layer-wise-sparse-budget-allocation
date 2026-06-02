# R22 H2O-style Token-level KV Retention Diagnostic

Decision status: `operator_sensitivity_evidence`

uniform is best in this H2O-style diagnostic; write as operator sensitivity evidence.

| Sparse platform | Sparse unit | Model | Dataset | Method | Val Loss ↓ | Test PPL ↓ | Δ Val ↑ | Δ PPL ↑ | recent size | avg HH budget | avg total retained | tok/s |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| H2O-style KV retention diagnostic | KV tokens | facebook/opt-350m | Salesforce/wikitext/wikitext-103-raw-v1 | uniform | 3.7877 | 42.778 | +0.0000 | +0.000 | 128 | 128 | 256 | 654.6 |
| H2O-style KV retention diagnostic | KV tokens | facebook/opt-350m | Salesforce/wikitext/wikitext-103-raw-v1 | mono_inc | 3.8034 | 43.384 | -0.0157 | -0.606 | 128 | 128 | 256 | 684.2 |
| H2O-style KV retention diagnostic | KV tokens | facebook/opt-350m | Salesforce/wikitext/wikitext-103-raw-v1 | mono_dec | 3.7892 | 42.860 | -0.0015 | -0.082 | 128 | 128 | 256 | 680.6 |

Notes:
- Delta is defined as uniform - method; positive values indicate improvement over uniform.
- This is limited cross-operator diagnostic evidence, not a full H2O benchmark or H2O reproduction.
- All methods are matched by average heavy-hitter budget and average total retained budget.
