# R25 Sliding-window / Streaming-style Diagnostic on WikiText-103

Decision status: `window_operator_sensitivity_evidence_on_wikitext103`

mono_dec is best in this WikiText-103 sliding-window diagnostic; write as operator sensitivity evidence.

| Sparse platform | Sparse unit | Model | Dataset | Context | Method | Val Loss ↓ | Test PPL ↓ | Loss after active ↓ | Δ Val ↑ | Δ PPL ↑ | avg window | tok/s |
|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| Sliding-window-style diagnostic | local attention window | facebook/opt-350m | Salesforce/wikitext/wikitext-103-raw-v1 | 2048 | uniform | 7.5818 | 1986.469 | 8.8395 | +0.0000 | +0.000 | 512 | 119314.1 |
| Sliding-window-style diagnostic | local attention window | facebook/opt-350m | Salesforce/wikitext/wikitext-103-raw-v1 | 2048 | mono_inc | 7.5538 | 1926.784 | 8.9318 | +0.0281 | +59.684 | 512 | 118567.2 |
| Sliding-window-style diagnostic | local attention window | facebook/opt-350m | Salesforce/wikitext/wikitext-103-raw-v1 | 2048 | mono_dec | 7.3405 | 1554.199 | 8.8267 | +0.2413 | +432.270 | 512 | 118175.8 |

Notes:
- Delta is defined as uniform - method; positive values indicate improvement over uniform.
- This is lightweight cross-operator diagnostic evidence, not a full sliding-window benchmark or StreamingLLM reproduction.
- WikiText official validation/test splits may provide fewer than the requested 393,216 tokens; actual token counts are recorded in JSON metadata.
- The implementation uses evaluation-time attention masking and is not an optimized sparse runtime.
