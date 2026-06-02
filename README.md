# Layer-wise Sparse Budget Allocation

This repository contains the code, experiment configurations, lightweight diagnostics, and plotting utilities for the paper:

**Layer-wise Sparse Budget Allocation as an Operator-sensitive Design Dimension in Sparse Attention Systems**

The project studies **layer-wise sparse budget allocation** as an explicit design dimension in sparse attention systems. Rather than proposing a new sparse attention operator or a universal allocation rule, we ask a more controlled empirical question:

> Given the same average sparse budget, does the placement of that budget across Transformer depth affect model behavior?

We evaluate this question under matched compute or matched retention budgets across three sparse-unit settings:

- **NSA selected-block attention** as the main training-time controlled platform.
- **H2O-style token-level KV retention** as a lightweight inference-time diagnostic.
- **Sliding-window local attention** as a lightweight window-level diagnostic.

The main finding is that sparse budget placement is observable, but the preferred allocation direction is **operator-sensitive** and **dataset-sensitive**. In particular, the results should not be interpreted as showing that one schedule, such as `mono_inc`, is universally optimal.

## Scope

This repository supports the following parts of the study:

- NSA training and evaluation with layer-wise static sparse budget schedules.
- Cross-corpus NSA evaluation on C4, OpenWebText, and WikiText-103.
- H2O-style token-level KV retention diagnostics on OPT-350M.
- Sliding-window local-window diagnostics on OPT-350M.
- Behavioral diagnostics for NSA selected-block access patterns.
- Summary and plotting scripts for paper tables and figures.

This repository is **not**:

- a full H2O reproduction;
- a full StreamingLLM reproduction;
- a sparse attention SOTA benchmark;
- a leaderboard comparing different sparse attention systems by raw loss;
- a release of trained checkpoints or dataset caches.

Raw losses across platforms are not directly comparable. Only within-platform deltas relative to the matched uniform baseline are meaningful.

## Budget Schedules

The paper compares three main layer-wise allocation schedules.

For NSA selected-block attention, the average selected-block budget is matched at:

- average `actual_k = 16.0`
- estimated selected KV access `= 1024.0`

The main 24-layer schedules are:

| Schedule | Layer groups | Budget values |
|---|---:|---|
| `uniform` | all layers | `16` |
| `mono_inc` | layers 1-6 / 7-12 / 13-18 / 19-24 | `8 / 12 / 20 / 24` |
| `mono_dec` | layers 1-6 / 7-12 / 13-18 / 19-24 | `24 / 20 / 12 / 8` |

For H2O-style token-level KV retention:

- fixed recent window: `128`
- average heavy-hitter budget: `128`
- average total retained budget: `256`

For sliding-window diagnostics:

- context length: `2048`
- average local window size: `512`

## Repository Layout

```text
.
├── configs/                      # NSA model and budget schedule configs
├── native_sparse_attention/       # NSA model implementation
├── scripts/                       # Training, evaluation, and launcher scripts
├── diagnostics/                   # H2O-style, sliding-window, and behavior diagnostics
├── summaries/                     # Lightweight JSON/CSV/MD summaries used by the paper
├── figures/                       # Paper figures and plotting outputs
├── docs/                          # Reproducibility notes and audit tables
├── requirements.txt
└── README.md
```

Depending on the final cleanup, some scripts may remain at the repository root for compatibility with the original experiment pipeline.

## Main Components

### NSA Controlled Training Platform

NSA is used as the main controlled platform because its selected-block budget is explicit and can be matched across layer-wise schedules.

Relevant components:

```text
native_sparse_attention/
configs/
train.py
prepare_eval_cache.py
summarize_round21_wikitext_static.py
plot_operator_sensitive_results.py
```

### H2O-style Token-level KV Retention Diagnostic

This is a lightweight inference-time diagnostic on `facebook/opt-350m`. It is not intended to reproduce the full H2O system.

Relevant components:

```text
evaluate_h2o_kv_retention.py
summarize_h2o_kv_retention.py
summarize_h2o_window_robustness.py
```

### Sliding-window Local-window Diagnostic

This diagnostic applies layer-wise local attention window schedules to `facebook/opt-350m` at inference time.

Relevant components:

```text
evaluate_sliding_window_attention.py
audit_sliding_window_metrics.py
summarize_sliding_window_robustness.py
summarize_sliding_window_metric_audit.py
```

### NSA Behavioral Diagnostics

These diagnostics analyze layer-wise selected-block behavior and selected-attention-mass patterns within NSA.

Relevant components:

```text
analyze_layerwise_behavior.py
analyze_selected_attention_mass.py
summarize_layerwise_behavior_multiseed.py
summarize_p2plus_p3_update.py
```

## Reproducing Paper Figures

After preparing the lightweight summary files, run:

```bash
python plot_operator_sensitive_results.py
```

This generates the main operator-sensitive budget-placement figures and supplementary overview plots.

The deltas use the paper convention:

```text
Delta Val = ValLoss(uniform) - ValLoss(method)
Delta PPL = PPL_test(uniform) - PPL_test(method)
```

Positive values indicate improvement over the matched uniform baseline within the same platform, dataset, model, and average budget.

## Data

The experiments use public datasets:

- C4
- OpenWebText
- WikiText-103

Dataset caches are not included in this repository. Scripts expect users to prepare tokenized windows or caches locally according to the experiment notes.

## Checkpoints

Model checkpoints are not included.

For NSA experiments, trained checkpoints are large and should be regenerated if full reproduction is needed. For H2O-style and sliding-window diagnostics, the scripts use Hugging Face `facebook/opt-350m` at inference time.

## Results Included

This repository may include lightweight result summaries such as:

- final table JSON files;
- per-method summary CSV files;
- audit reports;
- figure source data;
- generated paper figures.

It does not include:

- raw dataset caches;
- full training logs;
- model checkpoints;
- remote machine logs;
- SSH or platform-specific configuration.

## Interpretation Boundary

The results should be interpreted as a compute-matched empirical study of budget placement.

Allowed interpretation:

- layer-wise sparse budget placement is an observable design dimension;
- allocation preference is operator-sensitive and dataset-sensitive;
- NSA selected-block attention can favor `mono_inc` on C4 and OpenWebText;
- token-level retention and local-window diagnostics may prefer different schedules.

Not claimed:

- `mono_inc` is universally optimal;
- these diagnostics are full H2O or StreamingLLM reproductions;
- raw losses can be compared across sparse platforms;
- this repository establishes a universal sparse allocation theory.

## Citation

If you use this code or build on this study, please cite the paper once it is available.

```bibtex
@misc{sparse_budget_allocation,
  title = {Layer-wise Sparse Budget Allocation as an Operator-sensitive Design Dimension in Sparse Attention Systems},
  author = {TBD},
  year = {2026},
  note = {Code release}
}
```

## License

This repository uses the MIT license. See `LICENSE` and `NOTICE.md`.

Repository URL:

```text
https://github.com/YuzQwQ/layer-wise-sparse-budget-allocation
```
