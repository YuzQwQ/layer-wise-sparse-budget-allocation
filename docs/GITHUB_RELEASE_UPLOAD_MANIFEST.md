# Paper Code Repository Upload Manifest

This manifest lists the recommended clean contents for a new paper-code GitHub repository for the sparse budget allocation manuscript.

Recommended repository name:

- `sparse-budget-allocation`
- `layer-wise-sparse-budget-allocation`

Recommended initial visibility:

- Private during submission preparation.
- Public only after advisor approval and after removing machine-specific logs, checkpoints, data caches, and credentials.

## 1. Repository Structure

```text
sparse-budget-allocation/
  README.md
  LICENSE
  CITATION.cff
  requirements.txt
  .gitignore
  configs/
  src/
  scripts/
  diagnostics/
  summaries/
  figures/
  docs/
```

## 2. Must Upload

These files are central to reproducing the paper-level experiments and figures.

### NSA Model, Training, and Budget Schedules

Use `native-sparse-attention` as the implementation base, but copy only the clean core rather than the whole current workspace.

```text
native-sparse-attention/README.md
native-sparse-attention/pyproject.toml
native-sparse-attention/setup.py
native-sparse-attention/train.py
native-sparse-attention/simple_train.py
native-sparse-attention/prepare_eval_cache.py
native-sparse-attention/evaluate_long_context.py
native-sparse-attention/native_sparse_attention/
native-sparse-attention/analysis/
```

Keep these config files:

```text
native-sparse-attention/configs/nsa_340M.json
native-sparse-attention/configs/nsa_340M_baseline.json
native-sparse-attention/configs/nsa_340M_phase1_monotone_inc.json
native-sparse-attention/configs/nsa_340M_phase1_monotone_dec.json
native-sparse-attention/configs/nsa_340M_phase1_u_shape.json
native-sparse-attention/configs/nsa_340M_phase1_u_shape_matched.json
native-sparse-attention/configs/nsa_340M_phase2a_tool.json
```

Optional, only if dynamic-budget exploratory results are included in supplementary reproduction:

```text
native-sparse-attention/configs/nsa_340M_phase2b_gate_scheduler.json
native-sparse-attention/configs/nsa_340M_phase2c_asymm.json
```

### Behavior Diagnostics

```text
native-sparse-attention/analyze_layerwise_behavior.py
native-sparse-attention/analyze_selected_attention_mass.py
native-sparse-attention/analyze_budget_attribution.py
summarize_layerwise_behavior_multiseed.py
summarize_p2plus_p3_update.py
```

### Cross-Operator Diagnostics

```text
evaluate_h2o_kv_retention.py
evaluate_sliding_window_attention.py
audit_sliding_window_metrics.py
summarize_h2o_kv_retention.py
summarize_h2o_c4_kv_retention.py
summarize_h2o_owt_kv_retention.py
summarize_h2o_window_robustness.py
summarize_sliding_window_wikitext.py
summarize_sliding_window_c4.py
summarize_sliding_window_owt.py
summarize_sliding_window_robustness.py
summarize_sliding_window_metric_audit.py
```

Remote launch/check scripts may be uploaded only after removing machine-specific hostnames, ports, passwords, and absolute remote filesystem paths. If not cleaned, do not upload them.

### Main Summary and Plotting

```text
plot_operator_sensitive_results.py
summarize_round12_static_reliability.py
summarize_round13_static_direction.py
summarize_round14_static_longtrain.py
summarize_round21_wikitext_static.py
summarize_round25_sliding_window_diagnostic.py
```

### Final Summary Files

Upload final lightweight summaries, not full experiment directories.

```text
output/doc/V56.1_deep_numeric_audit_report.md
output/doc/V56.1_result_source_audit_table_checked.md
output/doc/V56.1_result_source_audit_table_checked.csv
output/figures/operator_sensitive_results/
```

Recommended curated `summaries/` contents:

```text
summaries/round12_static_reliability_summary.json
summaries/round13_static_direction_summary.json
summaries/round14_static_longtrain_summary.json
summaries/round20_p2plus_p3_update_summary.json
summaries/round21_wikitext_static_summary.json
summaries/round22_h2o_kv_retention_summary.json
summaries/round26_h2o_window_robustness_summary.json
summaries/round27_sliding_window_robustness_summary.json
summaries/round27_sliding_window_metric_audit_summary.json
```

## 3. Useful but Optional

These can improve reproducibility, but should be cleaned before upload.

```text
ablation_remote_common.py
launch_ablation_remote.py
check_ablation_remote.py
remote_ablation_utils.py
summarize_round8_openwebtext.py
summarize_round9_mechanism.py
summarize_round10_budgetcal_longtrain.py
summarize_round11_mono_final.py
```

Upload them only if:

- paths are parameterized;
- SSH information is removed;
- result directories are not hard-coded to personal machines;
- no passwords, ports, tokens, or platform-specific credentials remain.

## 4. Do Not Upload

Do not upload these categories.

```text
results/**/ckpt_final.pt
results/**/*.pt
results/**/*.bin
results/**/*.safetensors
results/**/*.pkl
results/**/*.npy
results/**/*.npz
results/**/cache*
tmp/
tmp_c4_cache_smoke/
tmp_slimpajama_part000_head.bin
tmp_slimpajama_part000_head8mb.bin
__pycache__/
native-sparse-attention/**/__pycache__/
native-sparse-attention/3rdparty/
```

Also do not upload:

- raw C4 / OpenWebText / WikiText caches;
- Hugging Face model cache;
- remote stdout logs containing machine paths;
- SSH commands, passwords, ports, or platform account details;
- all old manuscript DOCX versions;
- old paper-editing scripts unless needed for paper reproduction.

## 5. Suggested `.gitignore`

```gitignore
# Python
__pycache__/
*.py[cod]
.pytest_cache/
.mypy_cache/

# Checkpoints and model/data caches
*.pt
*.bin
*.safetensors
*.pkl
*.npy
*.npz
*.gz
*.zip
*.tar
*.cache
cache/
data_cache/
hf_cache/

# Experiment logs and temporary outputs
tmp/
tmp_*/
results/**/ckpt*
results/**/cache*
results/**/stdout.log
results/**/stderr.log

# Documents and local manuscript drafts
*.docx
~$*.docx

# OS / editor
.DS_Store
Thumbs.db
.vscode/
.idea/
```

## 6. Minimal README Outline

```text
# Layer-wise Sparse Budget Allocation

This repository contains code and lightweight summaries for the paper:
"Layer-wise Sparse Budget Allocation as an Operator-sensitive Design Dimension in Sparse Attention Systems".

## Scope
- NSA selected-block controlled training platform.
- H2O-style token-level KV retention inference diagnostic.
- Sliding-window local-window inference diagnostic.
- Plotting and summary scripts for paper figures.

## Not Included
- Model checkpoints.
- Raw dataset caches.
- Full remote machine logs.

## Reproducing Main Tables
1. Prepare datasets / token caches.
2. Run NSA controlled experiments or use provided summary JSON files.
3. Run H2O-style and sliding-window diagnostics.
4. Run `plot_operator_sensitive_results.py`.

## Citation
See `CITATION.cff`.
```

## 7. Code Availability Sentence for Manuscript

If the repository is private during review:

```text
The training code, evaluation scripts, budget schedule configurations, and summary utilities used in this study will be released upon publication to support reproducibility.
```

If the repository is public:

```text
The training code, evaluation scripts, budget schedule configurations, and summary utilities used in this study are publicly available at https://github.com/<user>/<repo>.
```
