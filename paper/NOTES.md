# Research Paper Traceability, Verification, and Reconciliation Notes

**Manuscript Title:** Lightweight Reconstruction Modeling and Autonomous Extreme Value Thresholding for Syslog Anomaly Detection  
**Target Format:** IEEE Conference Format (`\documentclass[conference]{IEEEtran}`, Two Columns, Max 6 Pages)  
**Location:** `paper/`  
**Date of Audit:** November 17, 2026  

---

## 1. Authoritative Number Manifest and Source Traceability

Every quantitative metric, count, parameter dimension, and latency measurement reported in `paper/main.tex` is strictly traced to its canonical source artifact in the repository:

| Metric / Parameter | Value in Manuscript | Authoritative Source File / Script | Exact Column / Variable / Row |
| :--- | :---: | :--- | :--- |
| **HDFS Total Blocks** | 575,061 | `SPEC.md`, `data/HDFS_v1/anomaly_label.csv` | Dataset row count |
| **HDFS Anomalous Blocks** | 16,838 ($2.93\%$) | `SPEC.md`, `data/HDFS_v1/anomaly_label.csv` | Labeled anomalous rows |
| **HDFS Normal Blocks** | 558,223 ($97.07\%$) | `SPEC.md`, `data/HDFS_v1/anomaly_label.csv` | Labeled normal rows |
| **HDFS Base Protocol Subset** | 25,000 norm + 1,000 anom | `SPEC.md`, F{\"a}lt et al. 2025 Table I | Experimental protocol |
| **HDFS PCA Baseline Claim** | Prec 92.90, Rec 96.45, F1 94.64 | F{\"a}lt et al. (IEEE ICSRS 2025 Table I) | Base paper target |
| **HDFS Our PCA Reproduction** | Prec 92.34, Rec 96.96, F1 94.60 | `docs/archive/05_final_results.md` | `evaluate_pca_rpca_v3.py` |
| **HDFS Our PCA + POT F1** | $95.95\% \pm 1.47\%$ | `docs/archive/05_final_results.md` | Mean over 5 folds |
| **HDFS RPCA Baseline Claim** | Prec 89.88, Rec 91.44, F1 90.55 | F{\"a}lt et al. (IEEE ICSRS 2025 Table I) | Base paper target |
| **HDFS Our RPCA Best (30 seeds)**| $87.38\%$ (Mean $85.14\%$) | `docs/archive/05_final_results.md` | Max F1 over 30 seeds |
| **HDFS PCA Model Parameters** | 203 parameters | Mathematical matrix dimensions | $29 \times 6 = 174$ weights + 29 offsets |
| **HDFS Serialized Size** | $\sim 1.2\text{ KB}$ | Pickled PCA model artifact | File size on disk |
| **BGL Raw Log Lines** | 4,747,963 | `syslog_v2_clean/RESULTS.md` | Total raw line count |
| **BGL Drain Templates** | 1,848 | `data/BGL/preprocessed/BGL.log_templates.csv` | Unique induced templates |
| **BGL Total Sequences** | 1,286,817 | `syslog_v2_clean/RESULTS.md` | Node + 6h windows |
| **BGL Normal Sequences** | 1,157,289 ($89.93\%$) | `syslog_v2_clean/RESULTS.md` | Sequences with $y=0$ |
| **BGL Anomalous Sequences** | 129,528 ($10.07\%$) | `syslog_v2_clean/RESULTS.md` | Sequences with $y=1$ |
| **BGL Train Set Size (Normals)**| 937,404 ($0$ anomalies) | `syslog_v2_clean/data_utils.py` | 90% of non-test normals |
| **BGL Val Set Size** | 104,156 norm + 20,000 anom | `syslog_v2_clean/data_utils.py` | 10% non-test norm + subsample |
| **BGL Test Set Size per Fold** | 128,682 ($115,729$ N + $12,953$ A) | `syslog_v2_clean/data_utils.py` | 10% stratified test fold |
| **BGL Natural Test Prevalence** | **10.07%** | `syslog_v2_clean/test_results_10fold.csv` | Test anomalies / Total |
| **Mode A Precision** | $93.76\% \pm 2.54\%$ | `syslog_v2_clean/test_results_10fold.csv` | `mode_a_p` mean and std |
| **Mode A Recall** | $82.61\% \pm 1.96\%$ | `syslog_v2_clean/test_results_10fold.csv` | `mode_a_r` mean and std |
| **Mode A F1-Score** | $\mathbf{87.80\% \pm 1.31\%}$ | `syslog_v2_clean/test_results_10fold.csv` | `mode_a_f1` mean and std |
| **Mode A Mean Threshold $\theta_A$**| $0.2849 \pm 0.0493$ | `syslog_v2_clean/test_results_10fold.csv` | `mode_a_theta` mean |
| **Mode A Fallback Used** | **0 / 10** ($0.0\%$) | `syslog_v2_clean/test_results_10fold.csv` | `mode_a_fallback` sum |
| **Mode B Precision** | $93.06\% \pm 2.59\%$ | `syslog_v2_clean/test_results_10fold.csv` | `mode_b_p` mean and std |
| **Mode B Recall** | $84.71\% \pm 1.19\%$ | `syslog_v2_clean/test_results_10fold.csv` | `mode_b_r` mean and std |
| **Mode B F1-Score** | $\mathbf{88.66\% \pm 1.07\%}$ | `syslog_v2_clean/test_results_10fold.csv` | `mode_b_f1` mean and std |
| **Mode B Mean Threshold $\theta_B$**| $0.2579 \pm 0.0384$ | `syslog_v2_clean/test_results_10fold.csv` | `mode_b_theta` mean |
| **Mean PR-AUC** | $\mathbf{91.05\% \pm 0.32\%}$ | `syslog_v2_clean/test_results_10fold.csv` | `pr_auc` mean and std |
| **Mean ROC-AUC** | $\mathbf{97.34\% \pm 0.15\%}$ | `syslog_v2_clean/test_results_10fold.csv` | `roc_auc` mean and std |
| **Balanced View Mode A F1** | $\mathbf{90.16\% \pm 1.18\%}$ | `syslog_v2_clean/test_balanced_results_10fold.csv` | `mode_a_f1` on 50% prev |
| **Balanced View Mode B F1** | $\mathbf{91.36\% \pm 0.63\%}$ | `syslog_v2_clean/test_balanced_results_10fold.csv` | `mode_b_f1` on 50% prev |
| **Ablation AE Mode A F1** | $87.80\%$ | `syslog_v2_clean/ablation_results.csv` | Autoencoder row |
| **Ablation PCA (k=48) Mode A F1**| $59.97\% \pm 18.89\%$ | `syslog_v2_clean/ablation_results.csv` | PCA (k=48) row |
| **Ablation PCA Mode B F1** | $85.50\% \pm 1.54\%$ | `syslog_v2_clean/ablation_results.csv` | PCA (k=48) row |
| **Ablation PCA PR-AUC** | $84.26\% \pm 2.37\%$ | `syslog_v2_clean/ablation_results.csv` | PCA (k=48) row |
| **Ablation PCA ROC-AUC** | $96.78\% \pm 0.20\%$ | `syslog_v2_clean/ablation_results.csv` | PCA (k=48) row |
| **Ablation Isolation Forest Mode A**| $2.07\% \pm 1.83\%$ | `syslog_v2_clean/ablation_results.csv` | Isolation Forest row |
| **Ablation Isolation Forest Mode B**| $55.94\% \pm 13.91\%$ | `syslog_v2_clean/ablation_results.csv` | Isolation Forest row |
| **Ablation Isolation Forest PR-AUC**| $48.20\% \pm 6.32\%$ | `syslog_v2_clean/ablation_results.csv` | Isolation Forest row |
| **Ablation Random Guessing F1** | $12.34\%$ | `syslog_v2_clean/ablation_results.csv` | Random guessing row |
| **Ablation Predict All Normal F1**| $0.00\%$ | `syslog_v2_clean/ablation_results.csv` | Predict all normal row |
| **Chronological Mode A F1** | $\mathbf{0.03\%}$ ($P=76.92\%, R=0.02\%$) | `syslog_v2_clean/chronological_results.csv` | `mode_a_f1` |
| **Chronological Mode B F1** | $\mathbf{45.28\%}$ ($P=29.64\%, R=95.90\%$) | `syslog_v2_clean/chronological_results.csv` | `mode_b_f1` |
| **Chronological PR-AUC / ROC** | $30.32\%$ / $67.50\%$ | `syslog_v2_clean/chronological_results.csv` | `pr_auc`, `roc_auc` |
| **Chronological Test OOV Rate** | $\mathbf{68.91\%}$ ($935,199 / 1,357,066$) | `syslog_v2_clean/chronological_results.csv` | `oov_rate` |
| **Model Parameter Count** | **5,328** ($5,264$ trainable + $64$ mean) | `syslog_v2_clean/efficiency_results.csv` | `model_parameters` |
| **Serialized Model Pickle Size**| **70.25 KB** | `syslog_v2_clean/efficiency_results.csv` | `saved_model_kb` |
| **Single-Window Latency (Mean)**| **0.0432 ms** ($43.2\text{ }\mu\text{s}$) | `syslog_v2_clean/efficiency_results.csv` | `single_sample_mean_ms` |
| **Single-Window Latency (p99)** | **0.0476 ms** ($47.6\text{ }\mu\text{s}$) | `syslog_v2_clean/efficiency_results.csv` | `single_sample_p99_ms` |
| **Batch-256 Latency (Mean)** | **0.0959 ms** ($95.9\text{ }\mu\text{s}$) | `syslog_v2_clean/efficiency_results.csv` | `batch_256_mean_ms` |
| **Per-Window Batch Latency** | **0.37 microseconds** | `syslog_v2_clean/efficiency_results.csv` | `per_window_batch_latency_us` |
| **Batch-256 Latency (p99)** | **0.1111 ms** ($111.1\text{ }\mu\text{s}$) | `syslog_v2_clean/efficiency_results.csv` | `batch_256_p99_ms` |
| **Pipeline Throughput** | **62,076 windows/second** | `syslog_v2_clean/efficiency_results.csv` | `e2e_throughput_windows_per_sec` |
| **Benchmark Hardware CPU** | Apple M2 (ARM64, 8 cores, 8 GB RAM) | `syslog_v2_clean/efficiency_results.csv` | `hardware_cpu`, `hardware_ram` |
| **DeepLog Small-Data Baseline** | P 100.00, R 60.90, F1 75.70 | F{\"a}lt et al. 2025 Table I / Almodovar | Cited baseline |
| **LogBERT Small-Data Baseline** | P 24.38, R 78.80, F1 37.24 | F{\"a}lt et al. 2025 Table I / Almodovar | Cited baseline |
| **LogFIT Small-Data Baseline** | P 99.78, R 90.70, F1 95.02 | F{\"a}lt et al. 2025 Table I / Almodovar | Cited baseline |

---

## 2. Document Conflicts and Authoritative Resolutions

In accordance with Section 1 and Section 2 of the instructions, all discrepancies between legacy project documents and canonical files (`syslog_v2_clean/RESULTS.md` and related CSVs) are documented and resolved below:

| Discrepancy Topic | Legacy Document Value | Canonical `RESULTS.md` Value | Cause and Resolution |
| :--- | :--- | :--- | :--- |
| **BGL Detection F1** | $\sim 90.44\%$ (preliminary draft), $87.46\%$ (Review 1) | $\mathbf{87.80\% \pm 1.31\%}$ (Mode A) and $\mathbf{88.66\% \pm 1.07\%}$ (Mode B) | **Base-rate prevalence distortion:** Older scripts pooled all 129,528 anomalies into every test fold, creating an artificial $\sim 52.8\%$ test anomaly prevalence. When our leak-free models are tested on a balanced view ($50\%$ prevalence), they achieve **$90.16\%$** (Mode A) and **$91.36\%$** (Mode B). The legacy script also utilized seed retry loops (`if f1 < 0.80`). Section VI.B explicitly deconstructs this shift. |
| **Model Parameters** | 1,384 parameters | **5,328 parameters** | Older iteration used 32-d mean pooling with architecture $32 \to 16 \to 8 \to 16 \to 32$. The authoritative V2 model uses concatenated Mean+Max pooling (64-d) with architecture $64 \to 32 \to 16 \to 32 \to 64$, comprising 5,264 trainable weights/biases plus 64 centering mean offsets ($5,328$ total). |
| **Serialized Model Size** | 26.58 KB | **70.25 KB** | Corresponds to the expanded 64-d architecture and serialized metadata in Python pickle format. |
| **Inference Latency** | $0.0001\text{--}0.0002\text{ ms}$ | **0.0432 ms** ($43.2\text{ }\mu\text{s}$) single, **0.00037 ms** ($0.37\text{ }\mu\text{s}$) batch | Older measurement clocked only a simulated tensor forward pass without numpy/python overhead. The authoritative benchmark measures the complete unbatched and batched evaluation functions with warmup and 1,000 iterations. |
| **HDFS Thresholding** | 95th percentile test oracle | **EVT Peaks-Over-Threshold (POT)** | The base paper (F{\"a}lt et al.) used a test-set oracle threshold. Our contribution is replacing it with label-free POT fitted strictly on validation normal errors ($95.95\% \pm 1.47\%$). |

---

## 3. Items Marked for Verification (`[VERIFY]`)

The following literature comparison figures in Table~\ref{tab:published_reference_points} (Table VI) are explicitly tagged `[VERIFY]` in the manuscript:
1. **DeepLog HDFS F1 ($\sim 0.96$):** Reported in Du et al. (CCS 2017) using session grouping with 1% normal training split.
2. **LogAnomaly HDFS ($\sim 0.96$) and BGL ($\sim 0.96$):** Reported in Meng et al. (IJCAI 2019) using Template2Vec semantic vectors.
3. **PLELog HDFS (0.96) and BGL (0.98):** Reported in Yang et al. (ICSE 2021) using semi-supervised probabilistic label estimation.
4. **LogRobust HDFS (0.99) and BGL (0.75):** Reported in Zhang et al. (ESEC/FSE 2019); BGL result as cited in the NeuralLog comparative evaluation table.
5. **NeuralLog HDFS (0.98) and BGL (0.98):** Reported in Le & Zhang (ASE 2021) using BERT semantic extraction with supervised classification.
6. **LogFIT Citation Details:** Christian Almodovar et al. cited in F{\"a}lt et al. (IEEE ICSRS 2025 Table I). The original conference/journal venue was not explicitly specified in the ICSRS 2025 text; cited as technical report / literature comparison.

---

## 4. Items Excluded Due to Lack of Direct Evidence

Under the prompt instructions:
* *"Inference on saved models is allowed only if PROJECT_MAP.md says the result is already there; otherwise leave it out and say so in NOTES.md."*
* *"Include HDFS PR-AUC/ROC-AUC, natural-prevalence results and re-measured efficiency ONLY if PROJECT_MAP.md confirms they exist; label them 'supplementary' and say they come from saved models."*

**Audit Outcome:**
A comprehensive scan of the repository confirmed that no file named `PROJECT_MAP.md` exists. Therefore, in strict adherence to the precedence hierarchy:
1. **HDFS PR-AUC and ROC-AUC** are excluded from the manuscript.
2. **HDFS Natural-Prevalence Evaluation** (testing on the full 575,061 blocks without 25k/1k subsampling) is excluded from the manuscript.
3. **Re-measured HDFS Streaming Efficiency** is excluded from the manuscript; only the analytical parameter count (203 parameters, $\sim 1.2\text{ KB}$) is reported.
All these items are excluded to ensure 100% adherence to verified evidence.

---

## 5. Compliance with Writing and Vocabulary Guidelines

1. **Forbidden Phrases Audit:**
   * `"state of the art"` / `"SOTA"`: Zero occurrences in `paper/main.tex`. Replaced with *"published baselines"*, *"competitive methods"*, or specific paper citations.
   * `"outperforms DeepLog/LogBERT"` (unqualified): Zero occurrences. All comparisons explicitly contextualized under the restricted small-data protocol of F{\"a}lt et al.
   * `"strictly required"`: Zero occurrences. Replaced with *"demonstrates clear utility"* or *"is necessary in this setting"*.
   * `"disproved the base paper"`: Zero occurrences. Replaced with *"could not be reproduced across 30 random seeds"*.
   * `"zero leakage"`: Zero occurrences. Replaced with *"no test labels are used in any decision"*.

2. **Core Observation Framing:**
   Framed explicitly as an empirical observation based on two benchmarks, not a universal law:  
   *"Observation (based on two datasets only): On HDFS, where grouping by session identifier yields clean execution traces, a lightweight linear PCA is sufficient; on BGL, where multi-node streams are interleaved, a non-linear autoencoder on continuous semantic features demonstrated superior discriminative power."*

3. **Page Length Compliance:**
   The manuscript contains 2,757 prose body words, 7 compact tables, 1 TikZ architectural diagram, and 21 references, strictly formatting to $\sim 5.7\text{--}5.9$ pages in two-column `IEEEtran` conference format (maximum 6 pages).

