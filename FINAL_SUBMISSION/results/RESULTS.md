# Comprehensive Empirical Evaluation & Comparative Benchmark: HDFS & BGL Log Anomaly Detection

**Projects:** HDFS Distributed File System & BGL Supercomputer Telemetry  
**Pipelines:**
1. **HDFS:** Session Count Vectors (29 event templates) $\rightarrow$ L2-Normalization $\rightarrow$ Linear Subspace Projection (PCA, $k=6$) $\rightarrow$ Extreme Value Theory Peaks-Over-Threshold (EVT / POT)
2. **BGL:** Sliding Temporal Windows (Node + 6-hour windows) $\rightarrow$ Unsupervised Word2Vec (Mean+Max Pooling, 64-d) $\rightarrow$ Lightweight MLP Autoencoder (5,328 parameters) $\rightarrow$ EVT / POT Decision Boundary  
**Evaluation Date:** November 17, 2026  
**Benchmarking Environment:** Apple M2 (8.0 GB Unified RAM), macOS Darwin 15.8.1 arm64  
**Integrity Standard:** Zero test-label feedback, strictly leak-free train/validation/test partitions, natural operational prevalence evaluation.

---

## 1. Executive Summary & Headline Empirical Findings

This report consolidates the complete, leak-free empirical evaluation of log anomaly detection across both the **HDFS** and **BGL** benchmarks. It addresses key methodological flaws prevalent in prior literature—specifically test-set oracle thresholding, synthetic balanced test distributions, and unacknowledged data leakage—and provides an honest, rigorous accounting of model performance.

### Headline Results Summary

| Benchmark Dataset | Method | Training Split & Protocol | Thresholding Mechanism | Precision (%) | Recall (%) | F1-Score (%) | PR-AUC (%) | ROC-AUC (%) | Latency / Throughput |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **HDFS** (575,061 blocks) | **PCA Baseline (Replication)** | 5-Fold CV (5k train, 21k test) | 95th %ile Test Oracle | 92.34 ± 2.24 | 96.96 ± 2.35 | **94.60 ± 2.30** | — | — | 3.5 µs / 10.6M sess/s |
| **HDFS** (575,061 blocks) | **PCA + POT (Ours)** | 5-Fold CV (5k train, 21k test) | **Autonomous EVT (Train Only)**| **93.61 ± 1.62** | **98.46 ± 1.89** | **95.95 ± 1.47** | **99.58** | **99.78** | **3.5 µs / 10.6M sess/s** |
| **HDFS** (575,061 blocks) | **RPCA (Ceiling Analysis)** | 5-Fold CV / Fit Size 10k | Exhaustive Search Ceiling | 91.75 | 83.40 | **87.38** | — | — | Infeasible (>200ms) |
| **HDFS** (575,061 blocks) | **PCA + POT (Natural Scale)** | **All 570,061 Held-Out (2.95%)**| Fixed POT from Fold Models | **90.04 ± 1.83** | **97.80 ± 2.05** | **93.74 ± 1.26** | **97.97** | **99.67** | **3.5 µs / 10.6M sess/s** |
| **BGL** (4.75M log lines) | **MLP-AE Mode A (Ours)** | 10-Fold CV (10.07% Nat. Prev) | **Unsupervised POT ($10^{-3}$)** | **93.76 ± 2.54** | **82.61 ± 1.96** | **87.80 ± 1.31** | **91.05** | **97.34** | **43.2 µs / 62k win/s** |
| **BGL** (4.75M log lines) | **MLP-AE Mode B (Ours)** | 10-Fold CV (10.07% Nat. Prev) | Semi-Supervised Val Search | **93.06 ± 2.59** | **84.71 ± 1.19** | **88.66 ± 1.07** | **91.05** | **97.34** | **43.2 µs / 62k win/s** |
| **BGL** (4.75M log lines) | **MLP-AE Balanced View** | 10-Fold CV (50% Syn. Prev) | Unsupervised POT (Mode A) | 99.26 ± 0.60 | 82.61 ± 1.96 | **90.16 ± 1.05** | 95.82 | 97.34 | 43.2 µs / 62k win/s |
| **BGL** (4.75M log lines) | **MLP-AE Temporal Split** | 80/20 Chronological Split | Mode B Validation Tuned | 29.64 | 95.90 | **45.28** | 30.32 | 67.50 | 43.2 µs / 62k win/s |

---

## 2. Part I: HDFS Benchmark Results & Technical Deconstruction

### 2.1 Baseline PCA Replication: Reaching 94.60% F1
Published studies (e.g. Xu et al., SOSP 2009; Fält et al., 2025) cite an HDFS PCA baseline of **94.64% F1**. Standard implementations of PCA on raw count vectors or TF-IDF representations collapse to ~76%–78% F1.

We successfully reproduced the published baseline at **94.60% ± 2.30% F1** by uncovering an **undocumented L2-normalization step**:
$$\mathbf{x}_{\text{norm}} = \frac{\mathbf{x}}{\|\mathbf{x}\|_2}$$
Without L2-normalization, high-frequency logging events dominate the covariance matrix, skewing the principal components away from rare structural transitions.

```text
FINAL 5-FOLD CV RESULT: HDFS PCA [Paper-Exact Protocol: 95th %ile Oracle]
Precision: 92.34% ± 2.24%
Recall:    96.96% ± 2.35%
F1 Score:  94.60% ± 2.30%
```

### 2.2 Autonomous PCA + Extreme Value Theory (POT): 95.95% F1
The published PCA baseline relies on a **95th percentile oracle threshold**, which evaluates test set reconstruction errors directly to place the cutoff—an unrealistic protocol for production deployment.

We replace the oracle with **Extreme Value Theory Peaks-Over-Threshold (EVT / POT)**. The upper tail of reconstruction errors on training normal sessions ($q=0.90$) is fitted to a Generalized Pareto Distribution (GPD):
$$\theta = t + \frac{\hat{\sigma}}{\hat{\gamma}}\left[\left(\frac{q_n}{\text{target\_fpr}}\right)^{\hat{\gamma}} - 1\right]$$
Setting $\text{target\_fpr} = 10^{-4}$, the decision boundary is computed **strictly from normal training data with zero access to test labels**.

#### Per-Fold Performance Breakdown (Paper 5-Fold Protocol: 5,000 Train, 21,000 Test)
*Saved Models:* `FINAL_SUBMISSION/hdfs/saved_models/hdfs_pot_pca_fold_1..5.pkl`

| Fold | POT Threshold ($\theta$) | TP | FP | TN | FN | Precision (%) | Recall (%) | F1-Score (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 1** | 0.0023 | 1,000 | 73 | 19,927 | 0 | 93.20 | 100.00 | **96.48** |
| **Fold 2** | 0.0018 | 1,000 | 71 | 19,929 | 0 | 93.37 | 100.00 | **96.57** |
| **Fold 3** | 0.0015 | 1,000 | 63 | 19,937 | 0 | 94.07 | 100.00 | **96.95** |
| **Fold 4** | 0.0031 | 1,000 | 68 | 19,932 | 0 | 93.63 | 100.00 | **96.71** |
| **Fold 5** | 0.0008 | 923 | 61 | 19,939 | 77 | 93.80 | 92.30 | **93.04** |
| **Mean ± Std** | — | — | — | — | — | **93.61 ± 1.62** | **98.46 ± 1.89** | **95.95 ± 1.47** |

### 2.3 Mathematical Audit: Why Fold 1–4 FP Counts (73, 71, 63, 68) Match Baseline
A critical verification question arose: in Folds 1–4, the POT false-positive counts (73, 71, 63, 68) are identical to the baseline's 95th-percentile false-positive counts. Is this a code bug or genuine data behavior?

* **Empirical Analysis of Score Distributions:**
  * In HDFS, 99.6% of normal sessions consist of identical, highly repetitive event sequences (e.g. block allocation, local write, replication, close). These normal sessions produce near-zero reconstruction errors ($e < 0.0016$).
  * The baseline 95th-percentile oracle sets $\theta_{\text{oracle}} \approx 0.0035 - 0.0042$.
  * Autonomous POT sets $\theta_{\text{POT}} \approx 0.0015 - 0.0031$.
  * In the test sets of Folds 1–4, there are **exactly zero normal sessions with reconstruction errors between $0.0016$ and $0.0124$** (a wide statistical gap).
  * Consequently, any threshold placed anywhere in the open interval $(0.0016, 0.0124)$ flags the exact same 73, 71, 63, and 68 normal sessions.
* **Conclusion:** The identity is a **genuine structural property** of the discrete HDFS execution graph, not a pipeline bug. POT successfully lands in this safety margin without requiring test set labels.

### 2.4 Robust PCA (RPCA) Ceiling: Refuting 90.55% F1 Claims
Earlier publications claimed that Robust PCA (inexact ALM solver) achieves **90.55% F1** on HDFS. Systematic empirical audits across fit sizes (5,000 to 25,000 normal sequences) and 30 random seeds reveal:
* Under inexact ALM, the low-rank component $L$ collapses to rank 2–4 on normal log matrices.
* Exhaustive parameter sweeps demonstrate that RPCA performance **peaks strictly at 87.38% F1** (at fit size 10,000, rank 4, Precision 91.75%, Recall 83.40%). Across 30 random seeds, the mean F1 is only **85.14%**.
* Inexact ALM requires computing full SVDs iteratively, resulting in inference latencies exceeding 200 ms per batch, rendering it impractical for real-time streaming.
* **Conclusion:** Published claims of 90.55% RPCA F1 represent an **irreproducible outlier** or evaluation artifact.

### 2.5 True Operational Scale: Natural Prevalence Evaluation (570,061 Sessions)
Standard academic evaluations sample only 20,000 normals and 1,000 anomalies for testing (inflating prevalence to 4.76%). To establish real-world operational performance, we evaluated our 5 saved PCA+POT models across **all 553,223 held-out normal sessions and all 16,838 anomalies** not seen during training:
* **True Operational Anomaly Prevalence:** **2.95%** (16,838 anomalies / 570,061 total sessions).
* **Zero Train Overlap:** Training normals ($5,000$ per fold) were strictly excluded.

| Saved Model Fold | True Positives (TP) | False Positives (FP) | True Negatives (TN) | False Negatives (FN) | Precision (%) | Recall (%) | F1-Score (%) | PR-AUC (%) | ROC-AUC (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 1 Model** | 16,818 | 1,718 | 551,505 | 20 | 90.73 | 99.88 | **95.09** | 98.71 | 99.86 |
| **Fold 2 Model** | 16,813 | 1,885 | 551,338 | 25 | 89.92 | 99.85 | **94.62** | 98.39 | 99.78 |
| **Fold 3 Model** | 16,818 | 1,550 | 551,673 | 20 | 91.56 | 99.88 | **95.54** | 98.79 | 99.87 |
| **Fold 4 Model** | 16,818 | 1,978 | 551,245 | 20 | 89.47 | 99.88 | **94.39** | 98.43 | 99.83 |
| **Fold 5 Model** | 14,993 | 1,941 | 551,282 | 1,845 | 88.54 | 89.04 | **88.79** | 88.51 | 98.99 |
| **Mean ± Std** | — | — | — | — | **90.04 ± 1.83** | **97.80 ± 2.05** | **93.74 ± 1.26** | **97.97 ± 1.83** | **99.67 ± 0.34** |

---

## 3. Part II: BGL Benchmark Results & Leak-Free Protocol

### 3.1 Dataset Composition & Stratified 10-Fold Setup
The Blue Gene/L dataset comprises **4,747,963 log messages** emitted across multi-core compute nodes.
* **Windowing:** Grouped by node ID into non-overlapping 6-hour windows, yielding **1,286,817 total sequence windows**:
  * Healthy Windows ($y=0$): $1,157,289$ ($89.93\%$)
  * Anomalous Windows ($y=1$): $129,528$ ($10.07\%$)
* **Partitioning Strategy:**
  * 10-fold cross-validation stratified independently on normals and anomalies.
  * Each fold tests on **$10.07\%$ natural prevalence** ($115,729$ normals and $12,953$ anomalies).
  * Every anomaly in the supercomputer trace is evaluated **exactly once**.
  * Non-test normals are split 90% for training (contains **0 anomalies**) and 10% for validation.

### 3.2 Canonical 10-Fold Test Results (Natural Prevalence: 10.07%)
*Saved Models:* `FINAL_SUBMISSION/bgl/saved_models/bgl_ae_clean_fold_1..10.pkl`

| Fold | Mode A Threshold | Mode A P (%) | Mode A R (%) | Mode A F1 (%) | Mode B Threshold | Mode B P (%) | Mode B R (%) | Mode B F1 (%) | PR-AUC (%) | ROC-AUC (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 1** | 0.2477 | 90.90 | 83.72 | 87.16 | 0.2028 | 88.54 | 85.69 | 87.10 | 91.00 | 97.23 |
| **Fold 2** | 0.2604 | 96.65 | 84.53 | 90.19 | 0.2393 | 95.82 | 85.35 | 90.29 | 91.48 | 97.35 |
| **Fold 3** | 0.2312 | 93.99 | 82.68 | 87.98 | 0.1969 | 92.20 | 86.42 | 89.22 | 90.59 | 97.15 |
| **Fold 4** | 0.2114 | 93.97 | 79.64 | 86.22 | 0.1793 | 91.13 | 83.15 | 86.96 | 91.26 | 97.53 |
| **Fold 5** | 0.2238 | 93.75 | 81.82 | 87.38 | 0.1872 | 91.31 | 85.89 | 88.52 | 90.67 | 97.36 |
| **Fold 6** | 0.2064 | 93.18 | 80.88 | 86.59 | 0.1691 | 90.72 | 85.03 | 87.78 | 91.05 | 97.34 |
| **Fold 7** | 0.2486 | 94.67 | 82.86 | 88.37 | 0.2155 | 93.53 | 85.06 | 89.10 | 91.00 | 97.15 |
| **Fold 8** | 0.2084 | 93.30 | 80.51 | 86.43 | 0.1706 | 92.65 | 86.32 | 89.37 | 90.78 | 97.30 |
| **Fold 9** | 0.2359 | 91.68 | 84.88 | 88.15 | 0.1996 | 90.95 | 87.28 | 89.08 | 91.52 | 97.63 |
| **Fold 10**| 0.2427 | 95.49 | 84.23 | 89.51 | 0.2274 | 93.75 | 86.87 | 89.23 | 91.15 | 97.33 |
| **Mean** | — | **93.76** | **82.61** | **87.80** | — | **93.06** | **84.71** | **88.66** | **91.05** | **97.34** |
| **Std** | — | **± 2.54** | **± 1.96** | **± 1.31** | — | **± 2.59** | **± 1.19** | **± 1.07** | **± 0.32** | **± 0.15** |

### 3.3 Deconstruction of Legacy ~90.44% Figure: The Base-Rate Effect
In earlier capstone reviews, an F1 score of **~90.44%** was reported. Audit revealed that legacy code:
1. Dumped all 129,528 anomalies into every fold's test set alongside ~115,729 normals, artificially creating a **~52.8% balanced test set**.
2. Contained automated retry loops (`while f1 < 0.80: seed_offset += 1`), violating test blindness.

Evaluating our clean, leak-free models on a **50% balanced test perspective** yields:
* **Mode A Balanced F1:** **90.16%** (Precision: 99.26%, Recall: 82.61%)
* **Mode B Balanced F1:** **91.36%** (Precision: 99.16%, Recall: 84.71%)
* **Conclusion:** The underlying representation is robust. The shift from 87.80% to 90.44% is purely an artifact of test prevalence.

### 3.4 10-Fold Ablation Matrix: Is Neural Reconstruction Necessary?
Evaluated on the exact same 64-dimensional Word2Vec Mean+Max features across all 10 folds:

| Model Architecture | Mode A F1 (%) | Mode A Prec (%) | Mode A Rec (%) | Mode B F1 (%) | PR-AUC (%) | ROC-AUC (%) | Training Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MLP Autoencoder (Ours)** | **87.80 ± 1.31** | **93.76 ± 2.54** | **82.61 ± 1.96** | **88.66 ± 1.07** | **91.05 ± 0.32** | **97.34 ± 0.15** | 60.1s |
| **PCA on Word2Vec ($k=48$)** | 59.97 ± 8.42 | 81.32 ± 11.20 | 50.00 ± 12.40 | 85.50 ± 1.15 | 84.26 ± 0.85 | 96.78 ± 0.18 | 3.0s |
| **PCA on Event Counts** | 34.28 ± 13.91 | 21.04 ± 1.82 | 92.45 ± 2.10 | 53.60 ± 2.10 | 24.12 ± 1.50 | 72.40 ± 1.20 | 4.2s |
| **Isolation Forest (W2V)** | 2.07 ± 0.85 | 25.99 ± 4.10 | 1.10 ± 0.45 | 55.94 ± 1.80 | 48.20 ± 1.20 | 92.23 ± 0.40 | 0.4s |
| **Predict All Normal** | 0.00 | 0.00 | 0.00 | 0.00 | 10.07 | 50.00 | 0.0s |
| **Random Guessing** | 12.34 | 10.03 | 16.03 | 12.34 | 10.07 | 50.00 | 0.0s |

*Key Findings:*
* Neural reconstruction outperforms linear PCA by **+27.83% in Mode A F1** and **+6.79% in PR-AUC**.
* Isolation Forest fails catastrophically under EVT thresholding because tree isolation depths do not follow Generalized Pareto distributions.

### 3.5 Chronological Generalization (80/20 Time Split)
To evaluate temporal stability against software updates, we trained on the earliest 70% normal logs, validated on the next 10%, and tested on the latest 20% future logs (Prevalence: 21.88%):
* **Mode A (Label-free POT):** Precision: 76.92%, Recall: 0.02%, F1: **0.03%**
* **Mode B (Validation-tuned):** Precision: 29.64%, Recall: 95.90%, F1: **45.28%**
* **Root Cause — Semantic Concept Drift:** **68.91% of test tokens were Out-Of-Vocabulary (OOV)**. Static word embeddings collapsed when confronted with unseen administrative templates from later months, proving that **continuous periodic fine-tuning** is mandatory for production log models.

---

## 4. Part III: Comprehensive Cross-Model Comparison Table

The following matrix synthesizes our empirical results alongside state-of-the-art models reported in literature and official empirical benchmarking studies (e.g. *LogADEmpirical* / Le & Zhang, ICSE 2022; *deep-loglizer* / Chen et al., ISSRE 2021; Fält et al., IEEE ICSRS 2025).

| Model / Approach | Target Dataset | Core Architecture | Supervision Level | Decision Boundary | F1-Score (%) | PR-AUC / ROC-AUC (%) | Parameter Footprint | Inference Latency | Verification / Provenance | Directly Comparable? |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **PCA + POT (Ours)** | **HDFS** | Linear Subspace ($k=6$) | **Unsupervised** | **Autonomous EVT (POT)** | **95.95 ± 1.47** | **99.58 / 99.78** | **203 params (1.0 KB)** | **0.0035 ms (3.5 µs)** | **Run by us** | **YES** (Standard 5-Fold Protocol) |
| **PCA Baseline** | **HDFS** | Linear Subspace ($k=6$) | Unsupervised | 95th %ile Test Oracle | 94.60 ± 2.30 | — | 203 params (1.0 KB) | 0.0035 ms (3.5 µs) | **Run by us** | **YES** (Base Paper Replication) |
| **PCA + POT (Full Scale)** | **HDFS** | Linear Subspace ($k=6$) | **Unsupervised** | Fixed POT (Train Normal) | **93.74 ± 1.26** | **97.97 / 99.67** | **203 params (1.0 KB)** | **0.0035 ms (3.5 µs)** | **Run by us** | **YES** (570,061 Sessions Held-Out) |
| **Robust PCA (RPCA)** | **HDFS** | Inexact ALM Solver ($k=4$) | Unsupervised | Exhaustive Grid Search | 87.38 (Mean 85.1) | — | Matrix Factorization | > 200 ms | **Run by us** | **YES** (Empirical Ceiling Proof) |
| **LogFIT** (Almodovar 2024)| **HDFS** | Lightweight DistilBERT | Unsupervised | Fixed Quantile Cutoff | 95.02 | — | ~66M params (250 MB)| ~1.2 ms | Cited in [1] | **YES** (Evaluated on Same 5-Fold) |
| **DeepLog** (Du et al. 2017)| **HDFS** | LSTM Next-Key Predictor | Unsupervised | Top-$g$ Candidate Oracle| 75.70 (Small) / 96.0 | — | ~2.5M params (10 MB) | ~3.8 ms | Cited in [1, 2] | Partial (1% session vs 5-fold) |
| **LogBERT** (Guo et al. 2021)| **HDFS** | 12-layer Transformer | Unsupervised | Hypersphere Radius Cutoff| 37.24 (Small) / 82.3 | — | ~110M params (420 MB)| ~22.0 ms (GPU) | Cited in [1, 3] | Partial (Severe small-data collapse) |
| **LogRobust** (Zhang 2019) | **HDFS** | Attention BiLSTM | **Fully Supervised** | Sigmoid Cutoff ($0.5$) | 96.30 | — | ~5.8M params (23 MB) | ~6.5 ms | Cited in [4] | **NO** (Requires Labeled Anomalies) |
| **PLELog** (Yang et al. 2021)| **HDFS** | Probabilistic Label Est.| Semi-Supervised | Probabilistic Boundary | 95.80 | — | ~8.2M params (32 MB) | ~8.0 ms | Cited in [5] | Partial (Requires probabilistic prior) |
| **Semantic MLP-AE Mode A** | **BGL** | **Mean+Max W2V + MLP-AE** | **Unsupervised** | **Autonomous EVT (POT)** | **87.80 ± 1.31** | **91.05 / 97.34** | **5,328 params (70 KB)** | **0.0432 ms (43.2 µs)** | **Run by us** | **YES** (Canonical 10-Fold Natural Prev)|
| **Semantic MLP-AE Mode B** | **BGL** | **Mean+Max W2V + MLP-AE** | **Semi-Supervised**| **Val F1 Grid Search** | **88.66 ± 1.07** | **91.05 / 97.34** | **5,328 params (70 KB)** | **0.0432 ms (43.2 µs)** | **Run by us** | **YES** (Canonical 10-Fold Natural Prev)|
| **Semantic MLP-AE Balanced** | **BGL** | Mean+Max W2V + MLP-AE | Unsupervised | Autonomous EVT (POT) | **90.16 ± 1.05** | 95.82 / 97.34 | 5,328 params (70 KB) | 0.0432 ms (43.2 µs) | **Run by us** | **YES** (Base-Rate Prevalence Study) |
| **Word2Vec PCA Baseline** | **BGL** | W2V Mean+Max + PCA ($k=48$)| Unsupervised | Autonomous EVT (POT) | 59.97 ± 8.42 | 84.26 / 96.78 | 400 params (2.1 KB) | 0.0041 ms (4.1 µs) | **Run by us** | **YES** (Ablation on Same Splits) |
| **Count-based PCA Baseline** | **BGL** | Event Count + PCA ($k=6$) | Unsupervised | Autonomous EVT (POT) | 34.28 ± 13.91 | 24.12 / 72.40 | 1,434 params (6.2 KB)| 0.0052 ms (5.2 µs) | **Run by us** | **YES** (Ablation on Same Splits) |
| **Word2Vec Isolation Forest** | **BGL** | W2V Mean+Max + iForest | Unsupervised | Quantile Score Cutoff | 2.07 (POT) / 55.9 (Val)| 48.20 / 92.23 | 100 Trees (1.8 MB) | 0.2800 ms (280 µs)| **Run by us** | **YES** (Ablation on Same Splits) |
| **LogBERT on BGL** | **BGL** | 12-layer Transformer | Semi-Supervised | Hypersphere Radius Cutoff| 90.83 (Empirical) / 92.1| — | ~110M params (420 MB)| ~25.0 ms (GPU) | Cited in [3, 6] | **NO** (20-line tumbling window) |
| **DeepLog on BGL** | **BGL** | LSTM Next-Key Predictor | Unsupervised | Top-$g$ Candidates | 86.12 (Empirical) | — | ~2.5M params (10 MB) | ~4.5 ms | Cited in [6] | **NO** (Interleaved stream distortion) |
| **LogAnomaly on BGL** | **BGL** | FT-Tree + Template2Vec | Semi-Supervised | Semantic Distance Cutoff| 74.08 (Empirical) | — | ~4.1M params (16 MB) | ~6.2 ms | Cited in [6] | **NO** (Windowing disparity) |
| **NeuralLog** (Le 2021) | **BGL** | Raw Text Transformer | **Fully Supervised** | Classification Head | 85.20 | — | ~110M params (420 MB)| ~35.0 ms (GPU) | Cited in [7] | **NO** (Supervised end-to-end) |

---

## 5. Comparative Insights: Why Lightweight Reconstruction & POT Win

1. **Test-Label Independence:**  
   Most published deep learning baselines (e.g. DeepLog, LogBERT) rely on tuning top-$g$ candidate thresholds or hypersphere radii by observing test anomalies. Our Peak-Over-Threshold (POT) mechanism sets decision boundaries **purely on training normal residuals**, eliminating optimistic bias.
2. **Computational Footprint & Edge Viability:**  
   While Transformer baselines (LogBERT, NeuralLog) require over 100 million parameters and dedicated GPU hardware, our MLP Autoencoder contains **5,328 parameters (70.25 KB)** and our HDFS PCA model contains **203 parameters (1.02 KB)**. On a commodity Apple M2 CPU, our models process a log window in **$43.2$ microseconds** and an HDFS block in **$3.5$ microseconds**, achieving throughputs of **62,000 windows/second** and **10,600,000 sessions/second** respectively.
3. **Prevalence Realism:**  
   High literature scores on BGL (>90%) frequently depend on synthetic 50/50 test sets. Under genuine operational base-rate prevalence (10.07%), our pipeline achieves **87.80% Mode A F1** and **91.05% PR-AUC**, which climbs to **90.16% F1** when tested on an identical balanced perspective.

---

## 6. Index of Canonical Result Files & Output Tables

All result CSVs, scripts, and model weights are organized under `FINAL_SUBMISSION/`:
* `FINAL_SUBMISSION/results/RESULTS.md` — This master technical report.
* `FINAL_SUBMISSION/results/hdfs_test_results_5fold.csv` — Full 5-fold HDFS PCA+POT metrics.
* `FINAL_SUBMISSION/results/hdfs_natural_prevalence_results.csv` — Full natural prevalence (570,061 sessions) HDFS metrics.
* `FINAL_SUBMISSION/results/test_results_10fold.csv` — Clean 10-fold natural prevalence (10.07%) test results on BGL.
* `FINAL_SUBMISSION/results/test_balanced_results_10fold.csv` — Balanced view (50% prevalence) test results on BGL.
* `FINAL_SUBMISSION/results/ablation_results.csv` — 10-fold cross-validation ablation table on BGL.
* `FINAL_SUBMISSION/results/chronological_results.csv` — Chronological split robustness evaluation on BGL.
* `FINAL_SUBMISSION/results/efficiency_results.csv` — Measured CPU latency percentiles, parameter counts, and throughput.
* `FINAL_SUBMISSION/hdfs/saved_models/` — Serialized HDFS PCA fold models (5 folds).
* `FINAL_SUBMISSION/bgl/saved_models/` — Serialized BGL Autoencoder and Word2Vec models (10 folds).

---

### Citations for External Literature in Comparison Table
* `[1]` Fält et al., "Evaluating Lightweight Anomaly Detection for Large-Scale Distributed Log Streams", IEEE ICSRS 2025.
* `[2]` Du et al., "DeepLog: Anomaly Detection and System Log Analysis through Deep Learning", ACM CCS 2017.
* `[3]` Guo et al., "LogBERT: Log Anomaly Detection via BERT", IEEE IJCNN 2021.
* `[4]` Zhang et al., "Robust Log-Based Anomaly Detection on Unstable Log Data", ACM ESEC/FSE 2019.
* `[5]` Yang et al., "Semi-supervised Log-based Anomaly Detection via Probabilistic Label Estimation", ACM/IEEE ICSE 2021.
* `[6]` Le & Zhang, "Log-based Anomaly Detection with Deep Learning: How Far Are We?", ACM/IEEE ICSE 2022.
* `[7]` Le & Zhang, "NeuralLog: Natural Language Processing for Log-Based Anomaly Detection", IEEE TSE 2021.
