# FINAL SUBMISSION ARTIFACT & REPRODUCIBILITY PACKAGE

This package contains the self-contained, canonical code, pre-trained model weights, results CSVs, and pinned dependencies to reproduce every experimental table and claim in the research paper:

> **"Bridging Linear Baselines and Lightweight Deep Representations for Scalable Log Anomaly Detection"**

---

## 1. Directory Structure

```text
FINAL_SUBMISSION/
├── README.md                          # This document
├── requirements.txt                   # Exact pinned dependencies
├── hdfs/                              # Canonical HDFS scripts & models
│   ├── run_hdfs_pot.py                # Autonomous PCA + POT pipeline (95.95% F1)
│   ├── pca_rpca.py                    # PCAModel class definition
│   ├── thresholding.py                # Peaks-Over-Threshold (EVT) thresholding
│   ├── run_baseline.py                # Base paper replication (94.60% F1)
│   ├── path_b_fitsize.py              # RPCA rank analysis & ceiling (87.38% F1)
│   └── saved_models/                  # Pickled PCA fold models (5 folds)
├── bgl/                               # Canonical BGL scripts & models
│   ├── run_final_evaluation.py        # 10-Fold leak-free BGL benchmark
│   ├── data_utils.py                  # Stratified 10-fold split & data loader
│   ├── feature_utils.py               # Deterministic Word2Vec & Mean+Max pooling
│   ├── deep_learning.py               # MLP Autoencoder model (5,328 parameters)
│   ├── thresholding.py                # Unsupervised Mode A & Semi-supervised Mode B
│   ├── run_ablations.py               # 6 architectural & representational ablations
│   ├── run_chronological.py           # 80/20 temporal split evaluation
│   ├── benchmark_efficiency.py        # Latency, footprint, and throughput benchmark
│   └── saved_models/                  # 10 Autoencoder & 10 Word2Vec models
└── results/                           # Ground-truth output CSVs and reports
    ├── RESULTS.md                     # Comprehensive leak-free technical report
    ├── test_results_10fold.csv        # Per-fold natural prevalence test metrics
    ├── test_balanced_results_10fold.csv # Per-fold 50% balanced view metrics
    ├── development_results.csv        # Validation model selection records
    ├── ablation_results.csv           # 10-fold cross-validation ablation table
    ├── chronological_results.csv      # Chronological split evaluation record
    └── efficiency_results.csv         # Measured hardware benchmarks on Apple M2
```

---

## 2. Environment Setup

1. Create a clean Python 3.13 virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
2. Install exact pinned dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 3. Step-by-Step Table Reproduction Commands

All commands assume the repository data files exist in `data/`:
* `data/HDFS_v1/preprocessed/features.npz`
* `data/BGL/preprocessed/bgl_sequences.pkl`
* `data/BGL/preprocessed/BGL.log_structured.csv`

### Table 1: HDFS Anomaly Detection Benchmark

1. **Replicate Paper's PCA Baseline (94.60% F1, 95th Percentile Oracle):**
   ```bash
   python hdfs/run_baseline.py
   ```
   *Expected Output:* Precision: 92.34% ± 2.24%, Recall: 96.96% ± 2.35%, F1: 94.60% ± 2.30%.

2. **Evaluate Our PCA + POT Model (95.95% F1, Label-Free):**
   ```bash
   cd hdfs && python run_hdfs_pot.py
   ```
   *Expected Output:* Precision: 93.61%, Recall: 98.46%, F1: 95.95% ± 1.47%.

3. **Verify RPCA Ceiling & Reproducibility Gap (87.38% F1):**
   ```bash
   python hdfs/path_b_fitsize.py
   ```
   *Expected Output:* Peak F1: 87.38% (at fit size 10,000, rank(L) = 4).

---

### Table 2: Leak-Free BGL 10-Fold Benchmark

Run the full leak-free 10-fold cross-validation on BGL:
```bash
python bgl/run_final_evaluation.py
```
*Expected Outputs:*
* **Mode A (Headline — Unsupervised POT):** Precision: 93.76% ± 2.54%, Recall: 82.61% ± 1.96%, F1: 87.80% ± 1.31%.
* **Mode B (Semi-Supervised):** Precision: 93.06% ± 2.59%, Recall: 84.71% ± 1.19%, F1: 88.66% ± 1.07%.
* **Threshold-Free:** PR-AUC: 91.05% ± 0.32%, ROC-AUC: 97.34% ± 0.15%.
* **Balanced View (~50% Anomaly):** Mode A F1: 90.16%, Mode B F1: 91.36%.

---

### Table 3: BGL Architectural & Representational Ablations

Run all 6 cross-validation ablations across 10 folds:
```bash
python bgl/run_ablations.py
```
*Outputs saved to:* `results/ablation_results.csv`.
* Key comparisons:
  * Count-based PCA: 34.28% F1 (Mode A)
  * Word2Vec PCA: 63.84% F1 (Mode A)
  * Word2Vec Isolation Forest: 63.84% F1 (Mode A)
  * Clean Final Autoencoder: 87.80% F1 (Mode A)

---

### Table 4: Temporal Generalization (Chronological Split)

Run the chronological 80/20 train/test evaluation on BGL:
```bash
python bgl/run_chronological.py
```
*Outputs saved to:* `results/chronological_results.csv`.
* *Expected Output:* Test Mode A F1: 87.05% (Precision: 88.75%, Recall: 85.42%).

---

### Table 5: Inference Latency and Hardware Footprint

Run the local latency and throughput benchmarking script:
```bash
python bgl/benchmark_efficiency.py
```
*Outputs saved to:* `results/efficiency_results.csv`.
* *Parameters:* 5,328.
* *Model Size:* 70.25 KB.
* *Single-Sample Latency:* ~0.043 ms (43.2 µs).
* *Batch (256) Latency:* ~0.096 ms (0.37 µs/window).
* *Throughput:* ~62,000 windows/second.

