# FINAL SUBMISSION ARTIFACT & REPRODUCIBILITY PACKAGE

This package contains the self-contained, canonical code, pre-trained model weights, results CSVs, and pinned dependencies to reproduce every experimental table and claim in the research paper:

> **"Lightweight Reconstruction Modeling and Autonomous Extreme Value Thresholding for Syslog Anomaly Detection"**  
> *School of Computer Science and Engineering, VIT-AP University (Academic Year 2026–2027)*

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
│   ├── run_final_evaluation.py        # 10-Fold leak-free BGL benchmark (supports --load-saved)
│   ├── data_utils.py                  # Stratified 10-fold split & data loader
│   ├── feature_utils.py               # Deterministic Word2Vec & Mean+Max pooling
│   ├── deep_learning.py               # MLP Autoencoder model (5,328 parameters)
│   ├── thresholding.py                # Unsupervised Mode A & Semi-supervised Mode B
│   ├── run_ablations.py               # 6 architectural & representational ablations
│   ├── run_chronological.py           # 80/20 temporal split evaluation
│   ├── benchmark_efficiency.py        # Latency, footprint, and throughput benchmark
│   └── saved_models/                  # 10 Autoencoder & 10 Word2Vec pre-trained models
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

All scripts require Python 3.10+ (tested on Python 3.13 macOS arm64 / Linux x86_64).

1. **Activate or create your virtual environment:**
   ```bash
   # From the repository root:
   source venv/bin/activate
   # Or create a fresh one:
   python3 -m venv venv && source venv/bin/activate
   ```

2. **Install exact pinned dependencies:**
   ```bash
   pip install -r FINAL_SUBMISSION/requirements.txt
   ```

3. **Data Requirements:**  
   The preprocessed feature files are read from `data/` or the local directory:
   * `data/HDFS_v1/preprocessed/features.npz` (or `hdfs_improvement/features.npz`)
   * `data/BGL/preprocessed/bgl_sequences.pkl`
   * `data/BGL/preprocessed/BGL.log_structured.csv`

---

## 3. Quick Table-to-Script Mapping

| Paper Table | Description | Execution Script | Fast Mode (`--load-saved`) | Scratch Training | Output File |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **Table 1** | HDFS PCA + POT Benchmark | `FINAL_SUBMISSION/hdfs/run_hdfs_pot.py` | **~1 sec** | ~2 sec | Terminal Console |
| **Table 1** | HDFS Baseline Replication | `FINAL_SUBMISSION/hdfs/run_baseline.py` | **~2 sec** | ~2 sec | Terminal Console |
| **Table 1** | HDFS RPCA Ceiling Proof | `FINAL_SUBMISSION/hdfs/path_b_fitsize.py` | **~4 sec** | ~4 sec | Terminal Console |
| **Table 2** | BGL 10-Fold Leak-Free CV | `FINAL_SUBMISSION/bgl/run_final_evaluation.py` | **~15 sec** | ~6 min | `results/test_results_10fold.csv` |
| **Table 3** | BGL Baseline Ablations | `FINAL_SUBMISSION/bgl/run_ablations.py` | — | ~3 min | `results/ablation_results.csv` |
| **Table 4** | BGL Chronological Split | `FINAL_SUBMISSION/bgl/run_chronological.py` | — | ~40 sec | `results/chronological_results.csv` |
| **Table 5** | Latency & Hardware Profile | `FINAL_SUBMISSION/bgl/benchmark_efficiency.py` | **~2 sec** | ~2 sec | `results/efficiency_results.csv` |

---

## 4. Step-by-Step Reproduction Guide & Output Verification

### Table 1: HDFS Anomaly Detection Benchmark

#### A. Evaluate Our PCA + Autonomous POT Model (Headline: 95.95% F1)
Runs label-free Peaks-Over-Threshold decision boundaries using the saved PCA models.
```bash
python FINAL_SUBMISSION/hdfs/run_hdfs_pot.py
```
* **Runtime:** ~1 second.
* **Terminal Output:**
  ```text
  ================================================================================
  HDFS PCA [POT Threshold Improvement] - Standalone Run
  ================================================================================
  Loading data from data/HDFS_v1/preprocessed/features.npz...
  Fold 1: Loading saved model from FINAL_SUBMISSION/hdfs/saved_models/hdfs_pot_pca_fold_1.pkl...
  Fold 1: P: 93.20%, R: 100.00%, F1: 96.48% (Threshold: 0.0023)
  Fold 2: Loading saved model from FINAL_SUBMISSION/hdfs/saved_models/hdfs_pot_pca_fold_2.pkl...
  Fold 2: P: 93.37%, R: 100.00%, F1: 96.57% (Threshold: 0.0018)
  Fold 3: Loading saved model from FINAL_SUBMISSION/hdfs/saved_models/hdfs_pot_pca_fold_3.pkl...
  Fold 3: P: 94.07%, R: 100.00%, F1: 96.95% (Threshold: 0.0015)
  Fold 4: Loading saved model from FINAL_SUBMISSION/hdfs/saved_models/hdfs_pot_pca_fold_4.pkl...
  Fold 4: P: 93.63%, R: 100.00%, F1: 96.71% (Threshold: 0.0031)
  Fold 5: Loading saved model from FINAL_SUBMISSION/hdfs/saved_models/hdfs_pot_pca_fold_5.pkl...
  Fold 5: P: 93.80%, R:  92.30%, F1: 93.04% (Threshold: 0.0008)

  FINAL HDFS POT RESULT:
  Precision: 93.61%
  Recall:    98.46%
  F1 Score:  95.95%
  ```

#### B. Replicate Published PCA Baseline (94.60% F1, 95th Percentile Oracle)
Replicates the published PCA baseline under the exact 5-fold sampling protocol:
```bash
python FINAL_SUBMISSION/hdfs/run_baseline.py
```
* **Runtime:** ~2 seconds.
* **Terminal Output:**
  ```text
  FINAL 5-FOLD CV RESULT: HDFS PCA [Paper-Exact Protocol]
  Precision: 92.34% ± 2.24%
  Recall:    96.96% ± 2.35%
  F1 Score:  94.60% ± 2.30%
  ```

#### C. Verify RPCA Ceiling & Reproducibility Gap (87.38% F1)
Demonstrates why published Robust PCA claims of ~90.55% F1 are mathematically unreachable:
```bash
python FINAL_SUBMISSION/hdfs/path_b_fitsize.py
```
* **Runtime:** ~4 seconds.
* **Terminal Output:**
  ```text
  Test Set: 21000 (20k normal, 1k anomaly)
    Fit Size   Rank     Prec      Rec       F1
  --------------------------------------------------
        5000      4    86.88    83.40    85.10
       10000      4    91.75    83.40    87.38
       15000      4    87.61    83.40    85.45
       20000      4    81.21    83.40    82.29
       25000      4    87.04    83.30    85.13
  ```

---

### Table 2: Leak-Free BGL 10-Fold Benchmark

#### Fast Verification: Instant Evaluation of Pre-Trained Models (~15s)
Evaluates all 10 saved Word2Vec and Autoencoder models across test sets without retraining:
```bash
python FINAL_SUBMISSION/bgl/run_final_evaluation.py --load-saved
```
*(Tip: To quickly test Fold 1 only in ~5 seconds, run `python FINAL_SUBMISSION/bgl/run_final_evaluation.py --load-saved --folds 1`)*

#### Full Reproduction: Train All 10 Folds From Scratch (~6 min)
```bash
python FINAL_SUBMISSION/bgl/run_final_evaluation.py
```

* **Terminal Output:**
  ```text
  ========================================
  >>> Running Fold 1 / 10 <<<
  ========================================
  Thresholds -> Mode A: 0.2477 (fallback=False) | Mode B: 0.2028 (Val F1: 89.24%)
  Test Result Fold 1:
    Mode A (Headline): P=90.90%, R=83.72%, F1=87.16% | TP=10,844, FP=1,086, TN=114,643, FN=2,109
    Mode B (Val-Tuned): P=88.54%, R=85.69%, F1=87.10% | TP=11,100, FP=1,436, TN=114,293, FN=1,853
    Threshold-Free:    PR-AUC=91.00%, ROC-AUC=97.23%
  ...
  ================================================================================
  FINAL 10-FOLD TEST RESULTS SUMMARY (NATURAL PREVALENCE)
  ================================================================================
  Mode A (Headline Label-Free POT):
    Precision: 93.76% ± 2.54%
    Recall:    82.61% ± 1.96%
    F1-Score:  87.80% ± 1.31%

  Mode B (Validation-Tuned Threshold):
    Precision: 93.06% ± 2.59%
    Recall:    84.71% ± 1.19%
    F1-Score:  88.66% ± 1.07%

  Threshold-Free:
    PR-AUC:    91.05% ± 0.32%
    ROC-AUC:   97.34% ± 0.15%

  BALANCED VIEW (~50% Anomalies):
  Mode A: P=91.20%, R=89.15%, F1=90.16%
  Mode B: P=90.84%, R=91.90%, F1=91.36%
  ```
* **Generated Results CSVs:**
  * `FINAL_SUBMISSION/results/test_results_10fold.csv` (Per-fold metrics under natural 10.07% prevalence)
  * `FINAL_SUBMISSION/results/test_balanced_results_10fold.csv` (Per-fold metrics under 50% balanced view)

---

### Table 3: BGL Architectural & Representational Ablations

Compares Count-based PCA, Word2Vec-PCA, Isolation Forest, and our final Autoencoder across all 10 folds:
```bash
python FINAL_SUBMISSION/bgl/run_ablations.py
```
* **Runtime:** ~3 minutes.
* **Terminal Output Summary:**
  ```text
  ================================================================================
  ABLATION STUDY SUMMARY (Averaged across 10 folds on Test Set)
  ================================================================================
  | PCA (Count Vector)     | Mode A F1:  34.28% (P: 21.04%, R: 92.45%) | PR-AUC: 24.12% | ROC-AUC: 72.40% |
  | PCA (Word2Vec)         | Mode A F1:  63.84% (P: 49.32%, R: 90.85%) | PR-AUC: 58.12% | ROC-AUC: 88.50% |
  | Isolation Forest       | Mode A F1:  63.84% (P: 49.32%, R: 90.85%) | PR-AUC: 58.12% | ROC-AUC: 88.50% |
  | All-Normal Baseline    | Mode A F1:   0.00% (P:  0.00%, R:  0.00%) | PR-AUC: 10.07% | ROC-AUC: 50.00% |
  | Random Guessing        | Mode A F1:  18.29% (P: 10.07%, R: 99.90%) | PR-AUC: 10.07% | ROC-AUC: 50.00% |
  | Final Clean Autoencoder| Mode A F1:  87.80% (P: 93.76%, R: 82.61%) | PR-AUC: 91.05% | ROC-AUC: 97.34% |
  ================================================================================
  ```
* **Generated Results CSV:** `FINAL_SUBMISSION/results/ablation_results.csv`

---

### Table 4: Temporal Generalization (Chronological Split)

Trains strictly on the first 70% of chronological normal logs and tests on the latest 20% future logs (including drift and 68.9% OOV words):
```bash
python FINAL_SUBMISSION/bgl/run_chronological.py
```
* **Runtime:** ~40 seconds.
* **Terminal Output:**
  ```text
  ================================================================================
  CHRONOLOGICAL TEST RESULTS
  ================================================================================
  Mode A (Label-Free POT, theta=0.5937, fallback=True):
    TP: 10 | FP: 3 | TN: 231,455 | FN: 64,832
    Precision: 76.92% | Recall: 0.02% | F1-Score: 0.03%

  Mode B (Validation-Tuned, theta=0.1874):
    TP: 62,184 | FP: 147,616 | TN: 83,842 | FN: 2,658
    Precision: 29.64% | Recall: 95.90% | F1-Score: 45.28%

  Threshold-Free:
    PR-AUC:  30.32%
    ROC-AUC: 67.50%
  ================================================================================
  ```
* **Generated Results CSV:** `FINAL_SUBMISSION/results/chronological_results.csv`

---

### Table 5: Inference Latency and Hardware Footprint

Measures single-window latency, batch inference, parameter footprint, and throughput directly on your machine:
```bash
python FINAL_SUBMISSION/bgl/benchmark_efficiency.py
```
* **Runtime:** ~2 seconds.
* **Terminal Output:**
  ```text
  ================================================================================
  EFFICIENCY & LATENCY BENCHMARK
  Hardware: Apple M2 (arm64) | RAM: 8.0 GB
  ================================================================================
  | Model Parameters:                5328
  | Serialized Model Size:          70.25 KB
  | Single Window Latency:       0.0411 ms (p99: 0.0544 ms)
  | Batch (256) Latency:         0.0906 ms (p99: 0.1157 ms)
  | Per-window in Batch 256:       0.35 µs/window
  | End-to-End Throughput:        69,153.3 windows/second
  ================================================================================
  ```
* **Generated Results CSV:** `FINAL_SUBMISSION/results/efficiency_results.csv`

---

## 5. How to View Saved Results

All generated tables are stored as standard CSV files in `FINAL_SUBMISSION/results/`. You can view them directly using any CSV tool, Python, or shell:

```bash
# Print headline 10-fold test metrics:
python3 -c "
import pandas as pd
df = pd.read_csv('FINAL_SUBMISSION/results/test_results_10fold.csv')
print(df[['fold', 'mode_a_p', 'mode_a_r', 'mode_a_f1', 'mode_b_f1', 'pr_auc', 'roc_auc']].to_string(index=False))
"

# Print efficiency summary:
python3 -c "
import pandas as pd
df = pd.read_csv('FINAL_SUBMISSION/results/efficiency_results.csv')
print(df.to_string(index=False))
"
```

For the comprehensive technical audit answering all 23 reviewer queries (leak-free proofs, score distributions, confusion matrices, and external model comparisons), open **[PROJECT_MAP.md](../PROJECT_MAP.md)**.
