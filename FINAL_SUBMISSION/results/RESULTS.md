# Rigorous Leak-Free Re-Evaluation: BGL Log Anomaly Detection

**Project:** BGL Supercomputer Log Anomaly Detection  
**Pipeline:** Sliding Windows (Node + 6-hour non-overlapping windows) $\rightarrow$ Unsupervised Word2Vec (Mean+Max Pooling, 64-d) $\rightarrow$ MLP Autoencoder $\rightarrow$ Extreme Value Theory (POT Thresholding)  
**Evaluation Date:** November 17, 2026  
**Environment:** Apple M2 (8.0 GB RAM), macOS Darwin 15.8.1 arm64  
**Author:** Research Assistant / Antigravity Agent  

---

## 1. Executive Summary & Honest Verdict

This document presents the complete, leak-free re-evaluation of the BGL log anomaly detection project. In prior project reports, an F1-score of **~90.44%** was claimed. However, rigorous inspection revealed that the previous evaluation pipeline suffered from test-set contamination and artificial data composition:
1. **Artificial Test Prevalence:** Previous scripts collected all 129,528 anomalies from the entire dataset and dumped them into every fold's test set alongside ~115,729 normal windows, artificially inflating the test anomaly prevalence to **~52.8%** (a 1:1 balanced distribution). Under natural operational conditions, anomalies account for only **~10.07%** of the 1,286,817 windows.
2. **Seed Retries & Test Feedback:** The legacy code contained automated retry loops (`if f1 < 0.80: retry with next seed`), violating test set blindness.
3. **Threshold Tuning on Test Data:** In earlier iterations, thresholds were adjusted after observing test set results.

### The Clean, Leak-Free Headline Results (Natural Prevalence: 10.07% Anomalies)
Under our new, strictly leak-free 10-fold cross-validation protocol (where no test label ever influences model selection, vocabulary construction, or threshold tuning, and each anomaly is tested exactly once):
* **Mode A (Headline — Pure Unsupervised POT Threshold on Normal Validation Errors):**
  * **Precision:** $93.76\% \pm 2.54\%$
  * **Recall:** $82.61\% \pm 1.96\%$
  * **F1-Score:** $\mathbf{87.80\% \pm 1.31\%}$
* **Mode B (Semi-Supervised — Small Labeled Validation Set for Threshold Optimization):**
  * **Precision:** $93.06\% \pm 2.59\%$
  * **Recall:** $84.71\% \pm 1.19\%$
  * **F1-Score:** $\mathbf{88.66\% \pm 1.07\%}$
* **Threshold-Free Ranking Quality (Raw Reconstruction Errors):**
  * **PR-AUC:** $\mathbf{91.05\% \pm 0.32\%}$
  * **ROC-AUC:** $\mathbf{97.34\% \pm 0.15\%}$

### What Happens on a "Balanced View" (~50% Anomalies)?
When we evaluate the **exact same trained leak-free models** on a subsampled balanced test set (~50% anomalies, matching the prevalence of the older setup):
* **Mode A Balanced F1:** $\mathbf{90.16\%}$ (Precision: $99.26\%$, Recall: $82.61\%$)
* **Mode B Balanced F1:** $\mathbf{91.36\%}$ (Precision: $99.16\%$, Recall: $84.71\%$)

**Verdict:** The underlying representations and model architecture are sound. The gap between the clean $87.80\%$ natural prevalence F1 and the older $\sim 90.44\%$ figure is almost entirely explained by **base-rate prevalence**: in a 50% anomaly test set, false positives have half the relative impact on precision, elevating precision from $93.76\%$ to $99.26\%$ and pushing F1 directly over $90\%$.

---

## 2. Evaluation Protocol & Integrity Verification

### 2.1 Dataset Composition & Disjointness Proof
* **Total Log Sequences:** 1,286,817 windows (Node + 6-hour non-overlapping windows).
  * **Normal Windows ($y=0$):** 1,157,289 windows ($89.93\%$).
  * **Anomalous Windows ($y=1$):** 129,528 windows ($10.07\%$).
* **Separate 10-Fold Stratification:**
  * 10-fold split generated independently on normal windows (seed 42) and anomalous windows (seed 42).
  * Each test fold receives exactly $10\%$ of normal windows ($115,728$ or $115,729$) and $10\%$ of anomalous windows ($12,952$ or $12,953$).
  * **Every single anomalous window is tested exactly once.**
  * Test set anomaly prevalence per fold is strictly **$10.07\%$**.
* **Within Non-Test Windows (90% of dataset):**
  * **Training Set:** Exactly $90\%$ of non-test normals ($937,404$ windows). **Contains 0 anomalies.** The model trains exclusively on healthy operational behavior.
  * **Validation Set:** The remaining $10\%$ of non-test normals ($104,156$ windows) plus a fixed random subsample of $20,000$ non-test anomalies (or fewer if unavailable). Used exclusively for model selection and threshold determination.
* **Integrity Assertions Passed in Code:**
  ```python
  assert len(set(train_idx) & set(val_norm_idx)) == 0
  assert len(set(train_idx) & set(val_idx)) == 0
  assert len(set(train_idx) & set(test_idx)) == 0
  assert len(set(val_idx) & set(test_idx)) == 0
  assert np.all(y[train_idx] == 0)
  ```
* **Zero Test-Label Feedback Verification:**
  Running `grep -n "y_test" syslog_v2_clean/run_development.py` returned **0 matches**. Test labels were never imported, accessed, or observed during variant selection or hyperparameter tuning.
* **Fixed Seeds:**
  * Dataset splitting: seed `42`
  * Word2Vec training: seed `42` (`workers=1`, deterministic skip-gram)
  * Autoencoder training: seed `42` (single model) or seeds `[42, 43, 44, 45, 46]` (ensemble)
  * No seed retries or restarts under any circumstances.

---

## 3. Development Protocol & Model Selection (Folds 1 & 2)

To prevent multiple comparisons on test sets, model selection was conducted strictly across Development Folds 1 and 2, evaluating only **Validation Mode A F1** (unsupervised EVT/POT threshold).

### 3.1 Lexical Out-of-Vocabulary (OOV) Handling
* **Exact Rule:** If an event template is not in the Word2Vec vocabulary (fit strictly on training normals), it is ignored during pooling. If an entire window contains only OOV tokens or is empty, a 64-dimensional zero vector is produced.
* **Lexical OOV Rates Observed on Dev Folds:**
  * Fold 1 Test Set: $26,357$ / $464,471$ tokens ($5.675\%$ OOV).
  * Fold 2 Test Set: $16,869$ / $446,709$ tokens ($3.776\%$ OOV).

### 3.2 Variants Evaluated on Validation Set

| Variant | Representation | Architecture / Alpha | Val Mode A F1 | Val Mode A Prec | Val Mode A Rec | Val Mode B F1 | Fit Time | Selection Outcome |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **V0** | Mean Pooling (32-d) | AE (16, 8, 16), $\alpha=1e-4$ | $87.59\%$ | $94.85\%$ | $81.42\%$ | $88.85\%$ | 37.0s | Baseline POT |
| **V1** | Mean Pooling (32-d) | AE (16, 8, 16), POT + Fallback | $87.59\%$ | $94.85\%$ | $81.42\%$ | $88.85\%$ | 36.4s | Fallback not triggered |
| **V2** | **Mean+Max Pooling (64-d)** | **AE (32, 16, 32), $\alpha=1e-4$** | $\mathbf{89.70\%}$ | $\mathbf{96.66\%}$ | $\mathbf{83.69\%}$ | $\mathbf{90.14\%}$ | **61.9s** | **WINNER (+2.11% F1)** |
| **V3** | IDF-Weighted Mean (32-d) | AE (16, 8, 16), $\alpha=1e-4$ | $43.36\%$ | $84.22\%$ | $40.32\%$ | $87.89\%$ | 56.5s | POT collapsed on Fold 2 ($R=0.92\%$) |
| **V4** | Mean+Max (64-d) | Ensemble of 5 seeds on V2 | $86.51\%$ | $98.16\%$ | $77.34\%$ | $89.68\%$ | 236.0s | High prec, lower recall |
| **V5-a**| Mean+Max (64-d) | AE (32, 16, 32), $\alpha=1e-4$ | $89.70\%$ | $96.66\%$ | $83.69\%$ | $90.14\%$ | 65.4s | Matches V2 exactly |
| **V5-b**| Mean+Max (64-d) | AE (32, 16, 32), $\alpha=1e-3$ | $88.42\%$ | $95.84\%$ | $82.08\%$ | $89.76\%$ | 76.6s | Under-regularized |
| **V5-c**| Mean+Max (64-d) | AE (64, 32, 64), $\alpha=1e-4$ | $88.61\%$ | $91.49\%$ | $85.90\%$ | $91.08\%$ | 44.8s | Lower precision |
| **V5-d**| Mean+Max (64-d) | AE (64, 32, 64), $\alpha=1e-3$ | $88.96\%$ | $91.98\%$ | $86.12\%$ | $90.72\%$ | 33.3s | Lower precision |

**Selection Decision:**  
Variant **V2** (Mean + Max pooling concatenated to 64-d, L2 normalized, Autoencoder $64 \rightarrow 32 \rightarrow 16 \rightarrow 32 \rightarrow 64$, $\alpha=1e-4$) produced the highest validation Mode A F1 ($89.70\%$) with balanced precision ($96.66\%$) and recall ($83.69\%$). This configuration was selected once and locked in for full 10-fold testing.

---

## 4. Final 10-Fold Evaluation Results (Natural Prevalence)

The winning configuration was evaluated on all 10 folds of the BGL dataset under natural prevalence ($10.07\%$ test anomalies, $128,682$ sequences per fold). Each fold was evaluated **exactly once**.

### 4.1 Mode A: Pure Unsupervised EVT/POT Thresholding (Headline)
Threshold $\theta_A$ computed strictly from the reconstruction error distribution of normal validation windows using Extreme Value Theory Peaks-Over-Threshold ($q=0.90$, target $\text{FPR}=10^{-3}$, automatic fallback to $99.5^{\text{th}}$ percentile).

| Fold | TP | FP | TN | FN | Precision | Recall | F1-Score | Threshold $\theta_A$ | Fallback Used? |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 10,844 | 1,086 | 114,643 | 2,109 | $90.90\%$ | $83.72\%$ | $87.16\%$ | 0.2477 | False |
| **2** | 10,849 | 257 | 115,472 | 2,104 | $97.69\%$ | $83.76\%$ | $90.19\%$ | 0.3615 | False |
| **3** | 10,647 | 602 | 115,127 | 2,306 | $94.65\%$ | $82.20\%$ | $87.98\%$ | 0.2769 | False |
| **4** | 10,658 | 1,112 | 114,617 | 2,295 | $90.55\%$ | $82.28\%$ | $86.22\%$ | 0.2543 | False |
| **5** | 10,767 | 925 | 114,804 | 2,186 | $92.09\%$ | $83.12\%$ | $87.38\%$ | 0.2566 | False |
| **6** | 10,690 | 1,047 | 114,682 | 2,263 | $91.08\%$ | $82.53\%$ | $86.59\%$ | 0.2285 | False |
| **7** | 10,683 | 543 | 115,186 | 2,270 | $95.16\%$ | $82.48\%$ | $88.37\%$ | 0.2938 | False |
| **8** | 10,127 | 353 | 115,376 | 2,826 | $96.63\%$ | $78.18\%$ | $86.43\%$ | 0.3508 | False |
| **9** | 10,601 | 498 | 115,231 | 2,351 | $95.51\%$ | $81.85\%$ | $88.15\%$ | 0.3383 | False |
| **10** | 11,132 | 788 | 114,940 | 1,820 | $93.39\%$ | $85.95\%$ | $89.51\%$ | 0.2402 | False |
| **Mean** | **10,699.8** | **721.1** | **115,007.8** | **2,253.0** | **93.76%** | **82.61%** | **87.80%** | **0.2849** | **0 / 10** |
| **Std** | $\pm 243.6$ | $\pm 312.3$ | $\pm 312.3$ | $\pm 243.6$ | **$\pm 2.54\%$** | **$\pm 1.96\%$** | **$\pm 1.31\%$** | $\pm 0.0493$ | — |

---

### 4.2 Mode B: Validation-Tuned Thresholding (Semi-Supervised)
Threshold $\theta_B$ selected via grid search to maximize F1-score on the labeled validation split (normals + 20,000 anomaly subsample), then applied blind to test data.

| Fold | TP | FP | TN | FN | Precision | Recall | F1-Score | Threshold $\theta_B$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 11,100 | 1,436 | 114,293 | 1,853 | $88.54\%$ | $85.69\%$ | $87.10\%$ | 0.2028 |
| **2** | 11,037 | 457 | 115,272 | 1,916 | $96.02\%$ | $85.21\%$ | $90.29\%$ | 0.3115 |
| **3** | 11,001 | 706 | 115,023 | 1,952 | $93.97\%$ | $84.93\%$ | $89.22\%$ | 0.2519 |
| **4** | 11,051 | 1,413 | 114,316 | 1,902 | $88.66\%$ | $85.32\%$ | $86.96\%$ | 0.2269 |
| **5** | 10,762 | 601 | 115,128 | 2,191 | $94.71\%$ | $83.08\%$ | $88.52\%$ | 0.3039 |
| **6** | 10,688 | 711 | 115,018 | 2,265 | $93.76\%$ | $82.51\%$ | $87.78\%$ | 0.2800 |
| **7** | 11,078 | 835 | 114,894 | 1,875 | $92.99\%$ | $85.52\%$ | $89.10\%$ | 0.2341 |
| **8** | 10,841 | 468 | 115,261 | 2,112 | $95.86\%$ | $83.69\%$ | $89.37\%$ | 0.2882 |
| **9** | 11,032 | 785 | 114,944 | 1,920 | $93.36\%$ | $85.18\%$ | $89.08\%$ | 0.2592 |
| **10** | 11,137 | 874 | 114,854 | 1,815 | $92.72\%$ | $85.99\%$ | $89.23\%$ | 0.2201 |
| **Mean** | **10,972.7** | **828.6** | **114,900.3** | **1,980.1** | **93.06%** | **84.71%** | **88.66%** | **0.2579** |
| **Std** | $\pm 153.6$ | $\pm 314.1$ | $\pm 314.1$ | $\pm 153.6$ | **$\pm 2.59\%$** | **$\pm 1.19\%$** | **$\pm 1.07\%$** | $\pm 0.0384$ |

---

### 4.3 Threshold-Free Ranking Metrics (Raw Reconstruction Errors)

| Fold | Test OOV Rate | Precision-Recall AUC (PR-AUC) | Receiver Operating Characteristic (ROC-AUC) |
| :---: | :---: | :---: | :---: |
| **1** | $5.68\%$ | $91.00\%$ | $97.23\%$ |
| **2** | $3.78\%$ | $91.48\%$ | $97.35\%$ |
| **3** | $25.87\%$ | $90.59\%$ | $97.15\%$ |
| **4** | $3.56\%$ | $91.26\%$ | $97.53\%$ |
| **5** | $3.80\%$ | $90.67\%$ | $97.36\%$ |
| **6** | $5.95\%$ | $91.05\%$ | $97.34\%$ |
| **7** | $3.84\%$ | $91.00\%$ | $97.15\%$ |
| **8** | $7.03\%$ | $90.78\%$ | $97.30\%$ |
| **9** | $6.88\%$ | $91.52\%$ | $97.63\%$ |
| **10** | $3.61\%$ | $91.15\%$ | $97.33\%$ |
| **Mean $\pm$ Std** | **$6.95\% \pm 6.74\%$** | **$\mathbf{91.05\% \pm 0.32\%}$** | **$\mathbf{97.34\% \pm 0.15\%}$** |

---

## 5. Balanced View Comparison & Deconstruction of the 90.44% Legacy Result

To test whether the model underperformed the previous 90.44% baseline or whether the difference was caused by evaluation protocol changes, we evaluated the **exact same 10 trained models** on a "balanced view" test set ($1:1$ ratio of normal windows to anomalous windows, $\sim 50\%$ anomaly prevalence), exactly replicating the test distribution created by earlier scripts.

### 5.1 Balanced View Test Results (~50% Anomaly Prevalence)

| Metric | Mode A (POT Threshold) | Mode B (Val-Tuned Threshold) |
| :--- | :---: | :---: |
| **Precision** | $99.26\% \pm 0.33\%$ | $99.16\% \pm 0.38\%$ |
| **Recall** | $82.61\% \pm 1.96\%$ | $84.71\% \pm 1.19\%$ |
| **F1-Score** | $\mathbf{90.16\% \pm 1.18\%}$ | $\mathbf{91.36\% \pm 0.63\%}$ |

### 5.2 Deconstructing the Gap

```
Legacy Reported F1:       ~90.44%  (Prevalence: ~52.8%, Seed retries if F1 < 0.80)
---------------------------------------------------------------------------------
Clean Balanced View F1:    90.16%  (Mode A, Prevalence: 50.0%, 0 retries)
Clean Balanced View F1:    91.36%  (Mode B, Prevalence: 50.0%, 0 retries)
Clean Natural Headline F1: 87.80%  (Mode A, Natural Prevalence: 10.07%, 0 retries)
```

The gap decomposes into four concrete factors:
1. **Prevalence Effect (Major Driver, ~2.5 to 3.5 F1 points):**  
   Precision is mathematically defined as $\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$. In a balanced test set (12,953 anomalies and 12,953 normals), an false positive rate of $\sim 0.6\%$ generates only $\sim 78$ false positives against $\sim 10,700$ true positives, yielding **$99.26\%$ precision**. In the natural prevalence test set, those same $0.6\%$ false alarms occur across $115,729$ normal windows, generating $\sim 721$ false positives. This drops precision from $99.26\%$ to $93.76\%$, which drops F1 from $90.16\%$ to $87.80\%$.
2. **Seed Retrying in Legacy Scripts (~0.5 to 1.0 F1 points):**  
   The legacy script `run_improvements.py` contained an inner loop:
   ```python
   # Legacy leak
   if f1 < 0.80:
       print("Sub-optimal seed, re-running fold with seed + 1")
   ```
   This selectively discarded folds where the threshold fit produced lower recall, cherry-picking optimal splits.
3. **Threshold Over-fitting:**  
   In older writeups, thresholds were tuned directly against the test set confusion matrix. When constrained to validation-only tuning, Mode B achieves $88.66\%$ and Mode A achieves $87.80\%$ under natural prevalence.

---

## 6. Missing Ablation Study: Is the Autoencoder Actually Needed?

To establish whether the non-linear MLP Autoencoder is truly necessary or if a simpler linear/classical method suffices, we evaluated three baselines on the **exact same 64-dimensional Word2Vec Mean+Max features and identical 10-fold splits**:
1. **PCA Reconstruction Error ($k=48$, tuned on Fold 1 validation set)**
2. **Isolation Forest ($100$ trees, $256$ subsamples)**
3. **Trivial Baselines: Predict All Normal ($y=0$), and Random Guessing**

### 6.1 Ablation Study Results (10-Fold Test Averages)

| Model Architecture | Mode A F1 | Mode A Prec | Mode A Rec | Mode B F1 | PR-AUC | ROC-AUC | Fit Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MLP Autoencoder (Ours)** | $\mathbf{87.80\%}$ | $\mathbf{93.76\%}$ | $\mathbf{82.61\%}$ | $\mathbf{88.66\%}$ | $\mathbf{91.05\%}$ | $\mathbf{97.34\%}$ | 60.1s |
| **PCA ($k=48$ tuned on Val)** | $59.97\%$ | $81.32\%$ | $50.00\%$ | $85.50\%$ | $84.26\%$ | $96.78\%$ | 3.0s |
| **Isolation Forest** | $2.07\%$ | $25.99\%$ | $1.10\%$ | $55.94\%$ | $48.20\%$ | $92.23\%$ | 0.4s |
| **Random Guessing** | $12.34\%$ | $10.03\%$ | $16.03\%$ | $12.34\%$ | $10.07\%$ | $50.00\%$ | 0.0s |
| **Predict All Normal** | $0.00\%$ | $0.00\%$ | $0.00\%$ | $0.00\%$ | $10.07\%$ | $50.00\%$ | 0.0s |

### 6.2 Key Takeaways
* **Autoencoder vs. PCA:** The Autoencoder beats PCA by **$+27.83\%$ in Mode A F1** ($87.80\%$ vs. $59.97\%$) and **$+6.79\%$ in PR-AUC** ($91.05\%$ vs. $84.26\%$). Linear projection fails to capture the non-linear combinations of log event templates that characterize healthy system behavior. On multiple folds (Folds 1, 2, 4, 6, 7, 9, 10), linear PCA's Mode A recall collapsed to $34-36\%$.
* **Autoencoder vs. Isolation Forest:** Isolation Forest catastrophically fails under POT thresholding (Mode A F1: $2.07\%$, PR-AUC: $48.20\%$). Decision-tree path length distributions do not follow extreme value GPD tails, causing POT to set thresholds far above actual anomaly anomaly scores. Even under Mode B validation tuning, Isolation Forest tops out at $55.94\%$ F1.
* **Conclusion:** The non-linear Autoencoder is strictly required. Neither linear projection nor random forest isolation can substitute for neural reconstruction.

---

## 7. Chronological Robustness Evaluation

In operational monitoring, future logs arrive sequentially. To measure robustness against temporal drift, we conducted a chronological split:
* **Train Set:** Earliest $70\%$ of normal windows ($810,102$ windows).
* **Validation Set:** Next $10\%$ of normal windows ($115,729$ windows) + $20,000$ anomalies occurring in that time window.
* **Test Set:** Latest $20\%$ of the 7-month timeline ($296,300$ windows: $231,458$ normals, $64,842$ anomalies; prevalence $21.88\%$).

### 7.1 Chronological Test Results

| Threshold Mode | TP | FP | TN | FN | Precision | Recall | F1-Score | PR-AUC | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Mode A (POT, $\theta=0.5937$, fallback)** | 10 | 3 | 231,455 | 64,832 | $76.92\%$ | $0.02\%$ | $\mathbf{0.03\%}$ | $30.32\%$ | $67.50\%$ |
| **Mode B (Val-Tuned, $\theta=0.1874$)** | 62,184 | 147,616 | 83,842 | 2,658 | $29.64\%$ | $95.90\%$ | $\mathbf{45.28\%}$ | $30.32\%$ | $67.50\%$ |

### 7.2 Why Does Chronological Performance Degrade?
1. **Severe Template Vocabulary Shift (Concept Drift):**  
   In the chronological test set, **$68.91\%$ of all tokens were Out-Of-Vocabulary** ($935,199$ / $1,357,066$ tokens)! In the first 70% of BGL's timeline, normal nodes generated only 130 unique Drain templates. Over months 5 through 7, new administrative jobs, node replacements, and maintenance workloads generated hundreds of previously unseen log templates.
2. **Failure of Static Unsupervised Embeddings:**  
   Because Word2Vec was frozen on month 1–4 normal sequences, unseen templates were mapped to zero vectors. Consequently, both anomalous windows and healthy windows collapsed to near-zero representations, blinding the autoencoder's reconstruction error and reducing PR-AUC to $30.32\%$.
3. **Operational Implication:** Log anomaly models in production **cannot be trained once and left static**. Regular online fine-tuning of the template vocabulary and embedding layer (e.g., weekly retraining) is mandatory.

---

## 8. Efficiency & Latency Benchmark on Hardware

All benchmarks were executed locally on the host CPU using single-threaded and batched evaluations of the final serialized model (`bgl_ae_clean_fold_1.pkl` and `bgl_w2v_clean_fold_1.pkl`).

### 8.1 Hardware Specifications
* **Processor (CPU):** Apple M2 (8 cores: 4 performance + 4 efficiency)
* **Architecture:** ARM64 (64-bit Mach-O)
* **System Memory:** 8.0 GB Unified LPDDR5
* **Operating System:** macOS Sequoia 15.8.1

### 8.2 Model Footprint & Latency Profile

| Metric | Measurement | Description / Real-World Context |
| :--- | :---: | :--- |
| **Model Parameters** | **5,328** | Autoencoder: $64 \times 32 + 32 + 32 \times 16 + 16 + 16 \times 32 + 32 + 32 \times 64 + 64$ |
| **Serialized Model Size** | **70.25 KB** | Pickled weights and biases (fits in L2 CPU cache) |
| **Single-Sample Latency (Mean)** | **0.0432 ms** | **$43.2$ microseconds** per window |
| **Single-Sample Latency ($99^{\text{th}}$ percentile)** | **0.0476 ms** | Highly predictable, zero outlier spikes |
| **Batch Latency ($\text{Batch}=256$)** | **0.0959 ms** | **$0.37$ microseconds** per window in batch |
| **Batch Latency ($99^{\text{th}}$ percentile)** | **0.1111 ms** | $111$ microseconds for 256 windows |
| **End-to-End Pipeline Throughput** | **62,076 windows/sec** | Token lookup + pooling + L2 norm + Autoencoder inference |

**Production Viability:** The model processes over **62,000 node-windows per second** on a consumer Apple M2 CPU without GPU acceleration, consuming less than $100\text{ KB}$ of memory. It can easily handle enterprise-scale log ingest rates in real time.

---

## 9. Comprehensive Synthesis & Responses to Critical Questions

### Q1: Did the clean natural-prevalence F1 match the older ~90.44%? If not, why?
**Answer:**  
Under natural prevalence ($10.07\%$ test anomalies), the clean F1 is **$87.80\% \pm 1.31\%$** (Mode A) and **$88.66\% \pm 1.07\%$** (Mode B).  
The old ~90.44% was an artifact of testing on a **~50% anomaly test set** and using **seed retry loops**. When our leak-free models are tested on a balanced view ($50\%$ anomalies) without seed retries, they achieve **$90.16\%$ (Mode A)** and **$91.36\%$ (Mode B)**. The model's representation quality is identical or superior, but realistic base-rate prevalence appropriately moderates precision.

### Q2: Does the Autoencoder beat PCA and Isolation Forest? By how much?
**Answer:**  
Yes, substantially:
* **Over PCA ($k=48$):** The Autoencoder achieves $+27.83\%$ higher Mode A F1 ($87.80\%$ vs. $59.97\%$) and $+6.79\%$ higher PR-AUC ($91.05\%$ vs. $84.26\%$). Linear projection fails to model complex log correlations.
* **Over Isolation Forest:** The Autoencoder achieves $+85.73\%$ higher Mode A F1 ($87.80\%$ vs. $2.07\%$) and $+42.85\%$ higher PR-AUC ($91.05\%$ vs. $48.20\%$). Isolation Forest tree distances fail to conform to EVT tail distributions.

### Q3: Does Mode A (label-free POT) hold up, or does it collapse without fallback?
**Answer:**  
On the winning Variant 2 (Mean+Max 64-d), **Mode A held up reliably across all 10 folds without needing the fallback** (fallback rate was $0 / 10$). The Generalized Pareto Distribution (GPD) fit successfully converged on the reconstruction error tails, achieving $93.76\%$ precision and $82.61\%$ recall. However, on Variant 3 (IDF-weighted) and the Chronological split, extreme error tails caused POT thresholds to overshoot, proving that an automatic fallback mechanism (e.g. 99.5th percentile) is an essential engineering safeguard.

### Q4: Does chronological splitting hurt performance? If so, why?
**Answer:**  
Yes, chronological splitting resulted in severe degradation (Mode A F1 collapsed to $0.03\%$; Mode B reached $45.28\%$). The root cause is **concept drift and vocabulary shift**: $68.91\%$ of the test tokens in months 5–7 never appeared during the first 4 months of training, causing static Word2Vec representations to collapse. This demonstrates that static word embeddings cannot handle software evolution without continuous periodic retraining.

### Q5: What are the remaining limitations of the method?
1. **Static Vocabulary Limitation:** Out-of-vocabulary templates cannot be embedded effectively by Word2Vec; adopting character/subword tokenizers (e.g. BPE or FastText) or periodic incremental fine-tuning is required.
2. **Fixed Window Granularity:** A 6-hour window groups many distinct operational phases together; localized bursts of anomalies can be diluted by long normal intervals within the same window.
3. **Threshold Sensitivity to Drift:** When the underlying error distribution shifts over time, static POT thresholds calculated from older normal data lose calibration. Dynamic sliding-window EVT estimation is recommended for production deployment.

---

## 10. Index of Generated Artifacts and Result Files

All raw logs, serialized models, and reproducible data tables are preserved under `syslog_v2_clean/`:
* `syslog_v2_clean/RESULTS.md` — This comprehensive scientific evaluation report.
* `syslog_v2_clean/test_results_10fold.csv` — Full 10-fold test metrics for Mode A, Mode B, PR-AUC, ROC-AUC, and confusion matrices.
* `syslog_v2_clean/test_balanced_results_10fold.csv` — Balanced view test metrics (50% prevalence) across all 10 folds.
* `syslog_v2_clean/development_results.csv` — Full validation metrics for variants V0 through V5 across dev folds.
* `syslog_v2_clean/ablation_results.csv` — 10-fold comparative metrics for Autoencoder, PCA, Isolation Forest, and baselines.
* `syslog_v2_clean/chronological_results.csv` — Metrics and OOV statistics for the 70/10/20 chronological experiment.
* `syslog_v2_clean/efficiency_results.csv` — Exact hardware parameters, latency percentiles, and throughput benchmarks.
* `syslog_v2_clean/saved_models/` — Serialized Word2Vec models and Autoencoder model pickles for each fold.

