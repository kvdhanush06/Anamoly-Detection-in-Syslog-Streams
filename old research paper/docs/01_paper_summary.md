# Paper Summary: Lightweight Optimization based Log-file Anomaly Detection

**Authors:** Markus Fält, Stefan Forsström, Qing He, Tingting Zhang (Mid Sweden University)  
**Venue:** 2025 9th International Conference on System Reliability and Safety (ICSRS)  
**DOI:** 10.1109/ICSRS68021.2025.11422052

---

## Problem Statement

Detect anomalies in system log streams without requiring labeled training data or deep learning infrastructure. The method targets lightweight, real-time deployment on edge/embedded systems where compute is constrained.

---

## Full Architecture

```
Raw Logs ──> Parsing & Grouping (block_id) ──> TF-IDF Matrix X (n × d)
                                                     │
               ┌─────────────────────────────────────┴──────────────────────────────────┐
               ▼                                                                         ▼
       [ Standard PCA ]                                                          [ Robust PCA ]
    • Mean-center X                                                           • Solve min||L||_* + λ||S||_1
    • SVD: X = U Σ V^T                                                        • SVD: L^T = U Σ V^T
    • Projection P = V_k  (k=6)                                               • Projection P = U_k (k=9)
               │                                                                         │
               └─────────────────────────────────────┬──────────────────────────────────┘
                                                     ▼
                                     [ Anomaly Scoring & Inference ]
                                       • X_proj = X_new P P^T
                                       • Error  e = ||X_new - X_proj||_2
                                       • Flag Anomaly if e > θ
                                         where θ = 95th percentile of test errors
```

---

## Step-by-Step Method

### Step 1 — Log Representation

Logs are grouped into sequences by `block_id`. Two feature encodings:

**Event Count Vector (Eq. 1)**
```
x_i = count(v_i in s)
```

**TF-IDF Vector (Eq. 2) — used in experiments**
```
x_i  = tf(v_i, s) * idf(v_i)
tf   = count(v_i in s) / total_events(s)
idf  = log( N / (1 + df(v_i)) )
```
- N = total sequences, df = sequences containing event v_i

### Step 2 — Standard PCA (baseline)

1. Mean-center X ∈ ℝ^{n×d}
2. SVD: X = U Σ V^T
3. **Projection matrix P = V_k** (top-k right singular vectors), k=6
4. k=6 chosen to explain ~95% of total variance AND give best empirical F1

### Step 3 — Robust PCA (main method)

**Objective** (Eq. 3):
```
min_{L,S}  ||L||_*  +  λ||S||_1     s.t.  X = L + S

||L||_*  = Σ σ_i(L)            (nuclear norm  → promotes low rank)
||S||_1  = Σ|S_ij|             (L1 norm       → promotes sparsity)
λ = 1 / √max(n, d)             (Eq. 4)
```

**Solver:** Inexact Augmented Lagrange Multiplier (ALM), Lin, Chen & Ma 2010, arXiv:1009.5055

**Subspace extraction:**
1. SVD of **L^T ∈ ℝ^{d×n}**: L^T = U Σ V^T
2. **P = U_k** (top-k LEFT singular vectors of L^T = top-k right singular vectors of L), k=9
3. k=9 > k=6 because RPCA underestimates true variance; overestimating rank compensates

### Step 4 — Anomaly Scoring (both methods)

| Step | Formula |
|------|---------|
| Project | X_proj = X_new P P^T |
| Residual | e_i = \|\|x_i - x_proj_i\|\|_2 |
| Threshold | θ = Percentile_95({e_1,...,e_m}) |
| Decision | Anomaly if e_i > θ |

---

## Evaluation Protocol

**Dataset:** HDFS_v1 (logpai/loghub), 575,061 block sequences, 2.93% anomalous

**Table I setup — "Without anomalies in fitting data":**
- Sample 25,000 normal + 1,000 abnormal sequences
- 5-fold CV: each fold uses 5,000 normal for fitting
- Test on remaining 20,000 normal + 1,000 abnormal

**Metrics:** Precision, Recall, F1-score

---

## Paper's Target Numbers (Table I)

| Method  | Precision | Recall | F1    |
|---------|-----------|--------|-------|
| DeepLog | 100.00    | 60.90  | 75.70 |
| LogBERT | 24.02     | 83.80  | 37.24 |
| LogFIT  | 99.78     | 90.70  | 95.02 |
| **PCA** | **92.90** | **96.45** | **94.64** |
| **RPCA**| **89.88** | **91.44** | **90.55** |

(DeepLog, LogBERT, LogFIT cited from Almodovar et al., not reproduced by us)

---

## Key Paper Claims

1. PCA outperforms RPCA **without contamination** in fitting data (Table I setting)
2. RPCA outperforms PCA **with contamination** — RPCA's robustness advantage
3. Method is lightweight, no training labels needed
4. Main limitation: discards event order, semantic meaning, parameter values
5. Works well on HDFS specifically (template-friendly); may not generalise to BGL/Thunderbird
