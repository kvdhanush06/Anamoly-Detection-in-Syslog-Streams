# Project Spec: Lightweight Anomaly Detection for Syslog Streams

This document is the source of truth for this project. Any AI agent or teammate
working on this codebase should read this first before writing or modifying code.

## Base Paper

**Title:** Lightweight Optimization based Log-file Anomaly Detection
**Authors:** Markus Fält, Stefan Forsström, Qing He, Tingting Zhang (Mid Sweden University)
**Venue:** 2025 9th International Conference on System Reliability and Safety (ICSRS)
**DOI:** 10.1109/ICSRS68021.2025.11422052

## Goal

Reproduce the paper's PCA vs. Robust PCA log anomaly detection results on the
HDFS_v1 dataset, then propose and evaluate an improvement (efficiency, accuracy,
or generalization to a second dataset).

## Dataset

**HDFS_v1** (via LogHub, github.com/logpai/loghub)
- 11,175,629 raw log lines, sliced into traces by `block_id`
- ~575,061 total block sequences, ~3% labeled anomalous, ~97% normal
- Files used:
  - `anomaly_label.csv` — ground truth label (normal/anomaly) per block_id
  - `Event_occurrence_matrix.csv` — event count vectors per block (Eq. 1 below)
  - `HDFS_templates.csv` — vocabulary of unique log event templates
  - `Event_traces.csv` — raw ordered event sequences per block (not required by
    this method, since it discards event order — kept for reference/sanity checks)

## Method

### 1. Log Representation

Logs are grouped into sequences by `block_id`. Each sequence is converted into a
fixed-dimensional numeric vector using one of two encodings:

**Event Count Vector** (Eq. 1)
For vocabulary `V = {v1, ..., vd}` of unique log templates, sequence `s` maps to
vector `x ∈ R^d` where:
```
x_i = count(v_i in s)
```

**TF-IDF Vector** (Eq. 2) — the encoding actually used for the experiments
```
x_i = tf(v_i, s) * idf(v_i)

tf(v_i, s) = count(v_i in s) / sum_j count(v_j in s)
idf(v_i)   = log( N / (1 + df(v_i)) )
```
- `N` = total number of log sequences
- `df(v_i)` = number of sequences containing event `v_i`

### 2. Standard PCA baseline

- Mean-center data matrix `X`, decompose via SVD: `X = U Σ V^T`
- Keep top `k` components: `X_k = X V_k`
- **k = 6** for standard PCA (chosen to explain ~95% of variance; also empirically
  optimal for F1 in the paper)

### 3. Robust PCA (RPCA) — the main method

Decomposes data matrix into low-rank + sparse components:
```
X = L + S
min_{L,S} ||L||_* + λ||S||_1   s.t.  X = L + S
```
- `||L||_*` = nuclear norm (sum of singular values) — promotes low rank
- `||S||_1` = L1 norm — promotes sparsity (this captures the anomalies)
- `λ = 1 / sqrt(max(n, d))`
- Solved via **inexact Augmented Lagrange Multiplier (ALM)** method
  (Lin, Chen & Ma, 2010, arXiv:1009.5055) — fast/scalable version, not the
  general convex solver (which doesn't scale)
- **k = 9** for Robust PCA (higher than PCA's k=6 because the RPCA projection
  matrix underestimates true variance, so rank must be overestimated to
  capture anomalous events — this was empirically optimal for F1 in the paper)

### 4. Anomaly scoring (applies to both PCA and RPCA)

1. Decompose `L^T = U Σ V^T`, take top-k left singular vectors as projection
   matrix `P = U_k`
2. For new data `X_new`, project: `X_proj = X_new P P^T`
3. Compute reconstruction error: `||X_new - X_proj||_2`
4. **Anomaly if error > θ**, where θ = 95th percentile of the testing error
   (chosen as a rough estimate of the true ~3% anomaly ratio in HDFS)

## Evaluation Protocol

- Metrics: **Precision, Recall, F1-score**
- Two experimental setups in the paper:
  1. **With anomalies in fitting data**: random sampling across whole dataset,
     100,000 vectors for testing, fitting size incrementally increased from
     500 to 5,000 (steps of 500), each test repeated 40 times
  2. **Without anomalies in fitting data** (Table I result): 5-fold
     cross-validation using 25,000 normal + 1,000 abnormal sequences; each
     fold uses 5,000 of the 25,000 normal sequences for fitting, remaining
     20,000 normal + 1,000 abnormal for testing

## Target Numbers to Reproduce (Table I, 5-fold CV, no anomalies in fitting data)

| Method  | Precision | Recall | F1    |
|---------|-----------|--------|-------|
| DeepLog | 100.0     | 60.90  | 75.70 |
| LogBERT | 24.02     | 83.80  | 37.24 |
| LogFIT  | 99.78     | 90.70  | 95.02 |
| PCA     | 92.90     | 96.45  | 94.64 |
| RPCA    | 89.88     | 91.44  | 90.55 |

(DeepLog, LogBERT, LogFIT numbers are cited from Almodovar et al., not
reproduced by us — our reproduction target is the PCA and RPCA rows.)

## Known Paper Findings (context for interpreting our own results)

- Without anomalies in fitting data, **standard PCA outperforms RPCA** — makes
  sense, since RPCA's robustness advantage doesn't matter if there's nothing
  to be robust against.
- RPCA's advantage shows up when fitting data **does** contain some
  contamination (unlabeled anomalies) — see the paper's Fig. 2-4 experiments.
- Paper's own stated limitation: method discards event order, semantic
  meaning of log messages, and parameter values — assumes well-defined,
  easily-parsed templates. Works well on HDFS specifically because HDFS is
  template-friendly; may not generalize to messier log sources (e.g. BGL,
  Thunderbird).

## Our Gap / Improvement Direction

(Fill in once decided as a team — e.g. testing on BGL/Thunderbird for
generalization, or improving contamination-robustness further, or reducing
model size/inference time further while holding F1.)

## Repo Conventions

- Commit format: `type: description` (feat/fix/data/chore/docs/refactor/test/exp)
- Data files live in `/data` (gitignored, not pushed to GitHub)
- Code lives in `/src`, exploration in `/notebooks`, outputs in `/results`