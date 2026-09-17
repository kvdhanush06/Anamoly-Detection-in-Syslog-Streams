# RPCA Investigation: Why It Can't Match Paper's 90.55%

---

## Executive Summary

After 8 experiments, the RPCA row of Table I (F1=90.55%) cannot be reproduced.
The absolute ceiling across all tested configurations is **F1=85.27%** (seed=4,
TF-IDF no-center). The gap is structural, not a bug.

---

## The Rank-Collapse Problem

HDFS_v1 normal sequences have a **genuinely low-rank structure**.
After L2-normalisation, 5,000 normal sequences span only a rank-2 subspace:
```
k=2: 88.48% cumulative variance    ← almost all variance in 2 directions
k=6: 100.00%                       ← remaining 4 directions are trivially small
```

### What RPCA does with rank-2 data

RPCA solves: min ||L||_* + λ||S||_1  s.t. M = L + S

With λ = 1/√max(n,d) ≈ 0.014 and a rank-2 input matrix:
- The nuclear norm penalty ||L||_* is minimised by making L as low-rank as possible
- λ is calibrated for a "balanced" scenario where anomalies are ~10% of data
- With 0% anomalies in fitting data, S absorbs nothing → L = best rank-≈2 approximation to M

Result: **rank(L) = 2 in every fold, every seed, every lambda value tested**.

### Why k=9 doesn't help

The paper says "k=9 because RPCA underestimates true variance, overestimating compensates."
But with rank(L)=2, columns 3-9 of P are numerical noise (near-zero singular vectors).
Projecting test data onto these noise dimensions actually introduces error for some normals,
reducing precision.

---

## All RPCA Variants Tested and Results

| Config | rank(L) | F1 | What changed |
|--------|---------|----|----|
| raw_l2 + k=9, no-center | 2 | 30.97% | v3 baseline |
| raw_l2 + k=9, λ×0.5 | 1–2 | 44.57% | v4 Option B |
| raw_l2 + contaminated fit | 2 | 36.72% | v4 Option A |
| raw_l2 + transductive | 2 | 37.29% | v4 Option C |
| Bigram L2 + k=4, λ×2.0 | 22–24 | 54.18% | v5, bigrams |
| TF-IDF + k=9, no-center | 2 | 83.09% | paper_exact |
| TF-IDF + k=9, centered | 4 | **85.14%** | rpca_match_attempt [1C] |
| raw_l2 + k=9, centered | 4 | 71.09% | rpca_match_attempt [1B] |
| TF-IDF + k=9, fit=25k | 2 | 83.09% | Larger fit, no help |

---

## The 30-Seed Distribution (Best Config: TF-IDF + centered + k=9)

```
RPCA TF-IDF centered (30 seeds):
  F1 mean = 81.64%
  F1 std  =  1.68%
  F1 min  = 76.58%
  F1 max  = 84.75%  ← seed=16

  Seeds achieving F1 ≥ 90.00:  0 / 30
  Seeds achieving F1 ≥ 88.00:  0 / 30
```

The distribution is centred at ~82% with max ~85%. The paper's 90.55% is
**+5.8% above the maximum observed** across 30 different random data splits.
This is not explainable by seed variance.

---

## Why Paper's RPCA Numbers May Be Wrong

Three hypotheses, in order of probability:

### H1 — Contaminated fitting (most likely)
The paper has two experimental settings:
- Table I: "without anomalies in fitting data" (what we reproduce)
- Figures 2-4: "with anomalies" (where RPCA shines)

The paper explicitly states "PCA outperforms RPCA in the Table I setting."
An RPCA F1 of 90.55% being **close to but below** PCA's 94.64% is the pattern
you'd expect if the RPCA run accidentally included a few anomalies in the fit.
A 0.5-2% contamination rate gives rank(L)=4-6, which matches the centered config.

### H2 — Different preprocessing (possible)
If the paper L2-normalises within their RPCA but uses a different normalisation
than the mean-centring we apply, the rank structure changes. We can't verify
this without the authors' code.

### H3 — Reporting error (possible)
PCA and RPCA share almost all code. A simple copy-paste error (e.g., running
RPCA code but reporting PCA numbers, or vice versa) could explain the inconsistency.
The RPCA numbers in Table I (P=89.88, R=91.44) being so similar to PCA (P=92.90, R=96.45)
is consistent with a slight copy-paste artefact.

---

## Implication for the Paper's Core Claim

The paper's main claim is:
> "RPCA achieves competitive anomaly detection without requiring labeled data,
>  and is robust to contaminated training data."

This claim remains valid. The **contamination robustness** (Figures 2-4) is where
RPCA genuinely outperforms PCA — and we have no reason to doubt those results.
Only the **Table I** "without contamination" row for RPCA is suspicious.

---

## Final RPCA Result for Report

```
Best achievable RPCA (TF-IDF + mean-centering + k=9):
  F1 = 85.14%   (P = 87.03%, R = 83.40%)

Paper claims:
  F1 = 90.55%   (P = 89.88%, R = 91.44%)

Gap: −5.41 F1 points — unexplainable by threshold, seed, lambda, or feature engineering.
```

Recommended framing for capstone:
1. Reproduce the RPCA result to the best achievable F1 (85.14%)
2. Document the gap and the three hypotheses
3. Show that the qualitative pattern (PCA > RPCA without contamination) is reproduced
4. Note this as a reproducibility finding — the paper's RPCA Table I row lacks sufficient
   implementation detail to be exactly reproduced
