"""
Paper-exact reproduction of Fält, Forsström, He, Zhang (ICSRS 2025).

Implements every formula from the paper literally, as written:

  Feature:   TF-IDF vector (Eq. 2) — global IDF over all sequences
             NO L2 normalisation (not stated in paper)
             Negative IDF values are retained as-is

  PCA:       Mean-center X; SVD X = UΣV^T; P = V_k (k=6)
             Threshold θ = 95th-pctile of TEST reconstruction errors (Eq. 7)

  RPCA:      Inexact ALM; SVD L^T = UΣV^T; P = U_k (k=9) [left sing. vecs of L^T]
             Same threshold rule as PCA

  Evaluation: 5-fold CV, 25 000 normal + 1 000 abnormal
              Fit on 5 000 pure-normal sequences per fold
              Test on remaining 20 000 normal + 1 000 abnormal

Also run as comparison:
  raw_l2 + PCA k=6 + count-threshold → our empirical best (F1=94.56)

Produces the final side-by-side table.

Run from project root with venv activated:
    python src/paper_exact.py
"""

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

FEATURES_PATH    = "data/HDFS_v1/preprocessed/features.npz"
N_NORMAL_SAMPLE  = 25_000
N_ABNORMAL_SAMPLE = 1_000
N_FOLDS          = 5
FOLD_SIZE        = N_NORMAL_SAMPLE // N_FOLDS   # 5 000
PCA_K            = 6
RPCA_K           = 9
RANDOM_SEED      = 42


# ---------------------------------------------------------------------------
# RPCA — Inexact Augmented Lagrange Multiplier  (Lin, Chen & Ma 2010)
# λ = 1 / sqrt(max(n, d))  [Eq. 4]
# ---------------------------------------------------------------------------

def soft_threshold(X, tau):
    return np.sign(X) * np.maximum(np.abs(X) - tau, 0)


def inexact_alm_rpca(M, lam=None, tol=1e-7, max_iter=500):
    """Solves min ||L||_* + λ||S||_1 s.t. M = L + S.
    Returns (L, S, rank_L, sv_thresh)."""
    n, d = M.shape
    if lam is None:
        lam = 1.0 / np.sqrt(max(n, d))   # Eq. 4

    norm_two = np.linalg.norm(M, 2)
    norm_inf = np.max(np.abs(M)) / lam
    J   = max(norm_two, norm_inf)
    Y   = M / J

    mu     = 1.25 / norm_two
    mu_bar = mu * 1e7
    rho    = 1.5

    L      = np.zeros_like(M)
    S      = np.zeros_like(M)
    norm_M = np.linalg.norm(M, "fro")
    sv_thresh = np.array([])

    for _ in range(max_iter):
        # Low-rank update via SVD thresholding
        U, sv, Vt = np.linalg.svd(M - S + Y / mu, full_matrices=False)
        sv_thresh  = np.maximum(sv - 1.0 / mu, 0.0)
        L = U @ np.diag(sv_thresh) @ Vt

        # Sparse update via soft-thresholding
        temp = M - L + Y / mu
        S    = soft_threshold(temp, lam / mu)

        Z  = M - L - S
        Y  = Y + mu * Z
        mu = min(mu * rho, mu_bar)

        if np.linalg.norm(Z, "fro") / norm_M < tol:
            break

    rank_L = int(np.sum(sv_thresh > 1e-6))
    return L, S, rank_L, sv_thresh


# ---------------------------------------------------------------------------
# Projection matrix helpers
# ---------------------------------------------------------------------------

def pca_projection(X_fit, k):
    """
    Paper Sec. III-B: mean-center X, SVD: X = UΣV^T, P = V_k.
    V_k = top-k RIGHT singular vectors of centered X = rows of Vt[:k].
    Returns (P, mean).
    """
    mean  = X_fit.mean(axis=0)
    X_c   = X_fit - mean
    _, _, Vt = np.linalg.svd(X_c, full_matrices=False)
    P = Vt[:k].T   # shape (d, k)
    return P, mean


def rpca_projection(L, k):
    """
    Paper Sec. III-C: SVD of L^T: L^T = UΣV^T, P = U_k (top-k LEFT sing. vecs of L^T).
    Left singular vectors of L^T = right singular vectors of L = rows of Vt_L[:k].
    Returns P of shape (d, k).
    """
    _, _, Vt = np.linalg.svd(L, full_matrices=False)
    k = min(k, Vt.shape[0])
    P = Vt[:k].T   # shape (d, k)  — mathematically = U_k of L^T
    return P


# ---------------------------------------------------------------------------
# Anomaly scoring — Eq. 5-7
# ---------------------------------------------------------------------------

def recon_errors(X, P):
    """e_i = ||x_i - x_i P P^T||_2   (Eq. 6)"""
    X_proj = X @ P @ P.T
    return np.linalg.norm(X - X_proj, axis=1)


def threshold_test_pctile(errors, pctile=95):
    """
    θ = Percentile_95({e_1, ..., e_m})   (Eq. 7, paper-exact).
    Computed on TEST errors directly.
    """
    return np.percentile(errors, pctile)


def threshold_count(errors, fraction=0.05):
    """
    Equivalent rank-based: flag top-fraction% by count.
    Used as our empirical comparison baseline.
    """
    n_flag = int(np.ceil(fraction * len(errors)))
    return np.sort(errors)[-n_flag]   # returns θ value


def classify(errors, theta):
    return (errors > theta).astype(int)


def metrics(y_test, y_pred):
    p, r, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", pos_label=1, zero_division=0)
    return p * 100, r * 100, f1 * 100


# ---------------------------------------------------------------------------
# Single-fold evaluation
# ---------------------------------------------------------------------------

def eval_fold(X_fit, X_test, y_test):
    """
    Run paper-exact PCA and RPCA on one fold.
    Returns dict of (P, R, F1) for each variant.
    """
    out = {}

    # ── Paper-exact PCA ──────────────────────────────────────────────────
    P_pca, mean_pca = pca_projection(X_fit, PCA_K)
    Xt_c = X_test - mean_pca
    et_pca = recon_errors(Xt_c, P_pca)

    # Paper threshold: 95th-pctile of TEST errors
    theta_pca_test = threshold_test_pctile(et_pca)
    out["PCA | TF-IDF | test-pctile"] = metrics(y_test, classify(et_pca, theta_pca_test))

    # Comparison: count-based (top-5%) on test
    theta_pca_cnt = threshold_count(et_pca)
    out["PCA | TF-IDF | count-5%"] = metrics(y_test, classify(et_pca, theta_pca_cnt))

    # ── Paper-exact RPCA ─────────────────────────────────────────────────
    # RPCA applied to RAW (uncentered) fitting data — paper Eq. 3 uses X directly
    L, S, rank_L, sv_thresh = inexact_alm_rpca(X_fit)

    P_rpca = rpca_projection(L, RPCA_K)
    # Test data NOT centered for RPCA (paper does not specify centering for RPCA)
    et_rpca = recon_errors(X_test, P_rpca)

    theta_rpca_test = threshold_test_pctile(et_rpca)
    out[f"RPCA | TF-IDF | test-pctile | rank(L)={rank_L}"] = metrics(
        y_test, classify(et_rpca, theta_rpca_test))

    theta_rpca_cnt = threshold_count(et_rpca)
    out[f"RPCA | TF-IDF | count-5%   | rank(L)={rank_L}"] = metrics(
        y_test, classify(et_rpca, theta_rpca_cnt))

    return out, rank_L


# ---------------------------------------------------------------------------
# Main 5-fold CV
# ---------------------------------------------------------------------------

def main():
    data    = np.load(FEATURES_PATH)
    X_tfidf = data["X_tfidf"]
    X_raw   = data["X_counts"].astype(np.float32)
    y       = data["y"]

    print(f"TF-IDF shape: {X_tfidf.shape}")
    print(f"Raw    shape: {X_raw.shape}")
    print(f"Labels: {y.shape}  anomaly={y.mean()*100:.2f}%")
    print(f"\nNegative TF-IDF values: {(X_tfidf < 0).sum():,}  "
          f"({(X_tfidf < 0).mean()*100:.2f}%)")

    rng = np.random.default_rng(RANDOM_SEED)
    normal_idx   = np.where(y == 0)[0]
    abnormal_idx = np.where(y == 1)[0]

    sampled_normal   = rng.choice(normal_idx,   size=N_NORMAL_SAMPLE,   replace=False)
    sampled_abnormal = rng.choice(abnormal_idx, size=N_ABNORMAL_SAMPLE, replace=False)
    rng.shuffle(sampled_normal)

    # Also prepare L2-normalised raw counts (our v3 best)
    def l2_norm(X):
        n = np.linalg.norm(X, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return X / n

    X_raw_l2 = l2_norm(X_raw)

    # Accumulators
    agg = {}
    agg_raw = {}  # raw_l2 PCA reference

    def rec(d, key, p, r, f1):
        if key not in d:
            d[key] = {"p": [], "r": [], "f1": []}
        d[key]["p"].append(p)
        d[key]["r"].append(r)
        d[key]["f1"].append(f1)

    for fold in range(N_FOLDS):
        start = fold * FOLD_SIZE
        end   = start + FOLD_SIZE

        fit_idx         = sampled_normal[start:end]
        test_normal_idx = np.concatenate([sampled_normal[:start],
                                          sampled_normal[end:]])
        test_idx        = np.concatenate([test_normal_idx, sampled_abnormal])
        y_test          = y[test_idx]

        Xf_tfidf = X_tfidf[fit_idx]
        Xt_tfidf = X_tfidf[test_idx]

        print(f"\n{'='*68}")
        print(f"FOLD {fold+1}/{N_FOLDS}  "
              f"fit={len(fit_idx):,}  "
              f"test={len(test_idx):,}  "
              f"({len(test_normal_idx):,} normal + {N_ABNORMAL_SAMPLE} abnormal)")
        print(f"{'='*68}")

        # ── Paper-exact on TF-IDF ─────────────────────────────────────
        fold_out, rank_L = eval_fold(Xf_tfidf, Xt_tfidf, y_test)
        for key, (p, r, f1) in fold_out.items():
            rec(agg, key, p, r, f1)
            print(f"  {key:<55s}  P={p:6.2f}  R={r:6.2f}  F1={f1:6.2f}")

        # ── Our best: raw_l2 + PCA k=6 + count threshold ──────────────
        Xf_raw = X_raw_l2[fit_idx]
        Xt_raw = X_raw_l2[test_idx]

        P_raw, mean_raw = pca_projection(Xf_raw, PCA_K)
        Xt_raw_c = Xt_raw - mean_raw
        et_raw = recon_errors(Xt_raw_c, P_raw)
        theta_raw = threshold_count(et_raw)
        p, r, f1 = metrics(y_test, classify(et_raw, theta_raw))
        rec(agg_raw, "PCA | raw_l2 | count-5% | (our best)", p, r, f1)
        print(f"  {'PCA | raw_l2 | count-5% | (our best)':<55s}  "
              f"P={p:6.2f}  R={r:6.2f}  F1={f1:6.2f}")

    # ── Final summary table ──────────────────────────────────────────────
    print(f"\n\n{'='*80}")
    print("FINAL RESULTS — 5-fold average")
    print(f"{'='*80}")
    print(f"  {'Method':<58s}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}")
    print("  " + "─" * 76)

    best_f1 = -1
    best_key = None
    all_rows = list(agg.items()) + list(agg_raw.items())

    for key, vals in all_rows:
        pm  = np.mean(vals["p"])
        rm  = np.mean(vals["r"])
        f1m = np.mean(vals["f1"])
        mark = ""
        if f1m > best_f1:
            best_f1, best_key = f1m, key
            mark = "  ◀ BEST"
        print(f"  {key:<58s}  {pm:>6.2f}  {rm:>6.2f}  {f1m:>6.2f}{mark}")

    print("  " + "─" * 76)
    print(f"\n  {'📄 Paper PCA  (TF-IDF, k=6, test-pctile θ)':<58s}  "
          f"{'92.90':>6}  {'96.45':>6}  {'94.64':>6}")
    print(f"  {'📄 Paper RPCA (TF-IDF, k=9, test-pctile θ)':<58s}  "
          f"{'89.88':>6}  {'91.44':>6}  {'90.55':>6}")
    print(f"\n  Best of ours: {best_key}")
    print(f"  Best F1     : {best_f1:.2f}")

    # ── Threshold diagnostic ─────────────────────────────────────────────
    print(f"\n\n{'='*68}")
    print("THRESHOLD DIAGNOSTIC — fold-1 TF-IDF PCA")
    print(f"{'='*68}")
    rng2 = np.random.default_rng(RANDOM_SEED)
    sn2  = rng2.choice(normal_idx, size=N_NORMAL_SAMPLE, replace=False)
    rng2.shuffle(sn2)
    sa2  = rng2.choice(abnormal_idx, size=N_ABNORMAL_SAMPLE, replace=False)

    fit_idx2 = sn2[:FOLD_SIZE]
    test_idx2 = np.concatenate([sn2[FOLD_SIZE:], sa2])
    y_t2 = y[test_idx2]

    Xf2 = X_tfidf[fit_idx2]
    Xt2 = X_tfidf[test_idx2]
    P2, m2 = pca_projection(Xf2, PCA_K)
    et2 = recon_errors(Xt2 - m2, P2)

    normal_e   = et2[y_t2 == 0]
    abnormal_e = et2[y_t2 == 1]

    print(f"  Normal   errors: min={normal_e.min():.5f}  "
          f"p5={np.percentile(normal_e,5):.5f}  "
          f"median={np.median(normal_e):.5f}  "
          f"p95={np.percentile(normal_e,95):.5f}  "
          f"max={normal_e.max():.5f}")
    print(f"  Abnormal errors: min={abnormal_e.min():.5f}  "
          f"p5={np.percentile(abnormal_e,5):.5f}  "
          f"median={np.median(abnormal_e):.5f}  "
          f"p95={np.percentile(abnormal_e,95):.5f}  "
          f"max={abnormal_e.max():.5f}")

    theta_test = np.percentile(et2, 95)
    theta_cnt  = threshold_count(et2)
    print(f"\n  θ (95th-pctile of all test errors): {theta_test:.6f}")
    print(f"  θ (top-5% count equivalent)       : {theta_cnt:.6f}")
    print(f"  # flagged with test-pctile θ      : {(et2 > theta_test).sum()}")
    print(f"  # flagged with count θ             : {(et2 > theta_cnt).sum()}")
    print(f"  # true anomalies caught (pctile θ) : {((et2 > theta_test) & (y_t2==1)).sum()}")
    print(f"  # true anomalies caught (count θ)  : {((et2 > theta_cnt) & (y_t2==1)).sum()}")

    # Overlap analysis: how many anomalies have errors in the normal range?
    max_normal_err = normal_e.max()
    hidden_ab = (abnormal_e <= max_normal_err).sum()
    print(f"\n  Anomalies with error <= max(normal error) [{max_normal_err:.5f}]: "
          f"{hidden_ab} / {len(abnormal_e)}  ({hidden_ab/len(abnormal_e)*100:.1f}%)")
    print("  → These are 'hidden' anomalies that NO threshold can recover")
    print("    unless the feature space separates them better.")


if __name__ == "__main__":
    main()
