"""
PCA vs Robust PCA anomaly detection evaluation on HDFS_v1 — v2.

Fixes identified by diagnostics.py:

  FIX A: Test both TF-IDF and raw event counts as input features.
         The paper's k=6 for PCA is "explains ~95% variance" — most likely
         measured on raw counts, not TF-IDF.

  FIX B: Compute IDF weights fold-wise (fitting data only), then apply to
         test data.  The v1 code computed IDF globally — subtle leakage.

  FIX C: For RPCA, also try rank-adaptive k = rank(L) instead of hardcoded
         k=9.  Diagnostics showed rank(L)=2 with TF-IDF, so projecting onto
         k=9 adds 7 noise dimensions.  Also try column-normalised input to
         encourage a higher rank(L).

  FIX D: Test both threshold strategies:
         - "value" : θ = 95th-percentile of reconstruction errors on the
                         FITTING data (model-derived, inductive — Eq. 7)
         - "count" : top-5% of test-set errors by rank (previous approach)

Run from project root with venv activated:
    python src/evaluate_pca_rpca_v2.py
"""

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

FEATURES_PATH = "data/HDFS_v1/preprocessed/features.npz"

N_NORMAL_SAMPLE  = 25_000
N_ABNORMAL_SAMPLE = 1_000
N_FOLDS   = 5
FOLD_SIZE = N_NORMAL_SAMPLE // N_FOLDS   # 5 000

PCA_K  = 6
RPCA_K = 9

RANDOM_SEED = 42
ANOMALY_FRACTION = 0.05   # paper: 95th-percentile ≈ top-5%


# ---------------------------------------------------------------------------
# Robust PCA — inexact ALM (Lin, Chen & Ma 2010)
# ---------------------------------------------------------------------------

def soft_threshold(X, tau):
    return np.sign(X) * np.maximum(np.abs(X) - tau, 0)


def inexact_alm_rpca(M, lam=None, tol=1e-7, max_iter=500):
    """Returns (L, S, rank_L) where rank_L = effective rank of L."""
    n, d = M.shape
    if lam is None:
        lam = 1.0 / np.sqrt(max(n, d))

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
    sv_thresh = None

    for _ in range(max_iter):
        U, sv, Vt = np.linalg.svd(M - S + Y / mu, full_matrices=False)
        sv_thresh  = np.maximum(sv - 1.0 / mu, 0.0)
        L = U @ np.diag(sv_thresh) @ Vt

        temp = M - L + Y / mu
        S    = soft_threshold(temp, lam / mu)

        Z   = M - L - S
        Y   = Y + mu * Z
        mu  = min(mu * rho, mu_bar)

        if np.linalg.norm(Z, "fro") / norm_M < tol:
            break

    rank_L = int(np.sum(sv_thresh > 1e-6)) if sv_thresh is not None else 0
    return L, S, rank_L


# ---------------------------------------------------------------------------
# Projection + anomaly scoring
# ---------------------------------------------------------------------------

def get_projection_matrix(L, k):
    _, _, Vt = np.linalg.svd(L, full_matrices=False)
    k = min(k, Vt.shape[0])
    return Vt[:k].T


def reconstruction_errors(X, P):
    X_proj = X @ P @ P.T
    return np.linalg.norm(X - X_proj, axis=1)


# ---------------------------------------------------------------------------
# Fold-wise TF-IDF (FIX B)
# ---------------------------------------------------------------------------

def tfidf_from_counts(X_counts_fit, X_counts_test):
    """IDF computed from fitting data only."""
    N_fit  = X_counts_fit.shape[0]
    df_fit = (X_counts_fit > 0).sum(axis=0)
    idf    = np.log(N_fit / (1.0 + df_fit))

    def to_tfidf(X):
        row_sums = X.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return (X / row_sums) * idf

    return to_tfidf(X_counts_fit), to_tfidf(X_counts_test)


# ---------------------------------------------------------------------------
# Threshold + classify (FIX D)
# ---------------------------------------------------------------------------

def classify(err_fit, err_test, y_test, threshold_method):
    if threshold_method == "value":
        theta  = np.percentile(err_fit, 100.0 * (1.0 - ANOMALY_FRACTION))
        y_pred = (err_test > theta).astype(int)
    else:   # "count"
        n_flag = int(np.ceil(ANOMALY_FRACTION * len(err_test)))
        flags  = np.argsort(err_test)[-n_flag:]
        y_pred = np.zeros(len(err_test), dtype=int)
        y_pred[flags] = 1

    p, r, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", pos_label=1, zero_division=0
    )
    return p, r, f1


# ---------------------------------------------------------------------------
# Single-fold evaluation
# ---------------------------------------------------------------------------

def evaluate_fold(X_fit_raw, X_test_raw, y_test,
                  feature_type="raw", pca_k=PCA_K, rpca_k=RPCA_K,
                  threshold_method="count"):
    # --- feature prep ---
    if feature_type == "raw":
        X_fit, X_test = X_fit_raw.copy(), X_test_raw.copy()
    elif feature_type == "tfidf":
        X_fit, X_test = tfidf_from_counts(X_fit_raw, X_test_raw)
    else:
        raise ValueError(feature_type)

    out = {}

    # ---- PCA ----
    mean     = X_fit.mean(axis=0)
    Xf_c     = X_fit  - mean
    Xt_c     = X_test - mean
    P_pca    = get_projection_matrix(Xf_c, pca_k)
    ef_pca   = reconstruction_errors(Xf_c, P_pca)
    et_pca   = reconstruction_errors(Xt_c, P_pca)
    out["pca"] = classify(ef_pca, et_pca, y_test, threshold_method)

    # ---- RPCA (hardcoded k) ----
    L, _S, rank_L = inexact_alm_rpca(X_fit)
    P_r9  = get_projection_matrix(L, rpca_k)
    ef_r9 = reconstruction_errors(X_fit, P_r9)
    et_r9 = reconstruction_errors(X_test, P_r9)
    out["rpca_k9"] = classify(ef_r9, et_r9, y_test, threshold_method)

    # ---- RPCA (rank-adaptive k — FIX C) ----
    k_ada = max(rank_L, 1)
    P_ada = get_projection_matrix(L, k_ada)
    ef_a  = reconstruction_errors(X_fit, P_ada)
    et_a  = reconstruction_errors(X_test, P_ada)
    out[f"rpca_ada(rank={rank_L})"] = classify(ef_a, et_a, y_test, threshold_method)

    # ---- RPCA on column-normalised data ----
    std = X_fit.std(axis=0)
    std[std == 0] = 1.0
    Xf_n = X_fit  / std
    Xt_n = X_test / std
    L_n, _Sn, rank_Ln = inexact_alm_rpca(Xf_n)
    k_n  = max(rank_Ln, 1)
    P_n  = get_projection_matrix(L_n, k_n)
    ef_n = reconstruction_errors(Xf_n, P_n)
    et_n = reconstruction_errors(Xt_n, P_n)
    out[f"rpca_colnorm(rank={rank_Ln})"] = classify(ef_n, et_n, y_test, threshold_method)

    return out, rank_L, rank_Ln


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    data     = np.load(FEATURES_PATH)
    X_counts = data["X_counts"]
    y        = data["y"]

    rng = np.random.default_rng(RANDOM_SEED)

    normal_idx   = np.where(y == 0)[0]
    abnormal_idx = np.where(y == 1)[0]

    print(f"Total normal:   {len(normal_idx):,}")
    print(f"Total abnormal: {len(abnormal_idx):,}")

    sampled_normal   = rng.choice(normal_idx,   size=N_NORMAL_SAMPLE,   replace=False)
    sampled_abnormal = rng.choice(abnormal_idx, size=N_ABNORMAL_SAMPLE, replace=False)
    rng.shuffle(sampled_normal)

    configs = [
        ("raw",   "count"),
        ("raw",   "value"),
        ("tfidf", "count"),
        ("tfidf", "value"),
    ]

    agg = {}   # {(feat, thresh, method_name): {"p":[], "r":[], "f1":[]}}

    for fold in range(N_FOLDS):
        start = fold * FOLD_SIZE
        end   = start + FOLD_SIZE

        fit_idx         = sampled_normal[start:end]
        test_normal_idx = np.concatenate([sampled_normal[:start], sampled_normal[end:]])
        test_idx        = np.concatenate([test_normal_idx, sampled_abnormal])

        X_fit_raw  = X_counts[fit_idx]
        X_test_raw = X_counts[test_idx]
        y_test     = y[test_idx]

        print(f"\n{'=' * 60}")
        print(f"FOLD {fold+1}/{N_FOLDS}  "
              f"fit={len(fit_idx):,}  "
              f"test={len(test_idx):,} "
              f"({len(test_normal_idx):,} normal + {N_ABNORMAL_SAMPLE} abnormal)")
        print(f"{'=' * 60}")

        for feat, thresh in configs:
            fold_out, rank_L, rank_Ln = evaluate_fold(
                X_fit_raw, X_test_raw, y_test,
                feature_type=feat, pca_k=PCA_K, rpca_k=RPCA_K,
                threshold_method=thresh,
            )
            print(f"\n  feat={feat:5s}  thresh={thresh:5s}  "
                  f"rank(L_raw)={rank_L}  rank(L_colnorm)={rank_Ln}")
            for mname, (p, r, f1) in fold_out.items():
                key = (feat, thresh, mname)
                if key not in agg:
                    agg[key] = {"p": [], "r": [], "f1": []}
                agg[key]["p"].append(p)
                agg[key]["r"].append(r)
                agg[key]["f1"].append(f1)
                print(f"    {mname:<30s}  P={p*100:6.2f}  R={r*100:6.2f}  F1={f1*100:6.2f}")

    # Summary table
    print(f"\n\n{'=' * 80}")
    print("FINAL RESULTS — averaged over 5 folds")
    print(f"{'=' * 80}")
    print(f"{'Feat':6s}  {'Thr':5s}  {'Method':<30s}  {'Prec':>8s}  {'Rec':>8s}  {'F1':>8s}")
    print("-" * 80)

    best_f1  = -1
    best_key = None

    for (feat, thresh, mname), vals in sorted(agg.items()):
        pm  = np.mean(vals["p"])  * 100
        rm  = np.mean(vals["r"])  * 100
        f1m = np.mean(vals["f1"]) * 100
        mark = ""
        if f1m > best_f1:
            best_f1  = f1m
            best_key = (feat, thresh, mname)
            mark = "  ◀ BEST"
        print(f"{feat:6s}  {thresh:5s}  {mname:<30s}  "
              f"{pm:>8.2f}  {rm:>8.2f}  {f1m:>8.2f}{mark}")

    print("-" * 80)
    print("\nPaper Table I reference:")
    print("  PCA  — P=92.90  R=96.45  F1=94.64")
    print("  RPCA — P=89.88  R=91.44  F1=90.55")
    if best_key:
        print(f"\nBest: feat={best_key[0]}, thresh={best_key[1]}, method={best_key[2]}"
              f"  →  F1={best_f1:.2f}")


if __name__ == "__main__":
    main()
