"""
PCA vs Robust PCA anomaly detection evaluation on HDFS_v1 — v3.

Key change over v2: adds L2 row-normalisation of raw event count vectors
before PCA/RPCA.  This is standard practice in the PCA-for-log-anomaly
lineage (He et al. 2016, Xu et al. 2009) and the most likely explanation for
the ~12-point F1 gap that persisted through v2.

Feature variants tested:
  "raw"      — raw event counts (baseline)
  "raw_l2"   — L2-normalised raw counts  (the fix)
  "tfidf_l2" — fold-wise TF-IDF, then L2-normalised

Threshold: top-5% of test-set errors by rank (established as correct in v2).

Run from project root with venv activated:
    python src/evaluate_pca_rpca_v3.py
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
ANOMALY_FRACTION = 0.05


# ---------------------------------------------------------------------------
# RPCA — inexact ALM (Lin, Chen & Ma 2010)
# ---------------------------------------------------------------------------

def soft_threshold(X, tau):
    return np.sign(X) * np.maximum(np.abs(X) - tau, 0)


def inexact_alm_rpca(M, lam=None, tol=1e-7, max_iter=500):
    """Returns (L, S, rank_L)."""
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
    sv_thresh = np.array([])

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

    rank_L = int(np.sum(sv_thresh > 1e-6))
    return L, S, rank_L


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def l2_normalize(X):
    """Row-wise L2 normalisation to unit length."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


def tfidf_from_counts(X_fit_raw, X_test_raw):
    """Fold-wise TF-IDF (IDF from fitting data only)."""
    N_fit  = X_fit_raw.shape[0]
    df_fit = (X_fit_raw > 0).sum(axis=0)
    idf    = np.log(N_fit / (1.0 + df_fit))

    def to_tfidf(X):
        rs = X.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1.0
        return (X / rs) * idf

    return to_tfidf(X_fit_raw), to_tfidf(X_test_raw)


def get_projection_matrix(M, k):
    _, _, Vt = np.linalg.svd(M, full_matrices=False)
    k = min(k, Vt.shape[0])
    return Vt[:k].T


def recon_errors(X, P):
    return np.linalg.norm(X - X @ P @ P.T, axis=1)


def classify_count(err_test, y_test):
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

def evaluate_fold(X_fit_raw, X_test_raw, y_test):
    """
    Tests three feature variants × two methods.
    Returns dict: method_label → (p, r, f1)
    Also returns rank info for logging.
    """
    out     = {}
    ranks   = {}

    # Build feature matrices for each variant
    feature_sets = {}

    # --- raw (unchanged) ---
    feature_sets["raw"] = (X_fit_raw.copy(), X_test_raw.copy())

    # --- raw + L2 ---
    feature_sets["raw_l2"] = (l2_normalize(X_fit_raw), l2_normalize(X_test_raw))

    # --- fold-wise TF-IDF + L2 ---
    Xf_tfidf, Xt_tfidf = tfidf_from_counts(X_fit_raw, X_test_raw)
    feature_sets["tfidf_l2"] = (l2_normalize(Xf_tfidf), l2_normalize(Xt_tfidf))

    for feat_name, (Xf, Xt) in feature_sets.items():

        # ---- PCA ----
        mean  = Xf.mean(axis=0)
        Xf_c  = Xf - mean
        Xt_c  = Xt - mean
        P_pca = get_projection_matrix(Xf_c, PCA_K)
        et    = recon_errors(Xt_c, P_pca)
        out[f"{feat_name}__pca_k{PCA_K}"] = classify_count(et, y_test)

        # ---- RPCA (k=9 fixed) ----
        L, _S, rank_L = inexact_alm_rpca(Xf)
        ranks[feat_name] = rank_L
        P_rpca = get_projection_matrix(L, RPCA_K)
        et_r   = recon_errors(Xt, P_rpca)
        out[f"{feat_name}__rpca_k{RPCA_K}"] = classify_count(et_r, y_test)

        # ---- RPCA (rank-adaptive k) ----
        k_ada = max(rank_L, 1)
        if k_ada != RPCA_K:   # only show if different from k=9
            P_ada = get_projection_matrix(L, k_ada)
            et_a  = recon_errors(Xt, P_ada)
            out[f"{feat_name}__rpca_k{k_ada}(rank-L)"] = classify_count(et_a, y_test)

    return out, ranks


# ---------------------------------------------------------------------------
# Main 5-fold CV
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

    agg = {}   # label → {"p":[], "r":[], "f1":[]}

    for fold in range(N_FOLDS):
        start = fold * FOLD_SIZE
        end   = start + FOLD_SIZE

        fit_idx         = sampled_normal[start:end]
        test_normal_idx = np.concatenate([sampled_normal[:start], sampled_normal[end:]])
        test_idx        = np.concatenate([test_normal_idx, sampled_abnormal])

        X_fit_raw  = X_counts[fit_idx]
        X_test_raw = X_counts[test_idx]
        y_test     = y[test_idx]

        print(f"\n{'=' * 65}")
        print(f"FOLD {fold+1}/{N_FOLDS}  "
              f"fit={len(fit_idx):,}  "
              f"test={len(test_idx):,} "
              f"({len(test_normal_idx):,} normal + {N_ABNORMAL_SAMPLE} abnormal)")
        print(f"{'=' * 65}")

        fold_out, ranks = evaluate_fold(X_fit_raw, X_test_raw, y_test)

        print(f"  rank(L) per feature: {ranks}")
        for label, (p, r, f1) in fold_out.items():
            if label not in agg:
                agg[label] = {"p": [], "r": [], "f1": []}
            agg[label]["p"].append(p)
            agg[label]["r"].append(r)
            agg[label]["f1"].append(f1)
            print(f"  {label:<40s}  P={p*100:6.2f}  R={r*100:6.2f}  F1={f1*100:6.2f}")

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    print(f"\n\n{'=' * 80}")
    print("FINAL RESULTS — averaged over 5 folds")
    print(f"{'=' * 80}")
    print(f"  {'Method':<40s}  {'Prec':>8s}  {'Rec':>8s}  {'F1':>8s}")
    print("  " + "-" * 72)

    best_f1  = -1
    best_lbl = None

    for label, vals in sorted(agg.items()):
        pm  = np.mean(vals["p"])  * 100
        rm  = np.mean(vals["r"])  * 100
        f1m = np.mean(vals["f1"]) * 100
        mark = ""
        if f1m > best_f1:
            best_f1  = f1m
            best_lbl = label
            mark = "  ◀ BEST"
        print(f"  {label:<40s}  {pm:>8.2f}  {rm:>8.2f}  {f1m:>8.2f}{mark}")

    print("  " + "-" * 72)
    print("\n  Paper Table I reference:")
    print("    PCA  — P=92.90  R=96.45  F1=94.64")
    print("    RPCA — P=89.88  R=91.44  F1=90.55")
    if best_lbl:
        print(f"\n  Best config: {best_lbl}  →  F1={best_f1:.2f}")

    # Explained-variance check for L2-normalised raw counts
    print(f"\n\n{'=' * 65}")
    print("EXPLAINED VARIANCE CHECK  (fold-1, raw_l2 features)")
    print(f"{'=' * 65}")
    rng2 = np.random.default_rng(RANDOM_SEED)
    sn   = rng2.choice(np.where(y == 0)[0], size=N_NORMAL_SAMPLE, replace=False)
    rng2.shuffle(sn)
    Xf_raw_l2 = l2_normalize(X_counts[sn[:FOLD_SIZE]])
    mean_l2   = Xf_raw_l2.mean(axis=0)
    Xf_c_l2   = Xf_raw_l2 - mean_l2
    _, sv, _  = np.linalg.svd(Xf_c_l2, full_matrices=False)
    total_var = (sv ** 2).sum()
    for k in range(1, 16):
        ev = (sv[:k] ** 2).sum() / total_var * 100
        marker = "  ← paper's k" if k == PCA_K else ""
        print(f"  k={k:2d}:  explained variance = {ev:6.2f}%{marker}")


if __name__ == "__main__":
    main()
