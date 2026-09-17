"""
PCA vs Robust PCA anomaly detection evaluation on HDFS_v1.

Reproduces the paper's "Evaluation without anomalies in fitting data"
experiment (Section V-A, Table I): 5-fold cross-validation using 25,000
normal + 1,000 abnormal sequences, comparing standard PCA (k=6) against
Robust PCA (k=9).

Requires features.npz from feature_extraction.py to already exist.

Run from project root with venv activated:
    python src/evaluate_pca_rpca.py
"""

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

FEATURES_PATH = "data/HDFS_v1/preprocessed/features.npz"

N_NORMAL_SAMPLE = 25000
N_ABNORMAL_SAMPLE = 1000
N_FOLDS = 5
FOLD_SIZE = N_NORMAL_SAMPLE // N_FOLDS  # 5000, used for fitting each fold

PCA_K = 6
RPCA_K = 9

RANDOM_SEED = 42  # paper doesn't specify a seed; we fix one for reproducibility


# ---------------------------------------------------------------------------
# Robust PCA via inexact Augmented Lagrange Multiplier (Lin, Chen & Ma, 2010)
# ---------------------------------------------------------------------------

def soft_threshold(X, tau):
    """Entrywise soft-thresholding operator, used for the sparse component."""
    return np.sign(X) * np.maximum(np.abs(X) - tau, 0)


def svd_threshold(X, tau):
    """Singular value soft-thresholding, used for the low-rank component."""
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    S_thresh = np.maximum(S - tau, 0)
    return U @ np.diag(S_thresh) @ Vt


def inexact_alm_rpca(M, lam=None, tol=1e-7, max_iter=500):
    """
    Solves: min_{L,S} ||L||_* + lambda*||S||_1  s.t.  M = L + S
    via the inexact ALM method. Returns (L, S).
    """
    n, d = M.shape
    if lam is None:
        lam = 1 / np.sqrt(max(n, d))  # matches Eq. 4 in the paper

    norm_two = np.linalg.norm(M, 2)
    norm_inf = np.max(np.abs(M)) / lam
    J = max(norm_two, norm_inf)
    Y = M / J

    mu = 1.25 / norm_two
    mu_bar = mu * 1e7
    rho = 1.5

    L = np.zeros_like(M)
    S = np.zeros_like(M)
    norm_M = np.linalg.norm(M, "fro")

    for _ in range(max_iter):
        L = svd_threshold(M - S + Y / mu, 1 / mu)
        S = soft_threshold(M - L + Y / mu, lam / mu)
        Z = M - L - S
        Y = Y + mu * Z
        mu = min(mu * rho, mu_bar)

        err = np.linalg.norm(Z, "fro") / norm_M
        if err < tol:
            break

    return L, S


# ---------------------------------------------------------------------------
# Projection + anomaly scoring (Eq. 5-7 in the paper, Section III-D)
# ---------------------------------------------------------------------------

def get_projection_matrix(L, k):
    """
    Decomposes L via SVD and returns the top-k right singular vectors as
    the projection matrix P (d x k). Equivalent to the paper's approach of
    decomposing L^T and taking the top-k left singular vectors.
    """
    _, _, Vt = np.linalg.svd(L, full_matrices=False)
    P = Vt[:k].T
    return P


def reconstruction_error(X, P):
    """L2 reconstruction error per row after projecting onto P and back."""
    X_proj = X @ P @ P.T
    return np.linalg.norm(X - X_proj, axis=1)


def evaluate_fold(X_fit, X_test, y_test, k, method):
    if method == "pca":
        # Paper explicitly defines PCA on a mean-centered matrix (Section III-B)
        mean = X_fit.mean(axis=0)
        X_fit_c = X_fit - mean
        X_test_c = X_test - mean
        L = X_fit_c
    elif method == "rpca":
        # RPCA's X = L + S formulation is applied directly to raw data,
        # not centered, consistent with the paper's Eq. 3-4
        X_fit_c = X_fit
        X_test_c = X_test
        L, _ = inexact_alm_rpca(X_fit_c)
    else:
        raise ValueError(f"Unknown method: {method}")

    P = get_projection_matrix(L, k)
    errors = reconstruction_error(X_test_c, P)

    # Rank-based top-5% selection instead of percentile-value comparison.
    # With only ~200 unique error values in ~21,000 samples, a value-based
    # threshold is unstable (one tied value can span thousands of points).
    # Selecting a fixed COUNT of highest-error samples sidesteps that.
    n_flag = int(np.ceil(0.05 * len(errors)))
    flagged_idx = np.argsort(errors)[-n_flag:]
    y_pred = np.zeros(len(errors), dtype=int)
    y_pred[flagged_idx] = 1

    print(f"    [debug] flagged: {y_pred.sum()}, unique error values: {len(np.unique(errors))}")

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", pos_label=1, zero_division=0
    )
    return precision, recall, f1


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def main():
    data = np.load(FEATURES_PATH)
    X_tfidf = data["X_tfidf"]
    y = data["y"]

    rng = np.random.default_rng(RANDOM_SEED)

    normal_idx = np.where(y == 0)[0]
    abnormal_idx = np.where(y == 1)[0]

    print(f"Total normal sequences available:   {len(normal_idx):,}")
    print(f"Total abnormal sequences available: {len(abnormal_idx):,}")

    sampled_normal = rng.choice(normal_idx, size=N_NORMAL_SAMPLE, replace=False)
    sampled_abnormal = rng.choice(abnormal_idx, size=N_ABNORMAL_SAMPLE, replace=False)

    rng.shuffle(sampled_normal)  # so the 5 folds are random slices, not ordered

    results = {"pca": {"precision": [], "recall": [], "f1": []},
               "rpca": {"precision": [], "recall": [], "f1": []}}

    for fold in range(N_FOLDS):
        start = fold * FOLD_SIZE
        end = start + FOLD_SIZE
        fit_idx = sampled_normal[start:end]
        test_normal_idx = np.concatenate(
            [sampled_normal[:start], sampled_normal[end:]]
        )
        test_idx = np.concatenate([test_normal_idx, sampled_abnormal])

        X_fit = X_tfidf[fit_idx]
        X_test = X_tfidf[test_idx]
        y_test = y[test_idx]

        print(f"\n{'=' * 50}")
        print(f"FOLD {fold + 1}/{N_FOLDS}")
        print(f"{'=' * 50}")
        print(f"Fitting on {len(fit_idx):,} normal sequences")
        print(f"Testing on {len(test_idx):,} sequences "
              f"({len(test_normal_idx):,} normal + {len(sampled_abnormal):,} abnormal)")

        for method, k in [("pca", PCA_K), ("rpca", RPCA_K)]:
            p, r, f1 = evaluate_fold(X_fit, X_test, y_test, k, method)
            results[method]["precision"].append(p)
            results[method]["recall"].append(r)
            results[method]["f1"].append(f1)
            print(f"  {method.upper():5s} (k={k}): "
                  f"P={p*100:.2f}  R={r*100:.2f}  F1={f1*100:.2f}")

    print(f"\n{'=' * 60}")
    print("FINAL RESULTS (averaged over 5 folds) vs. PAPER'S TABLE I")
    print(f"{'=' * 60}")
    print(f"{'Method':<10}{'Precision':<12}{'Recall':<12}{'F1':<12}")
    print(f"{'-' * 46}")
    for method in ["pca", "rpca"]:
        p_mean = np.mean(results[method]["precision"]) * 100
        r_mean = np.mean(results[method]["recall"]) * 100
        f1_mean = np.mean(results[method]["f1"]) * 100
        print(f"{method.upper():<10}{p_mean:<12.2f}{r_mean:<12.2f}{f1_mean:<12.2f}")

    print(f"\n{'-' * 46}")
    print("Paper's Table I reference values:")
    print(f"{'PCA':<10}{'92.90':<12}{'96.45':<12}{'94.64':<12}")
    print(f"{'RPCA':<10}{'89.88':<12}{'91.44':<12}{'90.55':<12}")


if __name__ == "__main__":
    main()