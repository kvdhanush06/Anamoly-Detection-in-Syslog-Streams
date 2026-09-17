"""
Diagnostics for the PCA/RPCA pipeline: investigates why recall is still
below the paper's reported numbers, and whether RPCA is behaving sensibly.

Run from project root with venv activated:
    python src/diagnostics.py
"""

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from evaluate_pca_rpca import inexact_alm_rpca, get_projection_matrix, reconstruction_error

FEATURES_PATH = "data/HDFS_v1/preprocessed/features.npz"
RANDOM_SEED = 42
N_NORMAL_SAMPLE = 25000
N_ABNORMAL_SAMPLE = 1000
FOLD_SIZE = 5000


def get_fold1_split(y):
    """Reproduces exactly the same fold-1 split used in evaluate_pca_rpca.py."""
    rng = np.random.default_rng(RANDOM_SEED)
    normal_idx = np.where(y == 0)[0]
    abnormal_idx = np.where(y == 1)[0]

    sampled_normal = rng.choice(normal_idx, size=N_NORMAL_SAMPLE, replace=False)
    sampled_abnormal = rng.choice(abnormal_idx, size=N_ABNORMAL_SAMPLE, replace=False)
    rng.shuffle(sampled_normal)

    fit_idx = sampled_normal[:FOLD_SIZE]
    test_normal_idx = sampled_normal[FOLD_SIZE:]
    test_idx = np.concatenate([test_normal_idx, sampled_abnormal])
    return fit_idx, test_idx


def evaluate_at_k(X_fit, X_test, y_test, k, method, verbose=False):
    if method == "pca":
        mean = X_fit.mean(axis=0)
        X_fit_c = X_fit - mean
        X_test_c = X_test - mean
        L = X_fit_c
        rpca_info = None
    elif method == "rpca":
        X_fit_c = X_fit
        X_test_c = X_test
        L, S, rpca_info = inexact_alm_rpca_verbose(X_fit_c)
    else:
        raise ValueError(method)

    P = get_projection_matrix(L, k)
    errors = reconstruction_error(X_test_c, P)

    n_flag = int(np.ceil(0.05 * len(errors)))
    flagged_idx = np.argsort(errors)[-n_flag:]
    y_pred = np.zeros(len(errors), dtype=int)
    y_pred[flagged_idx] = 1

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", pos_label=1, zero_division=0
    )

    if verbose:
        normal_err = errors[y_test == 0]
        abnormal_err = errors[y_test == 1]
        print(f"    error stats -> normal:   mean={normal_err.mean():.4f}  "
              f"median={np.median(normal_err):.4f}  max={normal_err.max():.4f}")
        print(f"    error stats -> abnormal: mean={abnormal_err.mean():.4f}  "
              f"median={np.median(abnormal_err):.4f}  max={abnormal_err.max():.4f}")
        if rpca_info:
            print(f"    RPCA converged: {rpca_info['converged']} "
                  f"in {rpca_info['iterations']} iterations, "
                  f"final rel. error: {rpca_info['final_error']:.6f}, "
                  f"rank(L): {rpca_info['rank_L']}, "
                  f"nonzero fraction of S: {rpca_info['sparsity_S']:.4f}")

    return precision, recall, f1


def inexact_alm_rpca_verbose(M, lam=None, tol=1e-7, max_iter=500):
    """Same as inexact_alm_rpca but also returns convergence diagnostics."""
    n, d = M.shape
    if lam is None:
        lam = 1 / np.sqrt(max(n, d))

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

    converged = False
    final_err = None
    for i in range(max_iter):
        U, sv, Vt = np.linalg.svd(M - S + Y / mu, full_matrices=False)
        sv_thresh = np.maximum(sv - 1 / mu, 0)
        L = U @ np.diag(sv_thresh) @ Vt

        temp = M - L + Y / mu
        S = np.sign(temp) * np.maximum(np.abs(temp) - lam / mu, 0)

        Z = M - L - S
        Y = Y + mu * Z
        mu = min(mu * rho, mu_bar)

        final_err = np.linalg.norm(Z, "fro") / norm_M
        if final_err < tol:
            converged = True
            break

    rank_L = np.sum(sv_thresh > 1e-6)
    sparsity_S = np.mean(np.abs(S) > 1e-6)

    return L, S, {
        "converged": converged,
        "iterations": i + 1,
        "final_error": final_err,
        "rank_L": rank_L,
        "sparsity_S": sparsity_S,
    }


def sweep_k(X_fit, X_test, y_test, method, k_range):
    print(f"\n  k-sweep for {method.upper()}:")
    best_f1 = -1
    best_k = None
    for k in k_range:
        p, r, f1 = evaluate_at_k(X_fit, X_test, y_test, k, method)
        marker = ""
        if f1 > best_f1:
            best_f1 = f1
            best_k = k
            marker = "  <- best so far"
        print(f"    k={k:2d}: P={p*100:.2f}  R={r*100:.2f}  F1={f1*100:.2f}{marker}")
    print(f"  Best: k={best_k}, F1={best_f1*100:.2f}")
    return best_k, best_f1


def main():
    data = np.load(FEATURES_PATH)
    X_tfidf = data["X_tfidf"]
    X_counts = data["X_counts"]
    y = data["y"]

    fit_idx, test_idx = get_fold1_split(y)
    y_test = y[test_idx]

    for feat_name, X in [("TF-IDF", X_tfidf), ("RAW COUNTS", X_counts)]:
        print(f"\n{'#' * 60}")
        print(f"FEATURE TYPE: {feat_name}")
        print(f"{'#' * 60}")

        X_fit = X[fit_idx]
        X_test = X[test_idx]

        print("\nPCA at paper's k=6 (verbose):")
        evaluate_at_k(X_fit, X_test, y_test, 6, "pca", verbose=True)

        print("\nRPCA at paper's k=9 (verbose):")
        evaluate_at_k(X_fit, X_test, y_test, 9, "rpca", verbose=True)

        sweep_k(X_fit, X_test, y_test, "pca", range(2, 16))
        sweep_k(X_fit, X_test, y_test, "rpca", range(2, 16))


if __name__ == "__main__":
    main()