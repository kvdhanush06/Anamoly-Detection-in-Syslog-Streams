"""
RPCA gap-closing — Options A, B, C  (v4)

Runs in cascade:
  Option A: Contaminated fitting — add a fraction of anomalies to the RPCA
            fitting set (PCA still uses pure-normal fit as per the paper).
            Tries contamination ratios 1 / 3 / 5 / 10 / 20 %.

  Option B: Lambda tuning — keep the fitting set pure-normal but reduce
            lambda (nuclear-norm penalty), forcing more singular values to
            survive SVD-thresholding → higher rank(L).
            Tries lam_scale = 1 (paper default) / 0.5 / 0.25 / 0.1 / 0.05.

  Option C: Transductive RPCA — decompose the test matrix itself (M = L+S),
            classify directly from the row-wise L2 norm of S (the sparse
            "anomaly" component).  No training-set fit needed.

All options use:
  - L2 row-normalised raw counts (confirmed correct for PCA in v3)
  - Top-5% count-based threshold (confirmed correct in v2)
  - Same 5-fold CV split as v1/v2/v3

PCA (pure-normal, raw_l2, k=6) reference row is printed for comparison.

Run from project root with venv activated:
    python src/evaluate_pca_rpca_v4.py
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
# RPCA — inexact ALM  (Lin, Chen & Ma 2010)
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
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


def get_projection_matrix(M, k):
    _, _, Vt = np.linalg.svd(M, full_matrices=False)
    k = min(k, Vt.shape[0])
    return Vt[:k].T


def recon_errors(X, P):
    return np.linalg.norm(X - X @ P @ P.T, axis=1)


def classify_count(err_test, y_test, fraction=ANOMALY_FRACTION):
    n_flag = int(np.ceil(fraction * len(err_test)))
    flags  = np.argsort(err_test)[-n_flag:]
    y_pred = np.zeros(len(err_test), dtype=int)
    y_pred[flags] = 1
    p, r, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", pos_label=1, zero_division=0
    )
    return p, r, f1


# ---------------------------------------------------------------------------
# Option A helpers
# ---------------------------------------------------------------------------

def rpca_option_a(Xf_l2, Xt_l2, y_test, contamination_ratio, ab_pool_l2, rng):
    """
    Add floor(len(Xf_l2) * contamination_ratio) anomaly rows to Xf_l2,
    run RPCA, project Xt_l2, classify.
    """
    n_contam = max(1, int(np.floor(len(Xf_l2) * contamination_ratio)))
    n_contam = min(n_contam, len(ab_pool_l2))
    contam_idx = rng.choice(len(ab_pool_l2), size=n_contam, replace=False)
    Xf_contam  = np.vstack([Xf_l2, ab_pool_l2[contam_idx]])

    L, _S, rank_L = inexact_alm_rpca(Xf_contam)
    P   = get_projection_matrix(L, RPCA_K)
    et  = recon_errors(Xt_l2, P)
    p, r, f1 = classify_count(et, y_test)
    return p, r, f1, rank_L, n_contam


# ---------------------------------------------------------------------------
# Option B helpers
# ---------------------------------------------------------------------------

def rpca_option_b(Xf_l2, Xt_l2, y_test, lam_scale):
    """
    Run RPCA with lambda = lam_scale * default_lambda.
    """
    n, d = Xf_l2.shape
    lam  = lam_scale / np.sqrt(max(n, d))
    L, _S, rank_L = inexact_alm_rpca(Xf_l2, lam=lam)
    P   = get_projection_matrix(L, RPCA_K)
    et  = recon_errors(Xt_l2, P)
    p, r, f1 = classify_count(et, y_test)
    return p, r, f1, rank_L


# ---------------------------------------------------------------------------
# Option C helpers
# ---------------------------------------------------------------------------

def rpca_option_c(Xt_l2, y_test, lam_scale=1.0):
    """
    Transductive: decompose the test matrix itself.  Classify by row-wise
    L2 norm of S (the sparse "anomaly" component).
    """
    n, d = Xt_l2.shape
    lam  = lam_scale / np.sqrt(max(n, d))
    _L, S, rank_L = inexact_alm_rpca(Xt_l2, lam=lam)
    s_scores = np.linalg.norm(S, axis=1)
    p, r, f1 = classify_count(s_scores, y_test)
    return p, r, f1, rank_L


def rpca_option_c_proj(Xf_l2, Xt_l2, y_test):
    """
    Hybrid transductive: decompose the test matrix with RPCA, extract the
    low-rank subspace from L, project Xt back through it, then score by
    reconstruction error from that subspace.  Combines the subspace
    estimation quality of transductive RPCA with inductive scoring.
    """
    n, d  = Xt_l2.shape
    lam   = 1.0 / np.sqrt(max(n, d))
    L, _S, rank_L = inexact_alm_rpca(Xt_l2, lam=lam)
    k = max(rank_L, 1)
    P = get_projection_matrix(L, k)
    et = recon_errors(Xt_l2, P)
    p, r, f1 = classify_count(et, y_test)
    return p, r, f1, rank_L


# ---------------------------------------------------------------------------
# Main 5-fold CV
# ---------------------------------------------------------------------------

CONTAM_RATIOS = [0.01, 0.03, 0.05, 0.10, 0.20]
LAM_SCALES    = [1.0, 0.5, 0.25, 0.10, 0.05]


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

    # accumulators: {label: {p:[], r:[], f1:[]}}
    agg_pca = {"p": [], "r": [], "f1": []}

    agg_a   = {cr: {"p": [], "r": [], "f1": [], "rank": []} for cr in CONTAM_RATIOS}
    agg_b   = {ls: {"p": [], "r": [], "f1": [], "rank": []} for ls in LAM_SCALES}
    agg_c   = {"sparse_score":    {"p": [], "r": [], "f1": [], "rank": []},
               "proj_recon":      {"p": [], "r": [], "f1": [], "rank": []}}

    for fold in range(N_FOLDS):
        start = fold * FOLD_SIZE
        end   = start + FOLD_SIZE

        fit_idx         = sampled_normal[start:end]
        test_normal_idx = np.concatenate([sampled_normal[:start], sampled_normal[end:]])
        test_idx        = np.concatenate([test_normal_idx, sampled_abnormal])

        # raw-count matrices
        Xf_raw = X_counts[fit_idx]
        Xt_raw = X_counts[test_idx]
        # anomaly pool (EXCLUDING the 1,000 already in test)
        remain_ab = np.setdiff1d(abnormal_idx, sampled_abnormal)
        Xab_pool  = l2_normalize(X_counts[remain_ab])
        y_test    = y[test_idx]

        # L2-normalised versions
        Xf_l2 = l2_normalize(Xf_raw)
        Xt_l2 = l2_normalize(Xt_raw)

        print(f"\n{'=' * 65}")
        print(f"FOLD {fold+1}/{N_FOLDS}  fit={len(fit_idx):,}  "
              f"test={len(test_idx):,}  ab_pool={len(Xab_pool):,}")
        print(f"{'=' * 65}")

        # --- PCA reference (pure-normal, raw_l2, k=6) ---
        mean  = Xf_l2.mean(axis=0)
        Xf_c  = Xf_l2 - mean
        Xt_c  = Xt_l2 - mean
        P_pca = get_projection_matrix(Xf_c, PCA_K)
        et_pca = recon_errors(Xt_c, P_pca)
        p, r, f1 = classify_count(et_pca, y_test)
        agg_pca["p"].append(p); agg_pca["r"].append(r); agg_pca["f1"].append(f1)
        print(f"  PCA ref (raw_l2, k=6)                   "
              f"P={p*100:6.2f}  R={r*100:6.2f}  F1={f1*100:6.2f}")

        # --- Option A ---
        print(f"\n  [Option A] Contaminated fit:")
        for cr in CONTAM_RATIOS:
            p, r, f1, rank_L, n_c = rpca_option_a(
                Xf_l2, Xt_l2, y_test, cr, Xab_pool, rng)
            agg_a[cr]["p"].append(p); agg_a[cr]["r"].append(r)
            agg_a[cr]["f1"].append(f1); agg_a[cr]["rank"].append(rank_L)
            print(f"    contam={cr*100:4.0f}%  n_added={n_c:4d}  rank(L)={rank_L:2d}  "
                  f"P={p*100:6.2f}  R={r*100:6.2f}  F1={f1*100:6.2f}")

        # --- Option B ---
        print(f"\n  [Option B] Lambda tuning (pure-normal fit, raw_l2):")
        for ls in LAM_SCALES:
            p, r, f1, rank_L = rpca_option_b(Xf_l2, Xt_l2, y_test, ls)
            agg_b[ls]["p"].append(p); agg_b[ls]["r"].append(r)
            agg_b[ls]["f1"].append(f1); agg_b[ls]["rank"].append(rank_L)
            n, d = Xf_l2.shape
            actual_lam = ls / np.sqrt(max(n, d))
            print(f"    lam_scale={ls:.2f}  λ={actual_lam:.5f}  rank(L)={rank_L:2d}  "
                  f"P={p*100:6.2f}  R={r*100:6.2f}  F1={f1*100:6.2f}")

        # --- Option C ---
        print(f"\n  [Option C] Transductive RPCA on test matrix:")
        # C1: sparse-score classifier
        p, r, f1, rank_L = rpca_option_c(Xt_l2, y_test)
        agg_c["sparse_score"]["p"].append(p); agg_c["sparse_score"]["r"].append(r)
        agg_c["sparse_score"]["f1"].append(f1); agg_c["sparse_score"]["rank"].append(rank_L)
        print(f"    sparse-score         rank(L)={rank_L:2d}  "
              f"P={p*100:6.2f}  R={r*100:6.2f}  F1={f1*100:6.2f}")

        # C2: hybrid (subspace from test-L, then projection error)
        p, r, f1, rank_L = rpca_option_c_proj(Xf_l2, Xt_l2, y_test)
        agg_c["proj_recon"]["p"].append(p); agg_c["proj_recon"]["r"].append(r)
        agg_c["proj_recon"]["f1"].append(f1); agg_c["proj_recon"]["rank"].append(rank_L)
        print(f"    proj-recon (test-L)  rank(L)={rank_L:2d}  "
              f"P={p*100:6.2f}  R={r*100:6.2f}  F1={f1*100:6.2f}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(f"\n\n{'=' * 80}")
    print("FINAL RESULTS — averaged over 5 folds")
    print(f"{'=' * 80}")

    pca_f1 = np.mean(agg_pca["f1"]) * 100
    print(f"\n  PCA reference (raw_l2, k=6):"
          f"  P={np.mean(agg_pca['p'])*100:.2f}"
          f"  R={np.mean(agg_pca['r'])*100:.2f}"
          f"  F1={pca_f1:.2f}")
    print(f"  Paper targets:  PCA F1=94.64   RPCA F1=90.55")

    # Option A
    print(f"\n  --- Option A: Contaminated fit ---")
    print(f"  {'Contam':>10s}  {'#added':>8s}  {'rank(L)':>8s}  {'Prec':>8s}  {'Rec':>8s}  {'F1':>8s}")
    best_a_f1, best_a_cr = -1, None
    for cr in CONTAM_RATIOS:
        pm  = np.mean(agg_a[cr]["p"])  * 100
        rm  = np.mean(agg_a[cr]["r"])  * 100
        f1m = np.mean(agg_a[cr]["f1"]) * 100
        rkm = np.mean(agg_a[cr]["rank"])
        n_a = int(np.floor(FOLD_SIZE * cr))
        mark = ""
        if f1m > best_a_f1:
            best_a_f1, best_a_cr = f1m, cr
            mark = " ◀"
        print(f"  {cr*100:>9.0f}%  {n_a:>8d}  {rkm:>8.1f}  {pm:>8.2f}  {rm:>8.2f}  {f1m:>8.2f}{mark}")

    # Option B
    print(f"\n  --- Option B: Lambda tuning (pure-normal fit) ---")
    print(f"  {'lam_scale':>10s}  {'rank(L)':>8s}  {'Prec':>8s}  {'Rec':>8s}  {'F1':>8s}")
    best_b_f1, best_b_ls = -1, None
    for ls in LAM_SCALES:
        pm  = np.mean(agg_b[ls]["p"])  * 100
        rm  = np.mean(agg_b[ls]["r"])  * 100
        f1m = np.mean(agg_b[ls]["f1"]) * 100
        rkm = np.mean(agg_b[ls]["rank"])
        mark = ""
        if f1m > best_b_f1:
            best_b_f1, best_b_ls = f1m, ls
            mark = " ◀"
        print(f"  {ls:>10.2f}  {rkm:>8.1f}  {pm:>8.2f}  {rm:>8.2f}  {f1m:>8.2f}{mark}")

    # Option C
    print(f"\n  --- Option C: Transductive RPCA ---")
    print(f"  {'Variant':<20s}  {'rank(L)':>8s}  {'Prec':>8s}  {'Rec':>8s}  {'F1':>8s}")
    best_c_f1 = -1
    for variant, vals in agg_c.items():
        pm  = np.mean(vals["p"])  * 100
        rm  = np.mean(vals["r"])  * 100
        f1m = np.mean(vals["f1"]) * 100
        rkm = np.mean(vals["rank"])
        mark = ""
        if f1m > best_c_f1:
            best_c_f1 = f1m
            mark = " ◀"
        print(f"  {variant:<20s}  {rkm:>8.1f}  {pm:>8.2f}  {rm:>8.2f}  {f1m:>8.2f}{mark}")

    # Overall best
    overall_best_f1 = max(best_a_f1, best_b_f1, best_c_f1)
    print(f"\n  Overall RPCA best F1 = {overall_best_f1:.2f}  (paper target: 90.55)")
    print(f"  A best: {best_a_f1:.2f} @ contam={best_a_cr*100:.0f}%")
    print(f"  B best: {best_b_f1:.2f} @ lam_scale={best_b_ls:.2f}")
    print(f"  C best: {best_c_f1:.2f}")


if __name__ == "__main__":
    main()
