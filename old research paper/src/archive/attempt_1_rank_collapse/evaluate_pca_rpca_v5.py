"""
PCA vs Robust PCA — v5: All improvements stacked.

Improvements over v3/v4:
  [1] RPCA dynamic k: pick k via 95% SVD-energy threshold on L, not hardcoded 9.
  [2] RPCA lambda sweep: α ∈ {0.5, 0.75, 1.0, 1.5, 2.0}, λ = α/√max(n,d).
  [3] New features: bigram transitions + time-interval stats (from features_v2.npz).
  [4] POT/EVT adaptive threshold: fit Generalised Pareto Distribution to tail
      of fitting-set reconstruction errors; derive θ at 5% false-alarm rate.

All results compared against:
  - v3 best (PCA, raw_l2, k=6, count-threshold) → F1=94.56
  - Paper targets: PCA=94.64, RPCA=90.55

Run from project root with venv activated:
    python src/evaluate_pca_rpca_v5.py
"""

import numpy as np
from scipy.stats import genpareto
from sklearn.metrics import precision_recall_fscore_support

FEATURES_V1_PATH = "data/HDFS_v1/preprocessed/features.npz"
FEATURES_V2_PATH = "data/HDFS_v1/preprocessed/features_v2.npz"

N_NORMAL_SAMPLE   = 25_000
N_ABNORMAL_SAMPLE = 1_000
N_FOLDS           = 5
FOLD_SIZE         = N_NORMAL_SAMPLE // N_FOLDS   # 5 000
PCA_K             = 6
RPCA_K_TARGET     = 9      # upper cap; dynamic k may be lower
RANDOM_SEED       = 42
ANOMALY_FRACTION  = 0.05

ALPHA_VALUES = [0.5, 0.75, 1.0, 1.5, 2.0]   # lambda scales to sweep


# ---------------------------------------------------------------------------
# RPCA — inexact ALM  (Lin, Chen & Ma 2010)
# ---------------------------------------------------------------------------

def soft_threshold(X, tau):
    return np.sign(X) * np.maximum(np.abs(X) - tau, 0)


def inexact_alm_rpca(M, lam=None, tol=1e-7, max_iter=500):
    """Returns (L, S, rank_L, sv_thresh) where sv_thresh are the final
    thresholded singular values (used for energy-based k selection)."""
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

        Z  = M - L - S
        Y  = Y + mu * Z
        mu = min(mu * rho, mu_bar)

        if np.linalg.norm(Z, "fro") / norm_M < tol:
            break

    rank_L = int(np.sum(sv_thresh > 1e-6))
    return L, S, rank_L, sv_thresh


# ---------------------------------------------------------------------------
# Projection helpers
# ---------------------------------------------------------------------------

def l2_normalize(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


def get_projection_matrix(M, k):
    """Top-k right singular vectors of M as a (d × k) projection matrix.
    Equivalent to top-k left singular vectors of M^T (as the paper specifies)."""
    _, sv_full, Vt = np.linalg.svd(M, full_matrices=False)
    k = min(k, Vt.shape[0])
    return Vt[:k].T, sv_full


def dynamic_k(sv_thresh, energy_threshold=0.95, k_max=RPCA_K_TARGET):
    """
    Choose smallest k such that cumulative energy of L ≥ energy_threshold.
    sv_thresh: thresholded singular values from RPCA (already >= 0).
    Falls back to k=1 if rank(L)=0.
    """
    sv = sv_thresh[sv_thresh > 1e-6]
    if len(sv) == 0:
        return 1
    total = (sv ** 2).sum()
    cumulative = np.cumsum(sv ** 2)
    k_energy = int(np.searchsorted(cumulative / total, energy_threshold)) + 1
    k_energy = max(1, min(k_energy, len(sv)))
    return min(k_energy, k_max)


def recon_errors(X, P):
    return np.linalg.norm(X - X @ P @ P.T, axis=1)


# ---------------------------------------------------------------------------
# Thresholding strategies
# ---------------------------------------------------------------------------

def threshold_count(err_test, y_test, fraction=ANOMALY_FRACTION):
    """Top-fraction% by rank (v3 baseline)."""
    n_flag = int(np.ceil(fraction * len(err_test)))
    flags  = np.argsort(err_test)[-n_flag:]
    y_pred = np.zeros(len(err_test), dtype=int)
    y_pred[flags] = 1
    p, r, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", pos_label=1, zero_division=0)
    return p, r, f1


def threshold_pot(err_fit, err_test, y_test,
                  tail_quantile=0.90, target_fpr=ANOMALY_FRACTION):
    """
    Peaks-Over-Threshold with GPD fit.
    1. Pre-threshold u = tail_quantile of fitting errors.
    2. Fit GPD to exceedances: err_fit[err_fit > u] - u.
    3. Compute θ for target_fpr false-alarm rate.
    4. Classify err_test > θ.
    Falls back to count-threshold if GPD fit fails.
    """
    n_fit = len(err_fit)
    u     = np.percentile(err_fit, tail_quantile * 100)
    exceedances = err_fit[err_fit > u] - u
    n_u   = len(exceedances)

    if n_u < 10:
        # not enough tail data — fall back
        return threshold_count(err_test, y_test)

    try:
        xi, loc, beta = genpareto.fit(exceedances, floc=0)
        # GPD quantile formula (Pickands-Balkema-de Haan)
        # θ = u + (β/ξ) * ((n_u / (n * p))^ξ - 1)   [ξ ≠ 0]
        # θ = u + β * log(n_u / (n * p))              [ξ = 0]
        ratio = n_u / (n_fit * target_fpr)
        if abs(xi) > 1e-8:
            theta = u + (beta / xi) * (ratio ** xi - 1.0)
        else:
            theta = u + beta * np.log(ratio)

        # safety: if theta is nonsensical, fall back
        if not np.isfinite(theta) or theta <= 0:
            return threshold_count(err_test, y_test)

        y_pred = (err_test > theta).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="binary", pos_label=1, zero_division=0)
        return p, r, f1

    except Exception:
        return threshold_count(err_test, y_test)


# ---------------------------------------------------------------------------
# Fold evaluation
# ---------------------------------------------------------------------------

def evaluate_fold_pca(Xf, Xt, y_test, k=PCA_K):
    """Standard PCA: mean-center, SVD, top-k projection."""
    mean  = Xf.mean(axis=0)
    Xf_c  = Xf - mean
    Xt_c  = Xt - mean
    P, sv = get_projection_matrix(Xf_c, k)
    ef    = recon_errors(Xf_c, P)
    et    = recon_errors(Xt_c, P)

    p_cnt, r_cnt, f1_cnt = threshold_count(et, y_test)
    p_pot, r_pot, f1_pot = threshold_pot(ef, et, y_test)
    return {
        "pca_count": (p_cnt, r_cnt, f1_cnt),
        "pca_pot":   (p_pot, r_pot, f1_pot),
    }


def evaluate_fold_rpca(Xf, Xt, y_test, lam):
    """RPCA with given lambda; uses dynamic k and both threshold strategies."""
    L, S, rank_L, sv_thresh = inexact_alm_rpca(Xf, lam=lam)
    k = dynamic_k(sv_thresh)

    P, _ = get_projection_matrix(L, k)
    ef   = recon_errors(Xf, P)
    et   = recon_errors(Xt, P)

    p_cnt, r_cnt, f1_cnt = threshold_count(et, y_test)
    p_pot, r_pot, f1_pot = threshold_pot(ef, et, y_test)
    return {
        f"rpca_a{lam:.4f}_k{k}_count": (p_cnt, r_cnt, f1_cnt),
        f"rpca_a{lam:.4f}_k{k}_pot":   (p_pot, r_pot, f1_pot),
    }, rank_L, k


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_cv(label, X_all, y, normal_idx, abnormal_idx,
           sampled_normal, sampled_abnormal, lam_values):
    """
    Run 5-fold CV for a given feature matrix X_all.
    Returns agg dict: {method_label: {p:[], r:[], f1:[]}}
    """
    agg = {}

    def record(agg, key, p, r, f1):
        if key not in agg:
            agg[key] = {"p": [], "r": [], "f1": []}
        agg[key]["p"].append(p)
        agg[key]["r"].append(r)
        agg[key]["f1"].append(f1)

    for fold in range(N_FOLDS):
        start = fold * FOLD_SIZE
        end   = start + FOLD_SIZE

        fit_idx         = sampled_normal[start:end]
        test_normal_idx = np.concatenate([sampled_normal[:start],
                                          sampled_normal[end:]])
        test_idx        = np.concatenate([test_normal_idx, sampled_abnormal])

        Xf = X_all[fit_idx]
        Xt = X_all[test_idx]
        y_test = y[test_idx]

        # PCA
        pca_out = evaluate_fold_pca(Xf, Xt, y_test, k=PCA_K)
        for key, (p, r, f1) in pca_out.items():
            record(agg, key, p, r, f1)

        # RPCA × lambda values
        n, d = Xf.shape
        for alpha in lam_values:
            lam = alpha / np.sqrt(max(n, d))
            rpca_out, rank_L, k_used = evaluate_fold_rpca(Xf, Xt, y_test, lam)
            for key, (p, r, f1) in rpca_out.items():
                # embed alpha and feature-set label
                full_key = f"[{label}] α={alpha:.2f} k={k_used} rank={rank_L} | {key.split('_count')[0].split('_pot')[0]}_{'count' if 'count' in key else 'pot'}"
                record(agg, full_key, p, r, f1)

    return agg


def print_agg(agg, title, reference=None):
    print(f"\n{'=' * 88}")
    print(f"  {title}")
    print(f"{'=' * 88}")
    print(f"  {'Method':<60s}  {'P':>6s}  {'R':>6s}  {'F1':>6s}")
    print("  " + "-" * 80)

    best_f1 = -1
    best_key = None
    rows = []
    for key, vals in sorted(agg.items()):
        pm  = np.mean(vals["p"])  * 100
        rm  = np.mean(vals["r"])  * 100
        f1m = np.mean(vals["f1"]) * 100
        rows.append((key, pm, rm, f1m))
        if f1m > best_f1:
            best_f1  = f1m
            best_key = key

    for key, pm, rm, f1m in rows:
        mark = "  ◀ BEST" if key == best_key else ""
        print(f"  {key:<60s}  {pm:>6.2f}  {rm:>6.2f}  {f1m:>6.2f}{mark}")

    if reference:
        print("  " + "-" * 80)
        for name, (p_ref, r_ref, f1_ref) in reference.items():
            print(f"  {'📄 ' + name:<60s}  {p_ref:>6.2f}  {r_ref:>6.2f}  {f1_ref:>6.2f}")

    return best_f1, best_key


def main():
    # Load features
    d1 = np.load(FEATURES_V1_PATH)
    d2 = np.load(FEATURES_V2_PATH)

    y         = d1["y"]
    X_raw_l2  = l2_normalize(d1["X_counts"].astype(np.float32))
    X_bigram  = d2["X_bigram_l2"]
    X_combined = d2["X_combined_l2"]

    print(f"Unigram L2   : {X_raw_l2.shape}")
    print(f"Bigram L2    : {X_bigram.shape}")
    print(f"Combined L2  : {X_combined.shape}")
    print(f"Labels       : {y.shape}  anomaly={y.mean()*100:.2f}%")

    rng = np.random.default_rng(RANDOM_SEED)
    normal_idx   = np.where(y == 0)[0]
    abnormal_idx = np.where(y == 1)[0]

    sampled_normal   = rng.choice(normal_idx,   size=N_NORMAL_SAMPLE,   replace=False)
    sampled_abnormal = rng.choice(abnormal_idx, size=N_ABNORMAL_SAMPLE, replace=False)
    rng.shuffle(sampled_normal)

    reference = {
        "Paper PCA  target":  (92.90, 96.45, 94.64),
        "Paper RPCA target":  (89.88, 91.44, 90.55),
        "Our v3 PCA best":    (92.30, 96.92, 94.56),
    }

    # ---- Feature set 1: unigram (baseline comparison) ----
    agg1 = run_cv("unigram", X_raw_l2, y,
                  normal_idx, abnormal_idx,
                  sampled_normal, sampled_abnormal,
                  lam_values=ALPHA_VALUES)
    b1, bk1 = print_agg(agg1, "FEATURE SET 1: Unigram L2 (raw counts, L2-norm)", reference)

    # ---- Feature set 2: bigram ----
    agg2 = run_cv("bigram", X_bigram, y,
                  normal_idx, abnormal_idx,
                  sampled_normal, sampled_abnormal,
                  lam_values=ALPHA_VALUES)
    b2, bk2 = print_agg(agg2, "FEATURE SET 2: Bigram L2 (transition counts, L2-norm)", reference)

    # ---- Feature set 3: combined ----
    agg3 = run_cv("combined", X_combined, y,
                  normal_idx, abnormal_idx,
                  sampled_normal, sampled_abnormal,
                  lam_values=ALPHA_VALUES)
    b3, bk3 = print_agg(agg3, "FEATURE SET 3: Combined L2 (unigram+bigram+time)", reference)

    # ---- Grand summary ----
    print(f"\n\n{'=' * 88}")
    print("GRAND SUMMARY — Best F1 per feature set")
    print(f"{'=' * 88}")
    print(f"  {'Feature set':<20s}  {'Best F1':>8s}  Config")
    print("  " + "-" * 80)
    for label, bf, bk in [("Unigram L2", b1, bk1),
                           ("Bigram L2", b2, bk2),
                           ("Combined L2", b3, bk3)]:
        print(f"  {label:<20s}  {bf:>8.2f}  {bk}")

    print("  " + "-" * 80)
    print(f"  {'Paper PCA  target':<20s}  {'94.64':>8s}")
    print(f"  {'Paper RPCA target':<20s}  {'90.55':>8s}")
    print(f"  {'Our v3 PCA best':<20s}  {'94.56':>8s}")


if __name__ == "__main__":
    main()
