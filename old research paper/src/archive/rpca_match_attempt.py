"""
Targeted RPCA gap-closing: two specific untested combinations.

Step 1 — raw_l2 + RPCA + mean-centering (symmetric with PCA treatment)
         Never tested before: v3 used raw_l2+RPCA without centering → F1=30%

Step 2 — TF-IDF + RPCA with larger fitting set (all 25k normal, no fold split)
         Paper fits on 5000 per fold; maybe authors used full 25k for RPCA

Also re-runs paper-exact baselines for comparison.

Run from project root: python src/rpca_match_attempt.py
"""

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

FEATURES_PATH    = "data/HDFS_v1/preprocessed/features.npz"
N_NORMAL_SAMPLE  = 25_000
N_ABNORMAL_SAMPLE = 1_000
N_FOLDS          = 5
FOLD_SIZE        = N_NORMAL_SAMPLE // N_FOLDS
RANDOM_SEED      = 42


# ---------------------------------------------------------------------------
# Core routines
# ---------------------------------------------------------------------------

def soft_threshold(X, tau):
    return np.sign(X) * np.maximum(np.abs(X) - tau, 0)


def inexact_alm_rpca(M, lam=None, tol=1e-7, max_iter=500):
    n, d = M.shape
    if lam is None:
        lam = 1.0 / np.sqrt(max(n, d))
    norm_two = np.linalg.norm(M, 2)
    norm_inf = np.max(np.abs(M)) / lam
    Y  = M / max(norm_two, norm_inf)
    mu = 1.25 / norm_two
    mu_bar = mu * 1e7; rho = 1.5
    L = np.zeros_like(M); S = np.zeros_like(M)
    norm_M = np.linalg.norm(M, "fro"); sv_thresh = np.array([])
    for _ in range(max_iter):
        U, sv, Vt = np.linalg.svd(M - S + Y / mu, full_matrices=False)
        sv_thresh  = np.maximum(sv - 1.0 / mu, 0.0)
        L = U @ np.diag(sv_thresh) @ Vt
        temp = M - L + Y / mu
        S    = soft_threshold(temp, lam / mu)
        Z = M - L - S; Y += mu * Z; mu = min(mu * rho, mu_bar)
        if np.linalg.norm(Z, "fro") / norm_M < tol:
            break
    rank_L = int(np.sum(sv_thresh > 1e-6))
    return L, S, rank_L


def l2_normalize(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


def recon_errors(X, P):
    return np.linalg.norm(X - X @ P @ P.T, axis=1)


def top_projection(M, k):
    _, _, Vt = np.linalg.svd(M, full_matrices=False)
    return Vt[:min(k, Vt.shape[0])].T


def classify_and_score(errors, y_test, pctile=95):
    theta = np.percentile(errors, pctile)
    y_pred = (errors > theta).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", pos_label=1, zero_division=0)
    return p * 100, r * 100, f1 * 100, y_pred.sum()


def record(agg, key, p, r, f1):
    if key not in agg:
        agg[key] = {"p": [], "r": [], "f1": []}
    agg[key]["p"].append(p); agg[key]["r"].append(r); agg[key]["f1"].append(f1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    data     = np.load(FEATURES_PATH)
    X_tfidf  = data["X_tfidf"]
    X_raw    = data["X_counts"].astype(np.float32)
    X_raw_l2 = l2_normalize(X_raw)
    y        = data["y"]

    rng = np.random.default_rng(RANDOM_SEED)
    normal_idx   = np.where(y == 0)[0]
    abnormal_idx = np.where(y == 1)[0]
    sampled_normal   = rng.choice(normal_idx,   size=N_NORMAL_SAMPLE,   replace=False)
    sampled_abnormal = rng.choice(abnormal_idx, size=N_ABNORMAL_SAMPLE, replace=False)
    rng.shuffle(sampled_normal)

    agg = {}

    for fold in range(N_FOLDS):
        start = fold * FOLD_SIZE; end = start + FOLD_SIZE
        fit_idx  = sampled_normal[start:end]
        test_idx = np.concatenate([
            np.concatenate([sampled_normal[:start], sampled_normal[end:]]),
            sampled_abnormal
        ])
        y_test = y[test_idx]

        Xf_tfidf  = X_tfidf[fit_idx];   Xt_tfidf  = X_tfidf[test_idx]
        Xf_raw_l2 = X_raw_l2[fit_idx];  Xt_raw_l2 = X_raw_l2[test_idx]

        print(f"\n{'='*68}")
        print(f"FOLD {fold+1}/{N_FOLDS}  fit={len(fit_idx):,}  test={len(test_idx):,}")
        print(f"{'='*68}")

        # ── Baseline: paper-exact TF-IDF + RPCA k=9 (no centering) ──────
        L, S, rk = inexact_alm_rpca(Xf_tfidf)
        P = top_projection(L, 9)
        et = recon_errors(Xt_tfidf, P)
        p, r, f1, nf = classify_and_score(et, y_test)
        record(agg, f"[BASE]  TF-IDF + RPCA k=9 no-center  rank(L)={rk}", p, r, f1)
        print(f"  [BASE]  TF-IDF+RPCA k=9 no-center  rank={rk:2d}  flagged={nf}  P={p:.2f}  R={r:.2f}  F1={f1:.2f}")

        # ── STEP 1A: raw_l2 + RPCA k=9, NO centering (was in v3) ────────
        L, S, rk = inexact_alm_rpca(Xf_raw_l2)
        P = top_projection(L, 9)
        et = recon_errors(Xt_raw_l2, P)
        p, r, f1, nf = classify_and_score(et, y_test)
        record(agg, f"[1A]    raw_l2 + RPCA k=9  no-center  rank(L)={rk}", p, r, f1)
        print(f"  [1A]    raw_l2+RPCA k=9 no-center   rank={rk:2d}  flagged={nf}  P={p:.2f}  R={r:.2f}  F1={f1:.2f}")

        # ── STEP 1B: raw_l2 + RPCA k=9, WITH mean-centering ─────────────
        mean_l2   = Xf_raw_l2.mean(axis=0)
        Xfc_l2    = Xf_raw_l2 - mean_l2
        Xtc_l2    = Xt_raw_l2 - mean_l2
        L, S, rk  = inexact_alm_rpca(Xfc_l2)
        P = top_projection(L, 9)
        et = recon_errors(Xtc_l2, P)
        p, r, f1, nf = classify_and_score(et, y_test)
        record(agg, f"[1B]    raw_l2 + RPCA k=9  CENTERED   rank(L)={rk}", p, r, f1)
        print(f"  [1B]    raw_l2+RPCA k=9 CENTERED    rank={rk:2d}  flagged={nf}  P={p:.2f}  R={r:.2f}  F1={f1:.2f}")

        # ── STEP 1C: TF-IDF + RPCA k=9, WITH mean-centering ─────────────
        mean_tf   = Xf_tfidf.mean(axis=0)
        Xfc_tf    = Xf_tfidf - mean_tf
        Xtc_tf    = Xt_tfidf - mean_tf
        L, S, rk  = inexact_alm_rpca(Xfc_tf)
        P = top_projection(L, 9)
        et = recon_errors(Xtc_tf, P)
        p, r, f1, nf = classify_and_score(et, y_test)
        record(agg, f"[1C]    TF-IDF + RPCA k=9  CENTERED   rank(L)={rk}", p, r, f1)
        print(f"  [1C]    TF-IDF+RPCA k=9 CENTERED    rank={rk:2d}  flagged={nf}  P={p:.2f}  R={r:.2f}  F1={f1:.2f}")

        # ── STEP 1D: raw_l2 + RPCA k=2 (true rank), no centering ────────
        L, S, rk  = inexact_alm_rpca(Xf_raw_l2)
        k_ada = max(rk, 1)
        P = top_projection(L, k_ada)
        et = recon_errors(Xt_raw_l2, P)
        p, r, f1, nf = classify_and_score(et, y_test)
        record(agg, f"[1D]    raw_l2 + RPCA k=rank({rk}) adaptive", p, r, f1)
        print(f"  [1D]    raw_l2+RPCA k={k_ada}(rank)             rank={rk:2d}  flagged={nf}  P={p:.2f}  R={r:.2f}  F1={f1:.2f}")

        # ── STEP 2A: TF-IDF + RPCA, fit on ALL 25k normal (no folds) ───
        # fit set = full 25k, test set = all 1k abnormal + 20k normal (same test)
        Xf_all_tf = X_tfidf[sampled_normal]      # all 25k normals
        L, S, rk  = inexact_alm_rpca(Xf_all_tf)
        P = top_projection(L, 9)
        et = recon_errors(Xt_tfidf, P)
        p, r, f1, nf = classify_and_score(et, y_test)
        record(agg, f"[2A]    TF-IDF + RPCA k=9  fit=25k    rank(L)={rk}", p, r, f1)
        print(f"  [2A]    TF-IDF+RPCA k=9 fit=25k     rank={rk:2d}  flagged={nf}  P={p:.2f}  R={r:.2f}  F1={f1:.2f}")

        # ── STEP 2B: raw_l2 + RPCA, fit on ALL 25k normal ───────────────
        Xf_all_l2 = X_raw_l2[sampled_normal]
        L, S, rk  = inexact_alm_rpca(Xf_all_l2)
        P = top_projection(L, 9)
        et = recon_errors(Xt_raw_l2, P)
        p, r, f1, nf = classify_and_score(et, y_test)
        record(agg, f"[2B]    raw_l2 + RPCA k=9  fit=25k    rank(L)={rk}", p, r, f1)
        print(f"  [2B]    raw_l2+RPCA k=9 fit=25k     rank={rk:2d}  flagged={nf}  P={p:.2f}  R={r:.2f}  F1={f1:.2f}")

        # ── PCA reference ────────────────────────────────────────────────
        mean_ref = Xf_raw_l2.mean(axis=0)
        _, _, Vt = np.linalg.svd(Xf_raw_l2 - mean_ref, full_matrices=False)
        P_pca = Vt[:6].T
        et_pca = recon_errors(Xt_raw_l2 - mean_ref, P_pca)
        p, r, f1, nf = classify_and_score(et_pca, y_test)
        record(agg, "[REF]   raw_l2 + PCA  k=6  (our best)", p, r, f1)
        print(f"  [REF]   raw_l2+PCA  k=6                rank=--  flagged={nf}  P={p:.2f}  R={r:.2f}  F1={f1:.2f}")

    # Summary
    print(f"\n\n{'='*80}")
    print("FINAL 5-FOLD AVERAGES")
    print(f"{'='*80}")
    print(f"  {'Method':<52s}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}")
    print("  " + "─"*72)

    best_f1 = -1; best_key = None
    for key, vals in agg.items():
        pm  = np.mean(vals["p"])
        rm  = np.mean(vals["r"])
        f1m = np.mean(vals["f1"])
        mark = ""
        if f1m > best_f1: best_f1, best_key, mark = f1m, key, "  ◀ BEST"
        print(f"  {key:<52s}  {pm:>6.2f}  {rm:>6.2f}  {f1m:>6.2f}{mark}")

    print("  " + "─"*72)
    print(f"  {'📄 Paper RPCA target':<52s}  {'89.88':>6}  {'91.44':>6}  {'90.55':>6}")
    print(f"  {'📄 Paper PCA  target':<52s}  {'92.90':>6}  {'96.45':>6}  {'94.64':>6}")
    print(f"\n  Best: {best_key}  →  F1={best_f1:.2f}")


if __name__ == "__main__":
    main()
