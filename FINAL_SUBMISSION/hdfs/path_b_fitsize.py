"""
Path B: Varying fit size for RPCA with mean-centering.

Config: TF-IDF + RPCA k=9 + mean-centering.
Varying fit size: 5k, 10k, 15k, 20k, 25k.

Run from project root: python src/path_b_fitsize.py
"""
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

FEATURES_PATH = "data/HDFS_v1/preprocessed/features.npz"
N_NORMAL_SAMPLE = 25_000
N_ABNORMAL_SAMPLE = 1_000

def soft_threshold(X, tau):
    return np.sign(X) * np.maximum(np.abs(X) - tau, 0)

def inexact_alm_rpca(M, lam=None, tol=1e-7, max_iter=500):
    n, d = M.shape
    if lam is None: lam = 1.0 / np.sqrt(max(n, d))
    norm2 = np.linalg.norm(M, 2); normI = np.max(np.abs(M))/lam
    Y = M / max(norm2, normI); mu = 1.25/norm2; rho = 1.5
    L = np.zeros_like(M); S = np.zeros_like(M); normM = np.linalg.norm(M, 'fro')
    for _ in range(max_iter):
        U, sv, Vt = np.linalg.svd(M - S + Y/mu, full_matrices=False)
        sv_t = np.maximum(sv - 1/mu, 0)
        L = U @ np.diag(sv_t) @ Vt
        S = soft_threshold(M - L + Y/mu, lam/mu)
        Z = M - L - S; Y += mu*Z; mu = min(mu*rho, mu*1e7)
        if np.linalg.norm(Z, 'fro') / normM < tol: break
    return L, S, int(np.sum(sv_t > 1e-6))

def main():
    data = np.load(FEATURES_PATH)
    X_tf = data["X_tfidf"]
    y = data["y"]
    
    rng = np.random.default_rng(42)
    sn = rng.choice(np.where(y==0)[0], size=N_NORMAL_SAMPLE, replace=False)
    sa = rng.choice(np.where(y==1)[0], size=N_ABNORMAL_SAMPLE, replace=False)
    rng.shuffle(sn)
    
    # We will use the standard fold-1 test set for comparability
    test_idx = np.concatenate([sn[5000:], sa])
    y_test = y[test_idx]
    Xt_tf = X_tf[test_idx]
    
    print(f"Test Set: {len(test_idx)} (20k normal, 1k anomaly)")
    print(f"{'Fit Size':>10}  {'Rank':>5}  {'Prec':>7}  {'Rec':>7}  {'F1':>7}")
    print("-" * 50)
    
    for fit_size in [5000, 10000, 15000, 20000, 25000]:
        fit_idx = sn[:fit_size]
        Xf_tf = X_tf[fit_idx]
        
        mean_tf = Xf_tf.mean(axis=0)
        L, S, rk = inexact_alm_rpca(Xf_tf - mean_tf)
        
        _, _, Vt = np.linalg.svd(L, full_matrices=False)
        P = Vt[:9].T
        
        Xt_c = Xt_tf - mean_tf
        et = np.linalg.norm(Xt_c - Xt_c @ P @ P.T, axis=1)
        
        theta = np.percentile(et, 95)
        y_pred = (et > theta).astype(int)
        
        p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', pos_label=1, zero_division=0)
        print(f"{fit_size:>10}  {rk:>5d}  {p*100:>7.2f}  {r*100:>7.2f}  {f1*100:>7.2f}")

if __name__ == "__main__":
    main()
