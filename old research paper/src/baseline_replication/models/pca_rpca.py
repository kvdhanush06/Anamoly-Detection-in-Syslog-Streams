"""
Modular PCA and RPCA models for anomaly detection.
Designed to integrate easily with adaptive thresholding.
"""
import numpy as np

def l2_normalize(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms

class PCAModel:
    def __init__(self, k=6):
        self.k = k
        self.mean_ = None
        self.P_ = None
        
    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        X_c = X - self.mean_
        _, _, Vt = np.linalg.svd(X_c, full_matrices=False)
        self.P_ = Vt[:self.k].T
        
    def predict_errors(self, X):
        X_c = X - self.mean_
        X_proj = X_c @ self.P_ @ self.P_.T
        errors = np.linalg.norm(X_c - X_proj, axis=1)
        return errors

def soft_threshold(X, tau):
    return np.sign(X) * np.maximum(np.abs(X) - tau, 0)

def inexact_alm_rpca(M, lam=None, tol=1e-7, max_iter=500):
    n, d = M.shape
    if lam is None: lam = 1.0 / np.sqrt(max(n, d))
    norm2 = np.linalg.norm(M, 2); normI = np.max(np.abs(M)) / lam
    Y = M / max(norm2, normI); mu = 1.25 / norm2; rho = 1.5
    L = np.zeros_like(M); S = np.zeros_like(M); normM = np.linalg.norm(M, 'fro')
    for _ in range(max_iter):
        U, sv, Vt = np.linalg.svd(M - S + Y/mu, full_matrices=False)
        sv_t = np.maximum(sv - 1/mu, 0)
        L = U @ np.diag(sv_t) @ Vt
        S = soft_threshold(M - L + Y/mu, lam/mu)
        Z = M - L - S; Y += mu * Z; mu = min(mu * rho, mu * 1e7)
        if np.linalg.norm(Z, 'fro') / normM < tol: break
    return L, S, int(np.sum(sv_t > 1e-6))

class RPCAModel:
    def __init__(self, k=9, center=False):
        self.k = k
        self.center = center
        self.mean_ = None
        self.P_ = None
        
    def fit(self, X):
        if self.center:
            self.mean_ = X.mean(axis=0)
            X_fit = X - self.mean_
        else:
            self.mean_ = np.zeros(X.shape[1])
            X_fit = X
            
        L, _, _ = inexact_alm_rpca(X_fit)
        _, _, Vt = np.linalg.svd(L, full_matrices=False)
        self.P_ = Vt[:self.k].T
        
    def predict_errors(self, X):
        X_c = X - self.mean_
        X_proj = X_c @ self.P_ @ self.P_.T
        errors = np.linalg.norm(X_c - X_proj, axis=1)
        return errors
