import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import IsolationForest

class AutoencoderModel:
    def __init__(self, hidden_layer_sizes=(16, 8, 16), alpha=1e-4, random_state=42, max_iter=100):
        self.clf = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation='relu',
            solver='adam',
            alpha=alpha,
            max_iter=max_iter,
            random_state=random_state,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10
        )
        self.mean_ = None
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha

    def fit(self, X):
        # Mean centering fit purely on training normal data
        self.mean_ = X.mean(axis=0)
        X_c = X - self.mean_
        self.clf.fit(X_c, X_c)

    def predict_errors(self, X):
        X_c = X - self.mean_
        X_rec = self.clf.predict(X_c)
        errors = np.linalg.norm(X_c - X_rec, axis=1)
        return errors

class IsolationForestModel:
    def __init__(self, n_estimators=100, random_state=42):
        self.clf = IsolationForest(
            n_estimators=n_estimators,
            contamination='auto',
            random_state=random_state,
            n_jobs=-1
        )
        self.mean_ = None

    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        X_c = X - self.mean_
        self.clf.fit(X_c)

    def predict_errors(self, X):
        X_c = X - self.mean_
        # Decision function is positive for inliers, negative for outliers
        # Negate so higher = more anomalous
        scores = -self.clf.decision_function(X_c)
        return scores

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

