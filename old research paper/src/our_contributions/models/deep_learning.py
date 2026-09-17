import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPRegressor

class IsolationForestModel:
    def __init__(self, contamination='auto', random_state=42):
        self.clf = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100
        )
        self.mean_ = None

    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        X_c = X - self.mean_
        self.clf.fit(X_c)

    def predict_errors(self, X):
        X_c = X - self.mean_
        # decision_function returns negative for anomalies, positive for inliers
        # We negate it so higher score = more anomalous
        scores = -self.clf.decision_function(X_c)
        return scores

class AutoencoderModel:
    def __init__(self, hidden_layer_sizes=(16, 8, 16), random_state=42):
        self.clf = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation='relu',
            solver='adam',
            max_iter=500,
            random_state=random_state,
            early_stopping=True
        )
        self.mean_ = None

    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        X_c = X - self.mean_
        # Autoencoder learns to reconstruct its input
        self.clf.fit(X_c, X_c)

    def predict_errors(self, X):
        X_c = X - self.mean_
        X_reconstructed = self.clf.predict(X_c)
        # Reconstruction error is the anomaly score
        scores = np.linalg.norm(X_c - X_reconstructed, axis=1)
        return scores
