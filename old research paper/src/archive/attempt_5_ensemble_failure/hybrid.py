"""
Hybrid Two-Stage Pipeline Model
Stage 1: Robust filtering using RPCA to purify normal data.
Stage 2: Non-linear detection using Isolation Forest trained on the purified data.
"""
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neural_network import MLPRegressor
from .pca_rpca import inexact_alm_rpca

class HybridRPCAModel:
    def __init__(self, layer_2='isolation_forest', center=True, contamination='auto', random_state=42, nu=0.05, gamma='scale', hidden_layer_sizes=(16, 8, 16)):
        self.center = center
        self.mean_ = None
        self.layer_2 = layer_2
        
        # Layer 2 options
        if self.layer_2 == 'isolation_forest':
            self.clf = IsolationForest(
                contamination=contamination, 
                random_state=random_state,
                n_estimators=100
            )
        elif self.layer_2 == 'ocsvm':
            self.clf = OneClassSVM(nu=nu, kernel='rbf', gamma=gamma)
        elif self.layer_2 == 'autoencoder':
            # Use a simple MLP to act as an Autoencoder
            self.clf = MLPRegressor(
                hidden_layer_sizes=hidden_layer_sizes,
                activation='relu',
                solver='adam',
                max_iter=500,
                random_state=random_state,
                early_stopping=True
            )
        else:
            raise ValueError("Unsupported layer_2 option. Choose 'isolation_forest', 'ocsvm', or 'autoencoder'.")
            
    def fit(self, X):
        """
        Fits the model by first mathematically cleaning X using RPCA,
        then training the Layer 2 detector on the cleaned L matrix.
        """
        if self.center:
            self.mean_ = X.mean(axis=0)
            X_fit = X - self.mean_
        else:
            self.mean_ = np.zeros(X.shape[1])
            X_fit = X
            
        # Stage 1: Robust Filter (RPCA)
        # We discard S (the gross anomalies) and keep L (purified normal subspace)
        L, S, _ = inexact_alm_rpca(X_fit)
        
        # Stage 2: Train non-linear detector on purified normal data
        if self.layer_2 == 'autoencoder':
            # Autoencoder learns to reconstruct the input L
            self.clf.fit(L, L)
        else:
            self.clf.fit(L)
        
    def predict_errors(self, X):
        """
        Predicts anomaly scores.
        """
        X_c = X - self.mean_
        
        if self.layer_2 == 'autoencoder':
            # Score is the reconstruction error
            X_reconstructed = self.clf.predict(X_c)
            scores = np.linalg.norm(X_c - X_reconstructed, axis=1)
        else:
            # decision_function returns negative values for outliers in sklearn.
            # We negate it so that higher scores = more anomalous.
            scores = -self.clf.decision_function(X_c)
            
        return scores
