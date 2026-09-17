import numpy as np

class MaxEnsembleModel:
    """
    Ensemble model that takes prediction scores from two independent models,
    normalizes them (using Z-score normalization based on the fitting set), 
    and combines them by taking the maximum of the two standardized scores.
    """
    def __init__(self, model_a, model_b):
        self.model_a = model_a
        self.model_b = model_b
        
        # Stored statistics for Z-score normalization
        self.mu_a = 0.0
        self.std_a = 1.0
        self.mu_b = 0.0
        self.std_b = 1.0

    def fit(self, X_a, X_b):
        """
        Fits both models independently.
        X_a: Features for model A (e.g., Count-based)
        X_b: Features for model B (e.g., Word2Vec Semantic)
        """
        # Fit models
        self.model_a.fit(X_a)
        self.model_b.fit(X_b)
        
        # Calculate baseline training error scores
        err_a = self.model_a.predict_errors(X_a)
        err_b = self.model_b.predict_errors(X_b)
        
        # Calculate scaling statistics
        self.mu_a = np.mean(err_a)
        self.std_a = np.std(err_a) + 1e-10
        
        self.mu_b = np.mean(err_b)
        self.std_b = np.std(err_b) + 1e-10

    def predict_errors(self, X_a, X_b):
        """
        Predicts combined normalized anomaly scores.
        """
        err_a = self.model_a.predict_errors(X_a)
        err_b = self.model_b.predict_errors(X_b)
        
        # Z-score normalization
        z_a = (err_a - self.mu_a) / self.std_a
        z_b = (err_b - self.mu_b) / self.std_b
        
        # Combine using Mean function
        # A sequence is anomalous if both models agree on average, smoothing out false positives
        combined_scores = np.mean([z_a, z_b], axis=0)
        return combined_scores
