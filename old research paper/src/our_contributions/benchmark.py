import os
import time
import joblib
import numpy as np
from models.pca_rpca import PCAModel, RPCAModel
from models.deep_learning import AutoencoderModel

def count_parameters_pca(model):
    """PCA parameters are the principal components matrix P (k x d) and mean vector (d)"""
    if hasattr(model, 'P_'):
        k, d = model.P_.shape
        return (k * d) + d
    return 0

def count_parameters_rpca(model):
    """RPCA parameters are the principal components matrix P (k x d) and mean vector (d)"""
    if hasattr(model, 'P_'):
        k, d = model.P_.shape
        return (k * d) + d
    return 0

def count_parameters_autoencoder(model):
    """Autoencoder parameters: weights and biases of MLPRegressor"""
    if hasattr(model.clf, 'coefs_'):
        params = 0
        for coef, intercept in zip(model.clf.coefs_, model.clf.intercepts_):
            params += coef.size + intercept.size
        # Mean vector
        params += model.mean_.size
        return params
    return 0

def benchmark_model(name, model, param_func, X):
    # 1. Fit the model to initialize parameters
    model.fit(X)
    
    # 2. Count parameters
    params = param_func(model)
    
    # 3. Model Size (Serialization)
    temp_file = f"temp_{name.replace(' ', '_')}.joblib"
    joblib.dump(model, temp_file)
    size_kb = os.path.getsize(temp_file) / 1024.0
    os.remove(temp_file)
    
    # 4. Inference Time
    # Warmup
    _ = model.predict_errors(X[:10])
    
    start_time = time.perf_counter()
    _ = model.predict_errors(X)
    end_time = time.perf_counter()
    
    avg_inference_ms = ((end_time - start_time) / len(X)) * 1000.0
    
    print(f"| {name:<25s} | {params:>10d} | {size_kb:>12.2f} KB | {avg_inference_ms:>18.4f} ms |")

def main():
    print("="*80)
    print("MODEL BENCHMARKING (Footprint & Inference)")
    print("="*80)
    
    # Generate dummy data for 32-dimensional Word2Vec semantic embeddings
    # We use 5000 samples for inference benchmarking
    X_dummy = np.random.randn(5000, 32).astype(np.float32)
    
    pca = PCAModel(k=6)
    rpca = RPCAModel(k=9, center=True)
    ae = AutoencoderModel(hidden_layer_sizes=(16, 8, 16))
    
    print(f"| {'Model':<25s} | {'Parameters':>10s} | {'Size (KB)':>15s} | {'Avg Inference/Sample':>20s} |")
    print("-" * 80)
    benchmark_model("PCA (k=6)", pca, count_parameters_pca, X_dummy)
    benchmark_model("RPCA (k=9)", rpca, count_parameters_rpca, X_dummy)
    benchmark_model("Autoencoder (16-8-16)", ae, count_parameters_autoencoder, X_dummy)
    print("="*80)

if __name__ == "__main__":
    main()
