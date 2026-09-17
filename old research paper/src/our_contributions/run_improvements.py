import os
import sys
import copy
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import KFold
import pickle

# Add the our_contributions models folder to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.deep_learning import AutoencoderModel
from models.thresholding import compute_threshold_pot
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'baseline_replication'))
from models.pca_rpca import PCAModel

os.makedirs("saved_models_our_contributions", exist_ok=True)

HDFS_FEATURES_COUNT = "data/HDFS_v1/preprocessed/features.npz"
BGL_FEATURES_W2V = "data/BGL/preprocessed/features_w2v.npz"

def l2_normalize(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms

def run_hdfs_pot_improvement():
    print(f"\n{'='*80}")
    print(f"Our Contribution 1: HDFS PCA [POT Threshold Improvement]")
    print(f"{'='*80}")
    
    data = np.load(HDFS_FEATURES_COUNT)
    X = l2_normalize(data["X_counts"].astype(np.float32))
    y = data["y"]
    
    normal_idx = np.where(y == 0)[0]
    abnormal_idx = np.where(y == 1)[0]
    
    rng = np.random.default_rng(42)
    sampled_normal_idx = rng.choice(normal_idx, 25000, replace=False)
    sampled_abnormal_idx = rng.choice(abnormal_idx, 1000, replace=False)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    metrics = {"p": [], "r": [], "f1": []}
    
    model = PCAModel(k=6)
    fold = 1
    
    for fit_n_idx, test_n_idx in kf.split(sampled_normal_idx):
        fit_idx = sampled_normal_idx[test_n_idx]
        test_idx = np.concatenate([sampled_normal_idx[fit_n_idx], sampled_abnormal_idx])
        
        Xf, Xt, y_test = X[fit_idx], X[test_idx], y[test_idx]
        
        model_path = f"saved_models_our_contributions/hdfs_pot_pca_fold_{fold}.pkl"
        if os.path.exists(model_path):
            print(f"Fold {fold}: Loading saved model from {model_path}...")
            with open(model_path, "rb") as f:
                fold_model = pickle.load(f)
        else:
            fold_model = copy.deepcopy(model)
            fold_model.fit(Xf)
            
            with open(model_path, "wb") as f:
                pickle.dump(fold_model, f)
            print(f"  -> Model saved to {model_path}")
        
        ef = fold_model.predict_errors(Xf)
        et = fold_model.predict_errors(Xt)
        
        theta = compute_threshold_pot(ef, q=0.90, target_fpr=1e-3)
        y_pred = (et > theta).astype(int)
        
        p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=0)
        print(f"Fold {fold}: P: {p*100:.2f}%, R: {r*100:.2f}%, F1: {f1*100:.2f}% (Threshold: {theta:.4f})")
        
        metrics["p"].append(p); metrics["r"].append(r); metrics["f1"].append(f1)
        fold += 1
        
    print(f"FINAL HDFS POT RESULT: P: {np.mean(metrics['p'])*100:.2f}%, R: {np.mean(metrics['r'])*100:.2f}%, F1: {np.mean(metrics['f1'])*100:.2f}%\n")

def run_bgl_autoencoder():
    print(f"{'='*80}")
    print(f"Our Contribution 2: BGL Autoencoder (Node-Based Semantic Word2Vec)")
    print(f"{'='*80}")
    
    data = np.load(BGL_FEATURES_W2V)
    X = data["X_l2"]
    y = data["y"]
    
    normal_idx = np.where(y == 0)[0]
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    metrics = {"p": [], "r": [], "f1": []}
    
    model = AutoencoderModel(hidden_layer_sizes=(16, 8, 16))
    fold = 1
    
    for fit_idx, test_normal_idx in kf.split(normal_idx):
        Xf = X[normal_idx[fit_idx]]
        test_idx = np.concatenate([normal_idx[test_normal_idx], np.where(y == 1)[0]])
        Xt, y_test = X[test_idx], y[test_idx]
        
        model_path = f"saved_models_our_contributions/bgl_autoencoder_fold_{fold}.pkl"
        if os.path.exists(model_path):
            print(f"Fold {fold}: Loading saved model from {model_path}...")
            with open(model_path, "rb") as f:
                fold_model = pickle.load(f)
        else:
            fold_model = copy.deepcopy(model)
            print(f"Fold {fold}: Fitting on {len(Xf)} semantic sequences...")
            fold_model.fit(Xf)
            
            with open(model_path, "wb") as f:
                pickle.dump(fold_model, f)
            print(f"  -> Model saved to {model_path}")
        
        ef = fold_model.predict_errors(Xf)
        et = fold_model.predict_errors(Xt)
        
        theta = compute_threshold_pot(ef, q=0.90, target_fpr=1e-3)
        y_pred = (et > theta).astype(int)
        
        p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=0)
        print(f"Fold {fold}: P: {p*100:.2f}%, R: {r*100:.2f}%, F1: {f1*100:.2f}% (Threshold: {theta:.4f})")
        
        metrics["p"].append(p); metrics["r"].append(r); metrics["f1"].append(f1)
        fold += 1

    print("-" * 80)
    print(f"FINAL BGL AUTOENCODER RESULT:")
    print(f"Precision: {np.mean(metrics['p'])*100:.2f}% ± {np.std(metrics['p'])*100:.2f}%")
    print(f"Recall:    {np.mean(metrics['r'])*100:.2f}% ± {np.std(metrics['r'])*100:.2f}%")
    print(f"F1 Score:  {np.mean(metrics['f1'])*100:.2f}% ± {np.std(metrics['f1'])*100:.2f}%")

if __name__ == "__main__":
    run_hdfs_pot_improvement()
    run_bgl_autoencoder()
