import os
import sys
import copy
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import KFold
import pickle

# Add the baseline_replication models folder to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.pca_rpca import PCAModel

os.makedirs("saved_models", exist_ok=True)

HDFS_FEATURES_COUNT = "data/HDFS_v1/preprocessed/features.npz"

def l2_normalize(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms

def run_hdfs_paper_exact_protocol():
    print(f"\n{'='*80}")
    print(f"Baseline Replication: HDFS PCA [Paper-Exact Protocol: 95th %ile Rank-Based]")
    print(f"{'='*80}")
    
    if not os.path.exists(HDFS_FEATURES_COUNT):
        print(f"File {HDFS_FEATURES_COUNT} not found.")
        return

    data = np.load(HDFS_FEATURES_COUNT)
    X_raw = data["X_counts"].astype(np.float32)
    X = l2_normalize(X_raw)
    y = data["y"]
    
    # 1. Exact Paper Sampling (25,000 normal, 1,000 abnormal)
    normal_idx = np.where(y == 0)[0]
    abnormal_idx = np.where(y == 1)[0]
    
    rng = np.random.default_rng(42)
    sampled_normal_idx = rng.choice(normal_idx, 25000, replace=False)
    sampled_abnormal_idx = rng.choice(abnormal_idx, 1000, replace=False)
    
    # 2. 5-Fold Split on the 25,000 normals
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    metrics = {"p": [], "r": [], "f1": []}
    
    model = PCAModel(k=6)
    fold = 1
    
    for fit_n_idx, test_n_idx in kf.split(sampled_normal_idx):
        fit_idx = sampled_normal_idx[test_n_idx]      # 5,000
        test_normal_idx = sampled_normal_idx[fit_n_idx] # 20,000
        test_idx = np.concatenate([test_normal_idx, sampled_abnormal_idx]) # 21,000
        
        Xf = X[fit_idx]
        Xt = X[test_idx]
        y_test = y[test_idx]
        
        model_path = f"saved_models/hdfs_pca_fold_{fold}.pkl"
        if os.path.exists(model_path):
            print(f"Fold {fold}: Loading saved model from {model_path}...")
            with open(model_path, "rb") as f:
                fold_model = pickle.load(f)
        else:
            fold_model = copy.deepcopy(model)
            print(f"Fold {fold}: Fitting on {len(Xf)} normal samples... ")
            fold_model.fit(Xf)
            
            with open(model_path, "wb") as f:
                pickle.dump(fold_model, f)
            print(f"  -> Model saved to {model_path}")
        
        et = fold_model.predict_errors(Xt)
        
        # Rank-based top 5% selection to avoid tie-inflation
        n_flag = int(np.ceil(0.05 * len(et)))
        flags = np.argsort(et)[-n_flag:]
        y_pred = np.zeros(len(et), dtype=int)
        y_pred[flags] = 1
        
        unique_errs = len(np.unique(et))
        print(f"  -> Unique Test Errors: {unique_errs}")
        print(f"  -> Threshold: Top {n_flag} samples (Rank-based 5%)")
        
        p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=0)
        
        num_flagged = int(y_pred.sum())
        num_true_anomalies = int(y_test.sum())
        test_size = len(y_test)
        true_rate = (num_true_anomalies / test_size) * 100
        flag_rate = (num_flagged / test_size) * 100
        
        print(f"  -> Flagged Anomalies: {num_flagged} ({flag_rate:.2f}%) vs True Anomalies: {num_true_anomalies} ({true_rate:.2f}%)")
        print(f"  -> Result: P: {p*100:.2f}%, R: {r*100:.2f}%, F1: {f1*100:.2f}%")
        
        metrics["p"].append(p)
        metrics["r"].append(r)
        metrics["f1"].append(f1)
        fold += 1
        
    print("-" * 80)
    p_mean, p_std = np.mean(metrics["p"])*100, np.std(metrics["p"])*100
    r_mean, r_std = np.mean(metrics["r"])*100, np.std(metrics["r"])*100
    f1_mean, f1_std = np.mean(metrics["f1"])*100, np.std(metrics["f1"])*100
    
    print(f"FINAL 5-FOLD CV RESULT: HDFS PCA [Paper-Exact Protocol]")
    print(f"Precision: {p_mean:.2f}% ± {p_std:.2f}%")
    print(f"Recall:    {r_mean:.2f}% ± {r_std:.2f}%")
    print(f"F1 Score:  {f1_mean:.2f}% ± {f1_std:.2f}%")

def run_bgl_baseline():
    print(f"\n{'='*80}")
    print(f"Baseline Application: BGL PCA (Count-based L2-Norm)")
    print(f"{'='*80}")
    
    BGL_FEATURES_COUNT = "data/BGL/preprocessed/features.npz"
    if not os.path.exists(BGL_FEATURES_COUNT):
        print(f"File {BGL_FEATURES_COUNT} not found.")
        return

    data = np.load(BGL_FEATURES_COUNT)
    X_raw = data["X_counts"].astype(np.float32)
    X = l2_normalize(X_raw)
    y = data["y"]
    
    # Standard 5-fold CV on the entire normal data for BGL
    normal_idx = np.where(y == 0)[0]
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    metrics = {"p": [], "r": [], "f1": []}
    
    model = PCAModel(k=6)
    fold = 1
    
    for fit_n_idx, test_n_idx in kf.split(normal_idx):
        fit_idx = normal_idx[test_n_idx]      # Use 1/5th for fitting to match HDFS scale loosely, or use 4/5ths. Let's use 4/5ths.
        fit_idx = normal_idx[fit_n_idx]
        test_normal_idx = normal_idx[test_n_idx]
        test_idx = np.concatenate([test_normal_idx, np.where(y == 1)[0]])
        
        Xf = X[fit_idx]
        Xt = X[test_idx]
        y_test = y[test_idx]
        
        model_path = f"saved_models/bgl_pca_fold_{fold}.pkl"
        if os.path.exists(model_path):
            print(f"Fold {fold}: Loading saved model from {model_path}...")
            with open(model_path, "rb") as f:
                fold_model = pickle.load(f)
        else:
            fold_model = copy.deepcopy(model)
            print(f"Fold {fold}: Fitting on {len(Xf)} normal samples... ")
            fold_model.fit(Xf)
            
            with open(model_path, "wb") as f:
                pickle.dump(fold_model, f)
            print(f"  -> Model saved to {model_path}")
        
        et = fold_model.predict_errors(Xt)
        
        # Rank-based top 5% selection
        n_flag = int(np.ceil(0.05 * len(et)))
        flags = np.argsort(et)[-n_flag:]
        y_pred = np.zeros(len(et), dtype=int)
        y_pred[flags] = 1
        
        p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=0)
        print(f"  -> Result: P: {p*100:.2f}%, R: {r*100:.2f}%, F1: {f1*100:.2f}%")
        
        metrics["p"].append(p)
        metrics["r"].append(r)
        metrics["f1"].append(f1)
        fold += 1
        
    print("-" * 80)
    p_mean, p_std = np.mean(metrics["p"])*100, np.std(metrics["p"])*100
    r_mean, r_std = np.mean(metrics["r"])*100, np.std(metrics["r"])*100
    f1_mean, f1_std = np.mean(metrics["f1"])*100, np.std(metrics["f1"])*100
    
    print(f"FINAL 5-FOLD CV RESULT: BGL PCA (Count-based Baseline)")
    print(f"Precision: {p_mean:.2f}% ± {p_std:.2f}%")
    print(f"Recall:    {r_mean:.2f}% ± {r_std:.2f}%")
    print(f"F1 Score:  {f1_mean:.2f}% ± {f1_std:.2f}%")

if __name__ == "__main__":
    run_hdfs_paper_exact_protocol()
    # To run BGL, we uncomment the line below:
    run_bgl_baseline()
