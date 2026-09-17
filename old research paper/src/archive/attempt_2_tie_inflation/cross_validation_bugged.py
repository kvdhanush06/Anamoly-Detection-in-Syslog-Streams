import os
import copy
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import KFold
from models.pca_rpca import PCAModel
from models.deep_learning import AutoencoderModel
from models.thresholding import compute_threshold_pot

HDFS_FEATURES_COUNT = "data/HDFS_v1/preprocessed/features.npz"
BGL_FEATURES_W2V = "data/BGL/preprocessed/features_w2v.npz"

def l2_normalize(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms

def run_hdfs_exact_protocol(name, model, features_path, use_pot=False):
    print(f"\n{'='*80}")
    print(f"5-Fold Cross Validation: {name}")
    print(f"{'='*80}")
    
    if not os.path.exists(features_path):
        print(f"File {features_path} not found.")
        return

    data = np.load(features_path)
    X_raw = data["X_counts"].astype(np.float32)
    X = l2_normalize(X_raw)
    y = data["y"]
    
    # 1. Exact Paper Sampling
    # "we select 25,000 normal sequences and 1,000 abnormal sequences"
    normal_idx = np.where(y == 0)[0]
    abnormal_idx = np.where(y == 1)[0]
    
    rng = np.random.default_rng(42)
    sampled_normal_idx = rng.choice(normal_idx, 25000, replace=False)
    sampled_abnormal_idx = rng.choice(abnormal_idx, 1000, replace=False)
    
    # 2. 5-Fold Split on the 25,000 normals
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    metrics = {"p": [], "r": [], "f1": []}
    
    fold = 1
    for fit_n_idx, test_n_idx in kf.split(sampled_normal_idx):
        fit_idx = sampled_normal_idx[fit_n_idx]      # 4/5 * 25000 = 20000. Wait, paper says fit on 5000, test on 20000!
        # "we divide the 25,000 normal sequences into 5 folds. We use 1 fold (5,000) for tuning and 4 folds (20,000) for validation"
        # So fit_idx should be the 1 fold (test_n_idx in sklearn terminology is 1/5th, so we swap them!)
        fit_idx = sampled_normal_idx[test_n_idx]     # 5,000
        test_normal_idx = sampled_normal_idx[fit_n_idx] # 20,000
        
        test_idx = np.concatenate([test_normal_idx, sampled_abnormal_idx]) # 21,000
        
        Xf = X[fit_idx]
        Xt = X[test_idx]
        y_test = y[test_idx]
        
        fold_model = copy.deepcopy(model)
        
        print(f"Fold {fold}: Fitting on {len(Xf)} normal samples... ")
        fold_model.fit(Xf)
        
        ef = fold_model.predict_errors(Xf)
        et = fold_model.predict_errors(Xt)
        
        try:
            unique_errs = len(np.unique(et))
            if use_pot:
                theta = compute_threshold_pot(ef, q=0.90, target_fpr=1e-3)
                y_pred = (et > theta).astype(int)
                print(f"  -> Unique Test Errors: {unique_errs}")
                print(f"  -> Threshold: {theta:.4f}")
            else:
                # Rank-based top 5% selection to avoid tie-inflation
                n_flag = int(np.ceil(0.05 * len(et)))
                flags = np.argsort(et)[-n_flag:]
                y_pred = np.zeros(len(et), dtype=int)
                y_pred[flags] = 1
                
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
        except Exception as e:
            print(f"Thresholding Failed: {e}")
        
        fold += 1
        
    print("-" * 80)
    p_mean, p_std = np.mean(metrics["p"])*100, np.std(metrics["p"])*100
    r_mean, r_std = np.mean(metrics["r"])*100, np.std(metrics["r"])*100
    f1_mean, f1_std = np.mean(metrics["f1"])*100, np.std(metrics["f1"])*100
    
    print(f"FINAL 5-FOLD CV RESULT: {name}")
    print(f"Precision: {p_mean:.2f}% ± {p_std:.2f}%")
    print(f"Recall:    {r_mean:.2f}% ± {r_std:.2f}%")
    print(f"F1 Score:  {f1_mean:.2f}% ± {f1_std:.2f}%")
    return f1_mean

def main():
    pca = PCAModel(k=6)
    
    # 1. HDFS PCA Original Protocol (fit_size=5000, 95th percentile test threshold)
    run_hdfs_exact_protocol("HDFS PCA [Paper-Exact Protocol: 95th %ile]", pca, HDFS_FEATURES_COUNT, use_pot=False)
    
    # 2. HDFS PCA POT Protocol (Our Improvement)
    run_hdfs_exact_protocol("HDFS PCA [POT Threshold (Our Improvement)]", pca, HDFS_FEATURES_COUNT, use_pot=True)

if __name__ == "__main__":
    main()
