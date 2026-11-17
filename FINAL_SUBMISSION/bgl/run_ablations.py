import os
import time
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, average_precision_score, roc_auc_score, confusion_matrix

from data_utils import load_data, get_10fold_splits
from thresholding import compute_threshold_mode_a, compute_threshold_mode_b
from deep_learning import PCAModel, IsolationForestModel
from feature_utils import train_w2v, get_mean_w2v, get_mean_max_w2v

def run_ablations(feature_type="mean_max"):
    feat_func = get_mean_max_w2v if feature_type == "mean_max" else get_mean_w2v
    print("=" * 80)
    print(f"ABLATION STUDY: Testing Baseline Models on the SAME Word2Vec Features ({feature_type})")
    print("Models: 1) PCA (k tuned on Val)  2) Isolation Forest  3) All-Normal  4) Random")
    print("=" * 80)
    
    sequences, y = load_data()
    splits = get_10fold_splits(y, seed=42)
    
    # 1. Choose optimal k for PCA using Fold 1 validation set
    f1 = splits[0]
    train_seqs = sequences[f1["train_idx"]].tolist()
    val_norm_seqs = sequences[f1["val_norm_idx"]].tolist()
    val_seqs = sequences[f1["val_idx"]].tolist()
    y_val = y[f1["val_idx"]]
    
    w2v = train_w2v(train_seqs, vector_size=32, window=5, min_count=1, workers=1, sg=1, seed=42)
    X_tr = feat_func(train_seqs, w2v)
    X_vn = feat_func(val_norm_seqs, w2v)
    X_v = feat_func(val_seqs, w2v)
    
    print("\nTuning PCA k on Fold 1 Validation Set...")
    best_k = 6
    best_val_f1 = -1.0
    candidate_ks = [2, 4, 6, 8, 12, 16, 20, 24, 32, 48] if feature_type == "mean_max" else [2, 4, 6, 8, 12, 16, 20]
    for k in candidate_ks:
        pca_tune = PCAModel(k=k)
        pca_tune.fit(X_tr)
        err_vn = pca_tune.predict_errors(X_vn)
        err_v = pca_tune.predict_errors(X_v)
        th, _ = compute_threshold_mode_a(err_vn)
        yp = (err_v > th).astype(int)
        _, _, f1_score, _ = precision_recall_fscore_support(y_val, yp, average="binary", zero_division=0)
        print(f"  k={k:2d} -> Val Mode A F1: {f1_score*100:.2f}%")
        if f1_score > best_val_f1:
            best_val_f1 = f1_score
            best_k = k
    print(f"Selected PCA k={best_k} (Val F1: {best_val_f1*100:.2f}%)\n")
    
    ablation_records = []
    
    # Evaluate across all 10 folds
    for split in splits:
        fold = split["fold"]
        print(f"\n--- Running Ablations on Fold {fold} ---")
        
        train_idx = split["train_idx"]
        val_norm_idx = split["val_norm_idx"]
        val_idx = split["val_idx"]
        test_idx = split["test_idx"]
        
        train_seqs = sequences[train_idx].tolist()
        val_norm_seqs = sequences[val_norm_idx].tolist()
        val_seqs = sequences[val_idx].tolist()
        test_seqs = sequences[test_idx].tolist()
        
        y_val = y[val_idx]
        y_test = y[test_idx]
        
        w2v_fold = train_w2v(train_seqs, vector_size=32, window=5, min_count=1, workers=1, sg=1, seed=42)
        X_tr = feat_func(train_seqs, w2v_fold)
        X_vn = feat_func(val_norm_seqs, w2v_fold)
        X_v = feat_func(val_seqs, w2v_fold)
        X_te = feat_func(test_seqs, w2v_fold)
        
        # A. PCA
        pca = PCAModel(k=best_k)
        t0 = time.time()
        pca.fit(X_tr)
        pca_time = time.time() - t0
        
        err_vn = pca.predict_errors(X_vn)
        err_v = pca.predict_errors(X_v)
        err_te = pca.predict_errors(X_te)
        
        th_a, _ = compute_threshold_mode_a(err_vn)
        th_b, _ = compute_threshold_mode_b(err_v, y_val)
        
        yp_a = (err_te > th_a).astype(int)
        tn_a, fp_a, fn_a, tp_a = confusion_matrix(y_test, yp_a).ravel()
        p_a, r_a, f1_a, _ = precision_recall_fscore_support(y_test, yp_a, average="binary", zero_division=0)
        
        yp_b = (err_te > th_b).astype(int)
        tn_b, fp_b, fn_b, tp_b = confusion_matrix(y_test, yp_b).ravel()
        p_b, r_b, f1_b, _ = precision_recall_fscore_support(y_test, yp_b, average="binary", zero_division=0)
        
        pr_auc = float(average_precision_score(y_test, err_te))
        roc_auc = float(roc_auc_score(y_test, err_te))
        
        ablation_records.append({
            "model": f"PCA (k={best_k})", "fold": fold, "fit_time_s": pca_time,
            "mode_a_theta": th_a, "mode_a_tp": tp_a, "mode_a_fp": fp_a, "mode_a_tn": tn_a, "mode_a_fn": fn_a,
            "mode_a_p": p_a, "mode_a_r": r_a, "mode_a_f1": f1_a,
            "mode_b_theta": th_b, "mode_b_tp": tp_b, "mode_b_fp": fp_b, "mode_b_tn": tn_b, "mode_b_fn": fn_b,
            "mode_b_p": p_b, "mode_b_r": r_b, "mode_b_f1": f1_b,
            "pr_auc": pr_auc, "roc_auc": roc_auc
        })
        print(f"Fold {fold} PCA: Mode A F1={f1_a*100:.2f}%, Mode B F1={f1_b*100:.2f}%, PR-AUC={pr_auc*100:.2f}%")
        
        # B. Isolation Forest
        iso = IsolationForestModel(n_estimators=100, random_state=42)
        t0 = time.time()
        iso.fit(X_tr)
        iso_time = time.time() - t0
        
        err_vn = iso.predict_errors(X_vn)
        err_v = iso.predict_errors(X_v)
        err_te = iso.predict_errors(X_te)
        
        th_a, _ = compute_threshold_mode_a(err_vn)
        th_b, _ = compute_threshold_mode_b(err_v, y_val)
        
        yp_a = (err_te > th_a).astype(int)
        tn_a, fp_a, fn_a, tp_a = confusion_matrix(y_test, yp_a).ravel()
        p_a, r_a, f1_a, _ = precision_recall_fscore_support(y_test, yp_a, average="binary", zero_division=0)
        
        yp_b = (err_te > th_b).astype(int)
        tn_b, fp_b, fn_b, tp_b = confusion_matrix(y_test, yp_b).ravel()
        p_b, r_b, f1_b, _ = precision_recall_fscore_support(y_test, yp_b, average="binary", zero_division=0)
        
        pr_auc = float(average_precision_score(y_test, err_te))
        roc_auc = float(roc_auc_score(y_test, err_te))
        
        ablation_records.append({
            "model": "Isolation Forest", "fold": fold, "fit_time_s": iso_time,
            "mode_a_theta": th_a, "mode_a_tp": tp_a, "mode_a_fp": fp_a, "mode_a_tn": tn_a, "mode_a_fn": fn_a,
            "mode_a_p": p_a, "mode_a_r": r_a, "mode_a_f1": f1_a,
            "mode_b_theta": th_b, "mode_b_tp": tp_b, "mode_b_fp": fp_b, "mode_b_tn": tn_b, "mode_b_fn": fn_b,
            "mode_b_p": p_b, "mode_b_r": r_b, "mode_b_f1": f1_b,
            "pr_auc": pr_auc, "roc_auc": roc_auc
        })
        print(f"Fold {fold} Isolation Forest: Mode A F1={f1_a*100:.2f}%, Mode B F1={f1_b*100:.2f}%, PR-AUC={pr_auc*100:.2f}%")
        
        # C. Predict All Normal
        yp_norm = np.zeros_like(y_test)
        tn, fp, fn, tp = confusion_matrix(y_test, yp_norm).ravel()
        p, r, f1, _ = precision_recall_fscore_support(y_test, yp_norm, average="binary", zero_division=0)
        ablation_records.append({
            "model": "Predict All Normal", "fold": fold, "fit_time_s": 0.0,
            "mode_a_theta": 0.0, "mode_a_tp": tp, "mode_a_fp": fp, "mode_a_tn": tn, "mode_a_fn": fn,
            "mode_a_p": p, "mode_a_r": r, "mode_a_f1": f1,
            "mode_b_theta": 0.0, "mode_b_tp": tp, "mode_b_fp": fp, "mode_b_tn": tn, "mode_b_fn": fn,
            "mode_b_p": p, "mode_b_r": r, "mode_b_f1": f1,
            "pr_auc": float(np.mean(y_test)), "roc_auc": 0.50
        })
        
        # D. Random Baseline
        rng = np.random.default_rng(42 + fold)
        p_anom = np.mean(y_val)
        yp_rand = rng.binomial(1, p_anom, size=len(y_test))
        tn, fp, fn, tp = confusion_matrix(y_test, yp_rand).ravel()
        p, r, f1, _ = precision_recall_fscore_support(y_test, yp_rand, average="binary", zero_division=0)
        ablation_records.append({
            "model": "Random Guessing", "fold": fold, "fit_time_s": 0.0,
            "mode_a_theta": 0.0, "mode_a_tp": tp, "mode_a_fp": fp, "mode_a_tn": tn, "mode_a_fn": fn,
            "mode_a_p": p, "mode_a_r": r, "mode_a_f1": f1,
            "mode_b_theta": 0.0, "mode_b_tp": tp, "mode_b_fp": fp, "mode_b_tn": tn, "mode_b_fn": fn,
            "mode_b_p": p, "mode_b_r": r, "mode_b_f1": f1,
            "pr_auc": float(np.mean(y_test)), "roc_auc": 0.50
        })

    df = pd.DataFrame(ablation_records)
    out_csv = "syslog_v2_clean/ablation_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nAblations complete! Saved to {out_csv}")
    
    summary = df.groupby("model")[["mode_a_p", "mode_a_r", "mode_a_f1", "mode_b_f1", "pr_auc", "roc_auc"]].mean().reset_index()
    print("\n" + "=" * 80)
    print("ABLATION STUDY SUMMARY (Averaged across 10 folds on Test Set)")
    print("=" * 80)
    for _, row in summary.iterrows():
        print(f"| {row['model']:<22s} | Mode A F1: {row['mode_a_f1']*100:6.2f}% (P: {row['mode_a_p']*100:5.2f}%, R: {row['mode_a_r']*100:5.2f}%) | Mode B F1: {row['mode_b_f1']*100:6.2f}% | PR-AUC: {row['pr_auc']*100:5.2f}% | ROC-AUC: {row['roc_auc']*100:5.2f}% |")
    print("=" * 80)
    return df

if __name__ == "__main__":
    run_ablations()
