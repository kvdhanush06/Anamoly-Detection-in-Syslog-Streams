import os
import time
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, average_precision_score, roc_auc_score, confusion_matrix

from data_utils import load_data, get_chronological_split
from thresholding import compute_threshold_mode_a, compute_threshold_mode_b
from deep_learning import AutoencoderModel
from feature_utils import train_w2v, compute_oov_stats, get_mean_w2v, get_mean_max_w2v

def run_chronological(best_arch=(32, 16, 32), best_alpha=1e-4, feature_type="mean_max"):
    feat_func = get_mean_max_w2v if feature_type == "mean_max" else get_mean_w2v
    print("=" * 80)
    print(f"CHRONOLOGICAL ROBUSTNESS EXPERIMENT (Feature: {feature_type})")
    print("Protocol: Train on earliest 70% of normals, Validate on next 10% of normals (+anomalies),")
    print("          Test on latest 20% of timeline (all its normals and anomalies).")
    print("=" * 80)
    
    sequences, y = load_data()
    split = get_chronological_split(y)
    
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
    
    print(f"Chronological Counts:")
    print(f"  Train Normals: {len(train_idx):,}")
    print(f"  Val Total:     {len(val_idx):,} (Normals: {len(val_norm_idx):,}, Anomalies: {len(split['val_anom_idx']):,})")
    print(f"  Test Total:    {len(test_idx):,} (Normals: {len(split['test_norm_idx']):,}, Anomalies: {len(split['test_anom_idx']):,})")
    
    test_prevalence = len(split['test_anom_idx']) / len(test_idx) * 100
    print(f"  Test Anomaly Prevalence: {test_prevalence:.2f}%")
    
    print("\nTraining Word2Vec strictly on earliest 70% normal sequences...")
    t0 = time.time()
    w2v = train_w2v(train_seqs, vector_size=32, window=5, min_count=1, workers=1, sg=1, seed=42)
    w2v_time = time.time() - t0
    print(f"Word2Vec trained in {w2v_time:.2f}s (Vocab size: {len(w2v.wv):,})")
    
    total_tokens, oov_tokens, oov_rate = compute_oov_stats(test_seqs, w2v)
    print(f"Chronological Test Set OOV: {oov_tokens:,} / {total_tokens:,} tokens ({oov_rate*100:.3f}%)")
    
    print("Extracting features...")
    X_train = feat_func(train_seqs, w2v)
    X_val_norm = feat_func(val_norm_seqs, w2v)
    X_val = feat_func(val_seqs, w2v)
    X_test = feat_func(test_seqs, w2v)
    
    print(f"Fitting Autoencoder (arch={best_arch}, alpha={best_alpha})...")
    ae = AutoencoderModel(hidden_layer_sizes=best_arch, alpha=best_alpha, random_state=42)
    t0 = time.time()
    ae.fit(X_train)
    ae_time = time.time() - t0
    print(f"Autoencoder fit in {ae_time:.2f}s")
    
    err_vn = ae.predict_errors(X_val_norm)
    err_v = ae.predict_errors(X_val)
    err_t = ae.predict_errors(X_test)
    
    # Thresholds
    theta_a, fallback_a = compute_threshold_mode_a(err_vn, q=0.90, target_fpr=1e-3, fallback_percentile=99.5)
    theta_b, val_f1_b = compute_threshold_mode_b(err_v, y_val)
    
    # Test Evaluation Mode A
    y_pred_a = (err_t > theta_a).astype(int)
    tn_a, fp_a, fn_a, tp_a = confusion_matrix(y_test, y_pred_a).ravel()
    p_a, r_a, f1_a, _ = precision_recall_fscore_support(y_test, y_pred_a, average="binary", zero_division=0)
    
    # Test Evaluation Mode B
    y_pred_b = (err_t > theta_b).astype(int)
    tn_b, fp_b, fn_b, tp_b = confusion_matrix(y_test, y_pred_b).ravel()
    p_b, r_b, f1_b, _ = precision_recall_fscore_support(y_test, y_pred_b, average="binary", zero_division=0)
    
    # Threshold-free metrics
    pr_auc = float(average_precision_score(y_test, err_t))
    roc_auc = float(roc_auc_score(y_test, err_t))
    
    print("\n" + "=" * 80)
    print("CHRONOLOGICAL TEST RESULTS")
    print("=" * 80)
    print(f"Mode A (Label-Free POT, theta={theta_a:.4f}, fallback={fallback_a}):")
    print(f"  TP: {tp_a:,} | FP: {fp_a:,} | TN: {tn_a:,} | FN: {fn_a:,}")
    print(f"  Precision: {p_a*100:.2f}% | Recall: {r_a*100:.2f}% | F1-Score: {f1_a*100:.2f}%")
    print(f"\nMode B (Validation-Tuned, theta={theta_b:.4f}):")
    print(f"  TP: {tp_b:,} | FP: {fp_b:,} | TN: {tn_b:,} | FN: {fn_b:,}")
    print(f"  Precision: {p_b*100:.2f}% | Recall: {r_b*100:.2f}% | F1-Score: {f1_b*100:.2f}%")
    print(f"\nThreshold-Free:")
    print(f"  PR-AUC:  {pr_auc*100:.2f}%")
    print(f"  ROC-AUC: {roc_auc*100:.2f}%")
    print("=" * 80)
    
    res = {
        "split": "chronological",
        "oov_rate": oov_rate,
        "mode_a_theta": theta_a, "mode_a_fallback": fallback_a,
        "mode_a_tp": tp_a, "mode_a_fp": fp_a, "mode_a_tn": tn_a, "mode_a_fn": fn_a,
        "mode_a_p": p_a, "mode_a_r": r_a, "mode_a_f1": f1_a,
        "mode_b_theta": theta_b,
        "mode_b_tp": tp_b, "mode_b_fp": fp_b, "mode_b_tn": tn_b, "mode_b_fn": fn_b,
        "mode_b_p": p_b, "mode_b_r": r_b, "mode_b_f1": f1_b,
        "pr_auc": pr_auc, "roc_auc": roc_auc
    }
    pd.DataFrame([res]).to_csv("syslog_v2_clean/chronological_results.csv", index=False)
    return res

if __name__ == "__main__":
    run_chronological()
