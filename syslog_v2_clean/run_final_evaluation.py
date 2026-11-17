import os
import sys
import time
import pickle
import argparse
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.metrics import precision_recall_fscore_support, average_precision_score, roc_auc_score, confusion_matrix

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
sys.path.insert(0, script_dir)

from data_utils import load_data, get_10fold_splits
from thresholding import compute_threshold_mode_a, compute_threshold_mode_b
from deep_learning import AutoencoderModel
from feature_utils import (
    train_w2v,
    compute_oov_stats,
    get_mean_w2v,
    get_mean_max_w2v,
    compute_idf_weights,
    get_idf_w2v
)

# Resolve results and model directories
if os.path.exists(os.path.join(repo_root, "FINAL_SUBMISSION", "results")):
    RESULTS_DIR = os.path.join(repo_root, "FINAL_SUBMISSION", "results")
elif os.path.exists(os.path.join(repo_root, "syslog_v2_clean")):
    RESULTS_DIR = os.path.join(repo_root, "syslog_v2_clean")
else:
    RESULTS_DIR = script_dir

potential_model_dirs = [
    os.path.join(script_dir, "saved_models"),
    os.path.join(repo_root, "FINAL_SUBMISSION", "bgl", "saved_models"),
    os.path.join(repo_root, "syslog_v2_clean", "saved_models"),
]
SAVED_MODELS_DIR = next((d for d in potential_model_dirs if os.path.exists(d)), os.path.join(script_dir, "saved_models"))
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)


def run_10fold_eval(arch=(32, 16, 32), alpha=1e-4, feature_type="mean_max", use_ensemble=False, ensemble_seeds=[42, 43, 44, 45, 46], max_folds=None, load_saved=False):
    print("=" * 80)
    print("FINAL 10-FOLD EVALUATION ON THE BGL TEST SET (NATURAL PREVALENCE)")
    print(f"Configuration: Arch={arch}, Alpha={alpha}, Feature={feature_type}, Ensemble={use_ensemble}")
    print(f"Mode: {'Evaluating Saved Models' if load_saved else 'Training from Scratch'} | Max Folds: {max_folds or 10}")
    print("Rule: Test data is evaluated EXACTLY ONCE per fold. ZERO test label feedback.")
    print("=" * 80)
    
    sequences, y = load_data()
    splits = get_10fold_splits(y, seed=42)
    if max_folds is not None:
        splits = splits[:max_folds]
    
    test_results = []
    balanced_results = []
    
    feat_func = get_mean_max_w2v if feature_type == "mean_max" else get_mean_w2v

    for split in splits:
        fold = split["fold"]
        print(f"\n{'='*40}\n>>> Running Fold {fold} / 10 <<<\n{'='*40}")
        
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
        
        w2v_save_path = os.path.join(SAVED_MODELS_DIR, f"bgl_w2v_clean_fold_{fold}.pkl")
        ae_save_path = os.path.join(SAVED_MODELS_DIR, f"bgl_ae_clean_fold_{fold}.pkl")
        
        if load_saved and os.path.exists(w2v_save_path) and os.path.exists(ae_save_path):
            print(f"Loading pre-trained Word2Vec from {w2v_save_path}...")
            w2v = Word2Vec.load(w2v_save_path)
            
            print(f"Loading pre-trained Autoencoder from {ae_save_path}...")
            with open(ae_save_path, "rb") as f:
                ae = pickle.load(f)
                
            print("Extracting features on validation and test sets...")
            X_vn = feat_func(val_norm_seqs, w2v)
            X_v = feat_func(val_seqs, w2v)
            X_te = feat_func(test_seqs, w2v)
            
            err_vn = ae.predict_errors(X_vn)
            err_v = ae.predict_errors(X_v)
            err_te = ae.predict_errors(X_te)
        else:
            # 1. Train Word2Vec purely on train normal sequences
            t0 = time.time()
            w2v = train_w2v(train_seqs, vector_size=32, window=5, min_count=1, workers=1, sg=1, seed=42)
            w2v_train_time = time.time() - t0
            w2v.save(w2v_save_path)
            print(f"Word2Vec trained in {w2v_train_time:.2f}s (Vocab: {len(w2v.wv):,})")
            
            # 2. OOV Stats on test set
            total_toks, oov_toks, oov_rate = compute_oov_stats(test_seqs, w2v)
            print(f"Test Set OOV: {oov_toks:,} / {total_toks:,} tokens ({oov_rate*100:.3f}%)")
            
            # 3. Extract features
            if feature_type == "mean":
                X_tr = get_mean_w2v(train_seqs, w2v)
                X_vn = get_mean_w2v(val_norm_seqs, w2v)
                X_v = get_mean_w2v(val_seqs, w2v)
                X_te = get_mean_w2v(test_seqs, w2v)
            elif feature_type == "mean_max":
                X_tr = get_mean_max_w2v(train_seqs, w2v)
                X_vn = get_mean_max_w2v(val_norm_seqs, w2v)
                X_v = get_mean_max_w2v(val_seqs, w2v)
                X_te = get_mean_max_w2v(test_seqs, w2v)
            elif feature_type == "idf":
                idf_dict = compute_idf_weights(train_seqs)
                X_tr = get_idf_w2v(train_seqs, w2v, idf_dict)
                X_vn = get_idf_w2v(val_norm_seqs, w2v, idf_dict)
                X_v = get_idf_w2v(val_seqs, w2v, idf_dict)
                X_te = get_idf_w2v(test_seqs, w2v, idf_dict)
                
            # 4. Fit Autoencoder
            if use_ensemble:
                models = []
                errs_vn = []
                errs_v = []
                errs_te = []
                t0 = time.time()
                for seed in ensemble_seeds:
                    m = AutoencoderModel(hidden_layer_sizes=arch, alpha=alpha, random_state=seed)
                    m.fit(X_tr)
                    models.append(m)
                    errs_vn.append(m.predict_errors(X_vn))
                    errs_v.append(m.predict_errors(X_v))
                    errs_te.append(m.predict_errors(X_te))
                ae_train_time = time.time() - t0
                err_vn = np.mean(errs_vn, axis=0)
                err_v = np.mean(errs_v, axis=0)
                err_te = np.mean(errs_te, axis=0)
                with open(ae_save_path, "wb") as f:
                    pickle.dump(models[0], f)
            else:
                ae = AutoencoderModel(hidden_layer_sizes=arch, alpha=alpha, random_state=42)
                t0 = time.time()
                ae.fit(X_tr)
                ae_train_time = time.time() - t0
                err_vn = ae.predict_errors(X_vn)
                err_v = ae.predict_errors(X_v)
                err_te = ae.predict_errors(X_te)
                with open(ae_save_path, "wb") as f:
                    pickle.dump(ae, f)
            print(f"Autoencoder fit in {ae_train_time:.2f}s")
        
        # 5. Threshold Calculation (Zero Test Leakage)
        theta_a, fallback_a = compute_threshold_mode_a(err_vn, q=0.90, target_fpr=1e-3, fallback_percentile=99.5)
        theta_b, val_f1_b = compute_threshold_mode_b(err_v, y_val)
        print(f"Thresholds -> Mode A: {theta_a:.4f} (fallback={fallback_a}) | Mode B: {theta_b:.4f} (Val F1: {val_f1_b*100:.2f}%)")
        
        # 6. Evaluation on Natural Prevalence Test Set
        yp_a = (err_te > theta_a).astype(int)
        tn_a, fp_a, fn_a, tp_a = confusion_matrix(y_test, yp_a).ravel()
        p_a, r_a, f1_a, _ = precision_recall_fscore_support(y_test, yp_a, average="binary", zero_division=0)
        
        yp_b = (err_te > theta_b).astype(int)
        tn_b, fp_b, fn_b, tp_b = confusion_matrix(y_test, yp_b).ravel()
        p_b, r_b, f1_b, _ = precision_recall_fscore_support(y_test, yp_b, average="binary", zero_division=0)
        
        pr_auc = float(average_precision_score(y_test, err_te))
        roc_auc = float(roc_auc_score(y_test, err_te))
        
        print(f"Test Result Fold {fold}:")
        print(f"  Mode A (Headline): P={p_a*100:.2f}%, R={r_a*100:.2f}%, F1={f1_a*100:.2f}% | TP={tp_a:,}, FP={fp_a:,}, TN={tn_a:,}, FN={fn_a:,}")
        print(f"  Mode B (Val-Tuned): P={p_b*100:.2f}%, R={r_b*100:.2f}%, F1={f1_b*100:.2f}% | TP={tp_b:,}, FP={fp_b:,}, TN={tn_b:,}, FN={fn_b:,}")
        print(f"  Threshold-Free:    PR-AUC={pr_auc*100:.2f}%, ROC-AUC={roc_auc*100:.2f}%")
        
        test_results.append({
            "fold": fold,
            "mode_a_theta": theta_a, "mode_a_fallback": fallback_a,
            "mode_a_tp": tp_a, "mode_a_fp": fp_a, "mode_a_tn": tn_a, "mode_a_fn": fn_a,
            "mode_a_p": p_a, "mode_a_r": r_a, "mode_a_f1": f1_a,
            "mode_b_theta": theta_b,
            "mode_b_tp": tp_b, "mode_b_fp": fp_b, "mode_b_tn": tn_b, "mode_b_fn": fn_b,
            "mode_b_p": p_b, "mode_b_r": r_b, "mode_b_f1": f1_b,
            "pr_auc": pr_auc, "roc_auc": roc_auc
        })
        
        # 7. Balanced Perspective
        test_norm_idx = np.where(y_test == 0)[0]
        test_anom_idx = np.where(y_test == 1)[0]
        n_anom = len(test_anom_idx)
        rng_bal = np.random.default_rng(42 + fold)
        sub_norm_idx = rng_bal.choice(test_norm_idx, size=n_anom, replace=False)
        bal_idx = np.concatenate([sub_norm_idx, test_anom_idx])
        
        err_bal = err_te[bal_idx]
        y_bal = y_test[bal_idx]
        
        yp_bal_a = (err_bal > theta_a).astype(int)
        tn_bal_a, fp_bal_a, fn_bal_a, tp_bal_a = confusion_matrix(y_bal, yp_bal_a).ravel()
        p_bal_a, r_bal_a, f1_bal_a, _ = precision_recall_fscore_support(y_bal, yp_bal_a, average="binary", zero_division=0)
        
        yp_bal_b = (err_bal > theta_b).astype(int)
        tn_bal_b, fp_bal_b, fn_bal_b, tp_bal_b = confusion_matrix(y_bal, yp_bal_b).ravel()
        p_bal_b, r_bal_b, f1_bal_b, _ = precision_recall_fscore_support(y_bal, yp_bal_b, average="binary", zero_division=0)
        
        balanced_results.append({
            "fold": fold,
            "mode_a_tp": tp_bal_a, "mode_a_fp": fp_bal_a, "mode_a_tn": tn_bal_a, "mode_a_fn": fn_bal_a,
            "mode_a_p": p_bal_a, "mode_a_r": r_bal_a, "mode_a_f1": f1_bal_a,
            "mode_b_tp": tp_bal_b, "mode_b_fp": fp_bal_b, "mode_b_tn": tn_bal_b, "mode_b_fn": fn_bal_b,
            "mode_b_p": p_bal_b, "mode_b_r": r_bal_b, "mode_b_f1": f1_bal_b
        })

    df_test = pd.DataFrame(test_results)
    out_test_csv = os.path.join(RESULTS_DIR, "test_results_10fold.csv")
    df_test.to_csv(out_test_csv, index=False)
    
    df_bal = pd.DataFrame(balanced_results)
    out_bal_csv = os.path.join(RESULTS_DIR, "test_balanced_results_10fold.csv")
    df_bal.to_csv(out_bal_csv, index=False)
    
    print("\n" + "=" * 80)
    print("FINAL TEST RESULTS SUMMARY (NATURAL PREVALENCE)")
    print("=" * 80)
    print(f"Mode A (Headline Label-Free POT):")
    print(f"  Precision: {df_test['mode_a_p'].mean()*100:.2f}% ± {df_test['mode_a_p'].std()*100:.2f}%")
    print(f"  Recall:    {df_test['mode_a_r'].mean()*100:.2f}% ± {df_test['mode_a_r'].std()*100:.2f}%")
    print(f"  F1-Score:  {df_test['mode_a_f1'].mean()*100:.2f}% ± {df_test['mode_a_f1'].std()*100:.2f}%")
    print(f"\nMode B (Validation-Tuned Threshold):")
    print(f"  Precision: {df_test['mode_b_p'].mean()*100:.2f}% ± {df_test['mode_b_p'].std()*100:.2f}%")
    print(f"  Recall:    {df_test['mode_b_r'].mean()*100:.2f}% ± {df_test['mode_b_r'].std()*100:.2f}%")
    print(f"  F1-Score:  {df_test['mode_b_f1'].mean()*100:.2f}% ± {df_test['mode_b_f1'].std()*100:.2f}%")
    print(f"\nThreshold-Free:")
    print(f"  PR-AUC:    {df_test['pr_auc'].mean()*100:.2f}% ± {df_test['pr_auc'].std()*100:.2f}%")
    print(f"  ROC-AUC:   {df_test['roc_auc'].mean()*100:.2f}% ± {df_test['roc_auc'].std()*100:.2f}%")
    print("=" * 80)
    
    print("\nBALANCED VIEW (~50% Anomalies - for direct comparison with older setup):")
    print(f"Mode A: P={df_bal['mode_a_p'].mean()*100:.2f}%, R={df_bal['mode_a_r'].mean()*100:.2f}%, F1={df_bal['mode_a_f1'].mean()*100:.2f}%")
    print(f"Mode B: P={df_bal['mode_b_p'].mean()*100:.2f}%, R={df_bal['mode_b_r'].mean()*100:.2f}%, F1={df_bal['mode_b_f1'].mean()*100:.2f}%")
    print("=" * 80)
    print(f"Results CSVs saved to:\n  - {out_test_csv}\n  - {out_bal_csv}")
    
    return df_test, df_bal


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Clean 10-Fold BGL Evaluation")
    parser.add_argument("--folds", type=int, default=None, help="Number of folds to run (default: all 10)")
    parser.add_argument("--load-saved", action="store_true", help="Load existing saved models to evaluate without retraining")
    args = parser.parse_args()

    run_10fold_eval(arch=(32, 16, 32), alpha=1e-4, feature_type="mean_max", use_ensemble=False, max_folds=args.folds, load_saved=args.load_saved)
