import os
import time
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from data_utils import load_data, get_10fold_splits
from thresholding import compute_threshold_pot, compute_threshold_mode_a, compute_threshold_mode_b
from deep_learning import AutoencoderModel
from feature_utils import (
    train_w2v,
    compute_oov_stats,
    get_mean_w2v,
    get_mean_max_w2v,
    compute_idf_weights,
    get_idf_w2v
)

def evaluate_predictions(y_true, y_pred):
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return p, r, f1

def run_dev():
    print("=" * 80)
    print("DEVELOPMENT PROTOCOL: Comparing Variants V0-V5 on Folds 1 & 2")
    print("SELECTION CRITERION: Validation F1 under Mode A (Label-Free Threshold)")
    print("STRICT RULE: Test labels are NEVER loaded or accessed during development.")
    print("=" * 80)
    
    sequences, y = load_data()
    splits = get_10fold_splits(y, seed=42)
    
    dev_splits = splits[:2] # Folds 1 and 2
    results = []
    
    # Store trained models/features for fold 1 & 2 to avoid redundant W2V training
    cache = {}
    
    for split in dev_splits:
        fold = split["fold"]
        print(f"\n{'='*40}\n>>> Preparing Data for Development Fold {fold} <<<\n{'='*40}")
        
        train_idx = split["train_idx"]
        val_norm_idx = split["val_norm_idx"]
        val_idx = split["val_idx"]
        test_idx = split["test_idx"]
        
        train_seqs = sequences[train_idx].tolist()
        val_norm_seqs = sequences[val_norm_idx].tolist()
        val_seqs = sequences[val_idx].tolist()
        test_seqs = sequences[test_idx].tolist()
        
        y_val = y[val_idx]
        
        print("Training Word2Vec on training normal sequences (workers=1, seed=42)...")
        t0 = time.time()
        w2v = train_w2v(train_seqs, vector_size=32, window=5, min_count=1, seed=42)
        print(f"Word2Vec trained in {time.time()-t0:.2f}s (Vocab size: {len(w2v.wv):,})")
        
        # OOV logging for test tokens (strictly lexical, no labels involved)
        total_tokens, oov_tokens, oov_rate = compute_oov_stats(test_seqs, w2v)
        print(f"Test Set OOV: {oov_tokens:,} / {total_tokens:,} tokens ({oov_rate*100:.3f}%)")
        
        # 1. Feature Extraction for V0 / V1 (Mean Pooling 32d)
        print("Extracting V0/V1 Mean-pooled features...")
        X_train_mean = get_mean_w2v(train_seqs, w2v)
        X_val_norm_mean = get_mean_w2v(val_norm_seqs, w2v)
        X_val_mean = get_mean_w2v(val_seqs, w2v)
        
        # 2. Feature Extraction for V2 (Mean + Max Pooling 64d)
        print("Extracting V2 Mean+Max pooled features...")
        X_train_mean_max = get_mean_max_w2v(train_seqs, w2v)
        X_val_norm_mean_max = get_mean_max_w2v(val_norm_seqs, w2v)
        X_val_mean_max = get_mean_max_w2v(val_seqs, w2v)
        
        # 3. Feature Extraction for V3 (IDF-weighted Mean Pooling 32d)
        print("Computing training IDF weights and extracting V3 features...")
        idf_dict = compute_idf_weights(train_seqs)
        X_train_idf = get_idf_w2v(train_seqs, w2v, idf_dict)
        X_val_norm_idf = get_idf_w2v(val_norm_seqs, w2v, idf_dict)
        X_val_idf = get_idf_w2v(val_seqs, w2v, idf_dict)
        
        cache[fold] = {
            "w2v": w2v,
            "oov_rate": oov_rate,
            "y_val": y_val,
            "mean": (X_train_mean, X_val_norm_mean, X_val_mean),
            "mean_max": (X_train_mean_max, X_val_norm_mean_max, X_val_mean_max),
            "idf": (X_train_idf, X_val_norm_idf, X_val_idf)
        }

    # =========================================================================
    # VARIANT 0: V0 (Mean 32d, AE 32-16-8-16-32, Pure POT without fallback)
    # =========================================================================
    print(f"\n{'='*60}\nEvaluating Variant 0: Baseline POT (No Fallback)\n{'='*60}")
    for fold in [1, 2]:
        c = cache[fold]
        X_tr, X_vn, X_v = c["mean"]
        y_val = c["y_val"]
        
        ae = AutoencoderModel(hidden_layer_sizes=(16, 8, 16), alpha=1e-4, random_state=42)
        t0 = time.time()
        ae.fit(X_tr)
        fit_time = time.time() - t0
        
        err_vn = ae.predict_errors(X_vn)
        err_v = ae.predict_errors(X_v)
        
        # Pure POT without safe bounds
        try:
            theta_a = float(compute_threshold_pot(err_vn, q=0.90, target_fpr=1e-3))
            fallback_used = False
        except Exception:
            theta_a = float(np.percentile(err_vn, 99.5))
            fallback_used = True
            
        y_pred_a = (err_v > theta_a).astype(int)
        p_a, r_a, f1_a = evaluate_predictions(y_val, y_pred_a)
        
        theta_b, f1_b = compute_threshold_mode_b(err_v, y_val)
        y_pred_b = (err_v > theta_b).astype(int)
        p_b, r_b, _ = evaluate_predictions(y_val, y_pred_b)
        
        print(f"Fold {fold} | Mode A: P={p_a*100:.2f}%, R={r_a*100:.2f}%, F1={f1_a*100:.2f}% (theta={theta_a:.4f}, fallback={fallback_used}) | Mode B: F1={f1_b*100:.2f}%")
        results.append({
            "variant": "V0", "description": "Mean 32d + AE (16,8,16) + Pure POT",
            "fold": fold, "fit_time_s": fit_time, "oov_rate": c["oov_rate"],
            "mode_a_theta": theta_a, "mode_a_fallback": fallback_used,
            "mode_a_p": p_a, "mode_a_r": r_a, "mode_a_f1": f1_a,
            "mode_b_theta": theta_b, "mode_b_p": p_b, "mode_b_r": r_b, "mode_b_f1": f1_b
        })

    # =========================================================================
    # VARIANT 1: V1 (Mean 32d, AE 32-16-8-16-32, POT + Auto Fallback 99.5th)
    # =========================================================================
    print(f"\n{'='*60}\nEvaluating Variant 1: POT + Auto Fallback\n{'='*60}")
    for fold in [1, 2]:
        c = cache[fold]
        X_tr, X_vn, X_v = c["mean"]
        y_val = c["y_val"]
        
        ae = AutoencoderModel(hidden_layer_sizes=(16, 8, 16), alpha=1e-4, random_state=42)
        t0 = time.time()
        ae.fit(X_tr)
        fit_time = time.time() - t0
        
        err_vn = ae.predict_errors(X_vn)
        err_v = ae.predict_errors(X_v)
        
        theta_a, fallback_used = compute_threshold_mode_a(err_vn, q=0.90, target_fpr=1e-3, fallback_percentile=99.5)
        y_pred_a = (err_v > theta_a).astype(int)
        p_a, r_a, f1_a = evaluate_predictions(y_val, y_pred_a)
        
        theta_b, f1_b = compute_threshold_mode_b(err_v, y_val)
        y_pred_b = (err_v > theta_b).astype(int)
        p_b, r_b, _ = evaluate_predictions(y_val, y_pred_b)
        
        print(f"Fold {fold} | Mode A: P={p_a*100:.2f}%, R={r_a*100:.2f}%, F1={f1_a*100:.2f}% (theta={theta_a:.4f}, fallback={fallback_used}) | Mode B: F1={f1_b*100:.2f}%")
        results.append({
            "variant": "V1", "description": "Mean 32d + AE (16,8,16) + Safe POT Fallback",
            "fold": fold, "fit_time_s": fit_time, "oov_rate": c["oov_rate"],
            "mode_a_theta": theta_a, "mode_a_fallback": fallback_used,
            "mode_a_p": p_a, "mode_a_r": r_a, "mode_a_f1": f1_a,
            "mode_b_theta": theta_b, "mode_b_p": p_b, "mode_b_r": r_b, "mode_b_f1": f1_b
        })

    # =========================================================================
    # VARIANT 2: V2 (Mean+Max 64d, AE 64-32-16-32-64, POT + Fallback)
    # =========================================================================
    print(f"\n{'='*60}\nEvaluating Variant 2: Mean + Max Pooling (64d)\n{'='*60}")
    for fold in [1, 2]:
        c = cache[fold]
        X_tr, X_vn, X_v = c["mean_max"]
        y_val = c["y_val"]
        
        ae = AutoencoderModel(hidden_layer_sizes=(32, 16, 32), alpha=1e-4, random_state=42)
        t0 = time.time()
        ae.fit(X_tr)
        fit_time = time.time() - t0
        
        err_vn = ae.predict_errors(X_vn)
        err_v = ae.predict_errors(X_v)
        
        theta_a, fallback_used = compute_threshold_mode_a(err_vn, q=0.90, target_fpr=1e-3, fallback_percentile=99.5)
        y_pred_a = (err_v > theta_a).astype(int)
        p_a, r_a, f1_a = evaluate_predictions(y_val, y_pred_a)
        
        theta_b, f1_b = compute_threshold_mode_b(err_v, y_val)
        y_pred_b = (err_v > theta_b).astype(int)
        p_b, r_b, _ = evaluate_predictions(y_val, y_pred_b)
        
        print(f"Fold {fold} | Mode A: P={p_a*100:.2f}%, R={r_a*100:.2f}%, F1={f1_a*100:.2f}% (theta={theta_a:.4f}, fallback={fallback_used}) | Mode B: F1={f1_b*100:.2f}%")
        results.append({
            "variant": "V2", "description": "Mean+Max 64d + AE (32,16,32)",
            "fold": fold, "fit_time_s": fit_time, "oov_rate": c["oov_rate"],
            "mode_a_theta": theta_a, "mode_a_fallback": fallback_used,
            "mode_a_p": p_a, "mode_a_r": r_a, "mode_a_f1": f1_a,
            "mode_b_theta": theta_b, "mode_b_p": p_b, "mode_b_r": r_b, "mode_b_f1": f1_b
        })

    # =========================================================================
    # VARIANT 3: V3 (IDF-weighted Mean 32d, AE 32-16-8-16-32, POT + Fallback)
    # =========================================================================
    print(f"\n{'='*60}\nEvaluating Variant 3: IDF-Weighted Mean Pooling (32d)\n{'='*60}")
    for fold in [1, 2]:
        c = cache[fold]
        X_tr, X_vn, X_v = c["idf"]
        y_val = c["y_val"]
        
        ae = AutoencoderModel(hidden_layer_sizes=(16, 8, 16), alpha=1e-4, random_state=42)
        t0 = time.time()
        ae.fit(X_tr)
        fit_time = time.time() - t0
        
        err_vn = ae.predict_errors(X_vn)
        err_v = ae.predict_errors(X_v)
        
        theta_a, fallback_used = compute_threshold_mode_a(err_vn, q=0.90, target_fpr=1e-3, fallback_percentile=99.5)
        y_pred_a = (err_v > theta_a).astype(int)
        p_a, r_a, f1_a = evaluate_predictions(y_val, y_pred_a)
        
        theta_b, f1_b = compute_threshold_mode_b(err_v, y_val)
        y_pred_b = (err_v > theta_b).astype(int)
        p_b, r_b, _ = evaluate_predictions(y_val, y_pred_b)
        
        print(f"Fold {fold} | Mode A: P={p_a*100:.2f}%, R={r_a*100:.2f}%, F1={f1_a*100:.2f}% (theta={theta_a:.4f}, fallback={fallback_used}) | Mode B: F1={f1_b*100:.2f}%")
        results.append({
            "variant": "V3", "description": "IDF-Weighted 32d + AE (16,8,16)",
            "fold": fold, "fit_time_s": fit_time, "oov_rate": c["oov_rate"],
            "mode_a_theta": theta_a, "mode_a_fallback": fallback_used,
            "mode_a_p": p_a, "mode_a_r": r_a, "mode_a_f1": f1_a,
            "mode_b_theta": theta_b, "mode_b_p": p_b, "mode_b_r": r_b, "mode_b_f1": f1_b
        })

    # Compare V1, V2, V3 mean validation F1 to pick feature representation for V4 & V5
    df_temp = pd.DataFrame(results)
    means = df_temp[df_temp["variant"].isin(["V1", "V2", "V3"])].groupby("variant")["mode_a_f1"].mean()
    best_feat_var = means.idxmax()
    print(f"\n>>> Best Feature Variant among V1-V3 is {best_feat_var} (Val F1: {means[best_feat_var]*100:.2f}%) <<<")
    
    # Feature key map
    feat_key = {"V1": "mean", "V2": "mean_max", "V3": "idf"}[best_feat_var]
    feat_dim = 64 if best_feat_var == "V2" else 32
    default_hidden = (32, 16, 32) if best_feat_var == "V2" else (16, 8, 16)

    # =========================================================================
    # VARIANT 4: V4 (Best Feature + 5-Seed Ensemble)
    # =========================================================================
    print(f"\n{'='*60}\nEvaluating Variant 4: 5-Seed Ensemble on {best_feat_var}\n{'='*60}")
    ensemble_seeds = [42, 43, 44, 45, 46]
    for fold in [1, 2]:
        c = cache[fold]
        X_tr, X_vn, X_v = c[feat_key]
        y_val = c["y_val"]
        
        errs_vn = []
        errs_v = []
        t0 = time.time()
        for seed in ensemble_seeds:
            ae = AutoencoderModel(hidden_layer_sizes=default_hidden, alpha=1e-4, random_state=seed)
            ae.fit(X_tr)
            errs_vn.append(ae.predict_errors(X_vn))
            errs_v.append(ae.predict_errors(X_v))
        fit_time = time.time() - t0
        
        # Average reconstruction errors across the 5 models
        err_vn_ens = np.mean(errs_vn, axis=0)
        err_v_ens = np.mean(errs_v, axis=0)
        
        theta_a, fallback_used = compute_threshold_mode_a(err_vn_ens, q=0.90, target_fpr=1e-3, fallback_percentile=99.5)
        y_pred_a = (err_v_ens > theta_a).astype(int)
        p_a, r_a, f1_a = evaluate_predictions(y_val, y_pred_a)
        
        theta_b, f1_b = compute_threshold_mode_b(err_v_ens, y_val)
        y_pred_b = (err_v_ens > theta_b).astype(int)
        p_b, r_b, _ = evaluate_predictions(y_val, y_pred_b)
        
        print(f"Fold {fold} | Mode A: P={p_a*100:.2f}%, R={r_a*100:.2f}%, F1={f1_a*100:.2f}% (theta={theta_a:.4f}, fallback={fallback_used}) | Mode B: F1={f1_b*100:.2f}%")
        results.append({
            "variant": "V4", "description": f"Ensemble 5 seeds on {best_feat_var}",
            "fold": fold, "fit_time_s": fit_time, "oov_rate": c["oov_rate"],
            "mode_a_theta": theta_a, "mode_a_fallback": fallback_used,
            "mode_a_p": p_a, "mode_a_r": r_a, "mode_a_f1": f1_a,
            "mode_b_theta": theta_b, "mode_b_p": p_b, "mode_b_r": r_b, "mode_b_f1": f1_b
        })

    # =========================================================================
    # VARIANT 5: V5 (Grid Search: Architectures & Alpha on best feature)
    # =========================================================================
    print(f"\n{'='*60}\nEvaluating Variant 5: Grid Search on {best_feat_var}\n{'='*60}")
    grid_archs = [(16, 8, 16), (32, 16, 32), (64, 16, 64)] if feat_dim == 32 else [(32, 16, 32), (64, 32, 64)]
    grid_alphas = [1e-4, 1e-3]
    
    for arch in grid_archs:
        for alpha in grid_alphas:
            v_name = f"V5_{arch}_{alpha}"
            print(f"\n--- Testing Architecture {arch}, alpha={alpha} ---")
            for fold in [1, 2]:
                c = cache[fold]
                X_tr, X_vn, X_v = c[feat_key]
                y_val = c["y_val"]
                
                ae = AutoencoderModel(hidden_layer_sizes=arch, alpha=alpha, random_state=42)
                t0 = time.time()
                ae.fit(X_tr)
                fit_time = time.time() - t0
                
                err_vn = ae.predict_errors(X_vn)
                err_v = ae.predict_errors(X_v)
                
                theta_a, fallback_used = compute_threshold_mode_a(err_vn, q=0.90, target_fpr=1e-3, fallback_percentile=99.5)
                y_pred_a = (err_v > theta_a).astype(int)
                p_a, r_a, f1_a = evaluate_predictions(y_val, y_pred_a)
                
                theta_b, f1_b = compute_threshold_mode_b(err_v, y_val)
                y_pred_b = (err_v > theta_b).astype(int)
                p_b, r_b, _ = evaluate_predictions(y_val, y_pred_b)
                
                print(f"Fold {fold} | Mode A: P={p_a*100:.2f}%, R={r_a*100:.2f}%, F1={f1_a*100:.2f}% (theta={theta_a:.4f}) | Mode B: F1={f1_b*100:.2f}%")
                results.append({
                    "variant": v_name, "description": f"Arch {arch}, alpha {alpha} on {best_feat_var}",
                    "fold": fold, "fit_time_s": fit_time, "oov_rate": c["oov_rate"],
                    "mode_a_theta": theta_a, "mode_a_fallback": fallback_used,
                    "mode_a_p": p_a, "mode_a_r": r_a, "mode_a_f1": f1_a,
                    "mode_b_theta": theta_b, "mode_b_p": p_b, "mode_b_r": r_b, "mode_b_f1": f1_b
                })

    df = pd.DataFrame(results)
    out_csv = "syslog_v2_clean/development_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nAll development evaluations saved to {out_csv}")
    
    # Summary Table
    print("\n" + "="*80)
    print("DEVELOPMENT SUMMARY (Averaged over Folds 1 & 2 on Validation Set)")
    print("="*80)
    summary = df.groupby(["variant", "description"])[["mode_a_p", "mode_a_r", "mode_a_f1", "mode_b_f1", "fit_time_s"]].mean().reset_index()
    summary = summary.sort_values(by="mode_a_f1", ascending=False)
    for _, row in summary.iterrows():
        print(f"| {row['variant']:<18s} | Mode A F1: {row['mode_a_f1']*100:6.2f}% (P: {row['mode_a_p']*100:5.2f}%, R: {row['mode_a_r']*100:5.2f}%) | Mode B F1: {row['mode_b_f1']*100:6.2f}% | Time: {row['fit_time_s']:5.1f}s | {row['description']}")
    print("="*80)
    
    best_overall = summary.iloc[0]["variant"]
    print(f"\nWINNER SELECTED BY VALIDATION F1 (Mode A): {best_overall}")

if __name__ == "__main__":
    run_dev()

