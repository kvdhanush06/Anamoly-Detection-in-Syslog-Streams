import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from gensim.models import Word2Vec

# Style configuration for academic reporting
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'font.family': 'sans-serif',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'figure.autolayout': True
})

OUTPUT_DIR = "report/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------------------------------------
# 1. Per-Fold Metrics (F1, PR-AUC, ROC-AUC)
# -------------------------------------------------------------
def plot_per_fold_metrics():
    df = pd.read_csv("syslog_v2_clean/test_results_10fold.csv")
    folds = df["fold"].values
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    
    # Subplot 1: Mode A vs Mode B F1
    width = 0.35
    x = np.arange(len(folds))
    rects1 = ax1.bar(x - width/2, df["mode_a_f1"] * 100, width, label="Mode A F1 (Label-Free POT)", color="#2b5c8f", alpha=0.9)
    rects2 = ax1.bar(x + width/2, df["mode_b_f1"] * 100, width, label="Mode B F1 (Val-Tuned)", color="#e07a5f", alpha=0.9)
    
    mean_a = df["mode_a_f1"].mean() * 100
    mean_b = df["mode_b_f1"].mean() * 100
    ax1.axhline(mean_a, color="#1d3f66", linestyle="--", linewidth=1.5, label=f"Mode A Mean ({mean_a:.2f}%)")
    ax1.axhline(mean_b, color="#b34b2f", linestyle=":", linewidth=1.5, label=f"Mode B Mean ({mean_b:.2f}%)")
    
    ax1.set_ylabel("F1-Score (%)")
    ax1.set_title("Per-Fold Anomaly Detection Performance (Natural Prevalence: 10.07%)")
    ax1.set_ylim(80, 95)
    ax1.legend(loc="lower right", framealpha=0.9)
    
    # Subplot 2: PR-AUC and ROC-AUC
    ax2.plot(x, df["roc_auc"] * 100, marker='o', linewidth=2, color="#3d405b", label=f"ROC-AUC (Mean: {df['roc_auc'].mean()*100:.2f}%)")
    ax2.plot(x, df["pr_auc"] * 100, marker='s', linewidth=2, color="#81b29a", label=f"PR-AUC (Mean: {df['pr_auc'].mean()*100:.2f}%)")
    
    ax2.set_xlabel("Cross-Validation Fold")
    ax2.set_ylabel("AUC Score (%)")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"Fold {f}" for f in folds])
    ax2.set_ylim(85, 100)
    ax2.legend(loc="lower right", framealpha=0.9)
    
    plt.savefig(os.path.join(OUTPUT_DIR, "per_fold_metrics.pdf"))
    plt.savefig(os.path.join(OUTPUT_DIR, "per_fold_metrics.png"), dpi=300)
    plt.close()
    print("Saved per_fold_metrics.(pdf/png)")

# -------------------------------------------------------------
# 2. Confusion Matrix Heatmaps (Mode A & Mode B Mean)
# -------------------------------------------------------------
def plot_confusion_matrices():
    df = pd.read_csv("syslog_v2_clean/test_results_10fold.csv")
    
    mean_a_cm = np.array([
        [df["mode_a_tn"].mean(), df["mode_a_fp"].mean()],
        [df["mode_a_fn"].mean(), df["mode_a_tp"].mean()]
    ])
    
    mean_b_cm = np.array([
        [df["mode_b_tn"].mean(), df["mode_b_fp"].mean()],
        [df["mode_b_fn"].mean(), df["mode_b_tp"].mean()]
    ])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    
    for ax, cm, title, p, r, f1 in [
        (ax1, mean_a_cm, "Mode A (Label-Free POT)", df["mode_a_p"].mean()*100, df["mode_a_r"].mean()*100, df["mode_a_f1"].mean()*100),
        (ax2, mean_b_cm, "Mode B (Val-Tuned)", df["mode_b_p"].mean()*100, df["mode_b_r"].mean()*100, df["mode_b_f1"].mean()*100)
    ]:
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title(f"{title}\nP: {p:.1f}%, R: {r:.1f}%, F1: {f1:.1f}%")
        tick_marks = np.arange(2)
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(['Pred Normal', 'Pred Anomaly'])
        ax.set_yticklabels(['True Normal', 'True Anomaly'])
        
        # Annotate numbers
        thresh = cm.max() / 2.0
        for i in range(2):
            for j in range(2):
                val = cm[i, j]
                ax.text(j, i, f"{int(round(val)):,}\n({val/cm.sum()*100:.1f}%)",
                        ha="center", va="center",
                        color="white" if val > thresh else "black",
                        fontsize=11, fontweight='bold')
        ax.set_ylabel('Ground Truth')
        ax.set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix_heatmap.pdf"))
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix_heatmap.png"), dpi=300)
    plt.close()
    print("Saved confusion_matrix_heatmap.(pdf/png)")

# -------------------------------------------------------------
# 3. Reconstruction Error Distribution (Normal vs Anomalous)
# -------------------------------------------------------------
def plot_reconstruction_error():
    sys.path.append("syslog_v2_clean")
    from data_utils import load_data, get_10fold_splits
    from feature_utils import get_mean_max_w2v
    
    sequences, y = load_data()
    splits = get_10fold_splits(y)
    fold1 = splits[0]
    
    with open("syslog_v2_clean/saved_models/bgl_ae_clean_fold_1.pkl", "rb") as f:
        ae = pickle.load(f)
    w2v = Word2Vec.load("syslog_v2_clean/saved_models/bgl_w2v_clean_fold_1.pkl")
    
    # Subsample 10k normals and 5k anomalies from test set
    norm_idx = fold1["test_norm_idx"][:10000]
    anom_idx = fold1["test_anom_idx"][:5000]
    
    X_norm = get_mean_max_w2v([sequences[i] for i in norm_idx], w2v)
    X_anom = get_mean_max_w2v([sequences[i] for i in anom_idx], w2v)
    
    err_norm = ae.predict_errors(X_norm)
    err_anom = ae.predict_errors(X_anom)
    
    # Read fold 1 thresholds from CSV
    df = pd.read_csv("syslog_v2_clean/test_results_10fold.csv")
    theta_a = df.loc[df["fold"] == 1, "mode_a_theta"].values[0]
    theta_b = df.loc[df["fold"] == 1, "mode_b_theta"].values[0]
    
    plt.figure(figsize=(9, 5))
    bins = np.linspace(0, 1.2, 80)
    plt.hist(err_norm, bins=bins, density=True, alpha=0.65, color="#2b5c8f", label=f"Normal Windows (Test, N=10,000)")
    plt.hist(err_anom, bins=bins, density=True, alpha=0.65, color="#e07a5f", label=f"Anomalous Windows (Test, N=5,000)")
    
    plt.axvline(theta_a, color="#1d3f66", linestyle="--", linewidth=2.0, label=f"Threshold $\\theta_A$ (POT: {theta_a:.4f})")
    plt.axvline(theta_b, color="#b34b2f", linestyle=":", linewidth=2.0, label=f"Threshold $\\theta_B$ (Val-Tuned: {theta_b:.4f})")
    
    plt.xlabel("Reconstruction Error ($L_2$ Norm: $\\|x - \\hat{x}\\|_2$)")
    plt.ylabel("Probability Density")
    plt.title("Separation of Reconstruction Errors on Fold 1 Test Set")
    plt.xlim(0, 1.2)
    plt.legend(loc="upper right", framealpha=0.95)
    
    plt.savefig(os.path.join(OUTPUT_DIR, "reconstruction_error_hist.pdf"))
    plt.savefig(os.path.join(OUTPUT_DIR, "reconstruction_error_hist.png"), dpi=300)
    plt.close()
    print("Saved reconstruction_error_hist.(pdf/png)")

# -------------------------------------------------------------
# 4. Model Ablation Study Bar Chart
# -------------------------------------------------------------
def plot_ablation_comparison():
    df = pd.read_csv("syslog_v2_clean/ablation_results.csv")
    models = ["MLP Autoencoder", "PCA (k=48)", "Isolation Forest"]
    
    # We aggregate mean per model
    agg = df.groupby("model").agg({
        "mode_a_f1": "mean",
        "mode_b_f1": "mean",
        "pr_auc": "mean",
        "roc_auc": "mean"
    }).loc[["PCA (k=48)", "Isolation Forest"]]
    
    # Add Autoencoder from test_results_10fold.csv
    df_ae = pd.read_csv("syslog_v2_clean/test_results_10fold.csv")
    ae_row = pd.Series({
        "mode_a_f1": df_ae["mode_a_f1"].mean(),
        "mode_b_f1": df_ae["mode_b_f1"].mean(),
        "pr_auc": df_ae["pr_auc"].mean(),
        "roc_auc": df_ae["roc_auc"].mean()
    }, name="MLP Autoencoder")
    
    plot_df = pd.concat([pd.DataFrame([ae_row]), agg])
    
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(plot_df))
    width = 0.20
    
    r1 = ax.bar(x - 1.5*width, plot_df["mode_a_f1"] * 100, width, label="Mode A F1 (POT)", color="#2b5c8f")
    r2 = ax.bar(x - 0.5*width, plot_df["mode_b_f1"] * 100, width, label="Mode B F1 (Val-Tuned)", color="#457b9d")
    r3 = ax.bar(x + 0.5*width, plot_df["pr_auc"] * 100, width, label="PR-AUC", color="#e07a5f")
    r4 = ax.bar(x + 1.5*width, plot_df["roc_auc"] * 100, width, label="ROC-AUC", color="#3d405b")
    
    ax.set_ylabel("Metric Score (%)")
    ax.set_title("Architectural Ablation on Identical 64-d Word2Vec Features")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df.index, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right", framealpha=0.95, ncol=2)
    
    # Add numerical labels
    for rects in [r1, r2, r3, r4]:
        for rect in rects:
            h = rect.get_height()
            if h > 3:
                ax.annotate(f"{h:.1f}%",
                            xy=(rect.get_x() + rect.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "ablation_comparison.pdf"))
    plt.savefig(os.path.join(OUTPUT_DIR, "ablation_comparison.png"), dpi=300)
    plt.close()
    print("Saved ablation_comparison.(pdf/png)")

# -------------------------------------------------------------
# 5. Chronological Split vs Random 10-Fold
# -------------------------------------------------------------
def plot_chronological_comparison():
    df_chrono = pd.read_csv("syslog_v2_clean/chronological_results.csv")
    df_10f = pd.read_csv("syslog_v2_clean/test_results_10fold.csv")
    
    categories = ["OOV Token Rate", "Mode A F1", "Mode B F1", "PR-AUC", "ROC-AUC"]
    random_cv_vals = [
        df_10f["test_oov_rate"].mean() * 100,
        df_10f["mode_a_f1"].mean() * 100,
        df_10f["mode_b_f1"].mean() * 100,
        df_10f["pr_auc"].mean() * 100,
        df_10f["roc_auc"].mean() * 100
    ]
    chrono_vals = [
        df_chrono["oov_rate"].values[0] * 100,
        df_chrono["mode_a_f1"].values[0] * 100,
        df_chrono["mode_b_f1"].values[0] * 100,
        df_chrono["pr_auc"].values[0] * 100,
        df_chrono["roc_auc"].values[0] * 100
    ]
    
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(categories))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, random_cv_vals, width, label="10-Fold Stratified Split (Random CV)", color="#2b5c8f")
    rects2 = ax.bar(x + width/2, chrono_vals, width, label="Chronological Split (70% Train / 10% Val / 20% Test)", color="#e07a5f")
    
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Temporal Generalization vs Stratified Cross-Validation")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontweight='bold')
    ax.set_ylim(0, 110)
    ax.legend(loc="upper right", framealpha=0.95)
    
    for rects in [rects1, rects2]:
        for rect in rects:
            h = rect.get_height()
            ax.annotate(f"{h:.1f}%",
                        xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8.5)
                        
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "chronological_comparison.pdf"))
    plt.savefig(os.path.join(OUTPUT_DIR, "chronological_comparison.png"), dpi=300)
    plt.close()
    print("Saved chronological_comparison.(pdf/png)")

# -------------------------------------------------------------
# 6. Prevalence Impact (Natural vs Balanced View)
# -------------------------------------------------------------
def plot_prevalence_impact():
    df_nat = pd.read_csv("syslog_v2_clean/test_results_10fold.csv")
    df_bal = pd.read_csv("syslog_v2_clean/test_balanced_results_10fold.csv")
    
    metrics = ["Mode A Prec", "Mode A Rec", "Mode A F1", "Mode B Prec", "Mode B Rec", "Mode B F1"]
    nat_vals = [
        df_nat["mode_a_p"].mean() * 100,
        df_nat["mode_a_r"].mean() * 100,
        df_nat["mode_a_f1"].mean() * 100,
        df_nat["mode_b_p"].mean() * 100,
        df_nat["mode_b_r"].mean() * 100,
        df_nat["mode_b_f1"].mean() * 100
    ]
    bal_vals = [
        df_bal["mode_a_p"].mean() * 100,
        df_bal["mode_a_r"].mean() * 100,
        df_bal["mode_a_f1"].mean() * 100,
        df_bal["mode_b_p"].mean() * 100,
        df_bal["mode_b_r"].mean() * 100,
        df_bal["mode_b_f1"].mean() * 100
    ]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(metrics))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, nat_vals, width, label="Natural Prevalence (10.07% Anomalies)", color="#2b5c8f")
    rects2 = ax.bar(x + width/2, bal_vals, width, label="Balanced View (50.0% Anomalies)", color="#e07a5f")
    
    ax.set_ylabel("Score (%)")
    ax.set_title("Prevalence Effect: Impact of Anomaly Base-Rate on Operational Metrics")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontweight='bold')
    ax.set_ylim(70, 105)
    ax.legend(loc="lower right", framealpha=0.95)
    
    for rects in [rects1, rects2]:
        for rect in rects:
            h = rect.get_height()
            ax.annotate(f"{h:.1f}%",
                        xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8.5)
                        
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "prevalence_impact.pdf"))
    plt.savefig(os.path.join(OUTPUT_DIR, "prevalence_impact.png"), dpi=300)
    plt.close()
    print("Saved prevalence_impact.(pdf/png)")

if __name__ == "__main__":
    plot_per_fold_metrics()
    plot_confusion_matrices()
    plot_reconstruction_error()
    plot_ablation_comparison()
    plot_chronological_comparison()
    plot_prevalence_impact()
    print("All figures successfully generated in report/figures/!")

