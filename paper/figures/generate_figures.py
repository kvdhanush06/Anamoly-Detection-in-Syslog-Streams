import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set academic IEEE styling
plt.rcParams.update({
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 8.5,
    'ytick.labelsize': 8.5,
    'legend.fontsize': 8.5,
    'figure.titlesize': 11,
    'font.family': 'sans-serif',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'figure.autolayout': True
})

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

def generate_ablation_plot():
    # Data from syslog_v2_clean/ablation_results.csv and test_results_10fold.csv
    models = ['MLP AE (Ours)', 'PCA (k=48)', 'Isolation Forest']
    mode_a_f1 = [87.80, 59.97, 2.07]
    mode_b_f1 = [88.66, 85.50, 55.94]
    pr_auc = [91.05, 84.26, 48.20]
    
    x = np.arange(len(models))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(4.8, 2.8))
    rects1 = ax.bar(x - width, mode_a_f1, width, label='Mode A F1 (POT)', color='#1f77b4', edgecolor='black', linewidth=0.5)
    rects2 = ax.bar(x, mode_b_f1, width, label='Mode B F1 (Val-Tuned)', color='#ff7f0e', edgecolor='black', linewidth=0.5)
    rects3 = ax.bar(x + width, pr_auc, width, label='PR-AUC', color='#2ca02c', edgecolor='black', linewidth=0.5)
    
    ax.set_ylabel('Score (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 105)
    ax.legend(loc='upper right', framealpha=0.95)
    ax.set_title('BGL Model Comparison on Identical 64-d Features')
    
    # Annotate bars
    for rects in [rects1, rects2, rects3]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 2),  # 2 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=7.5)
                        
    plt.savefig(os.path.join(OUT_DIR, "ablation_comparison.pdf"))
    plt.savefig(os.path.join(OUT_DIR, "ablation_comparison.png"), dpi=300)
    plt.close()
    print("Saved ablation_comparison.pdf")

def generate_chronological_plot():
    categories = ['CV (10-Fold)', 'Chronological (70/10/20)']
    mode_a_f1 = [87.80, 0.03]
    mode_b_f1 = [88.66, 45.28]
    pr_auc = [91.05, 30.32]
    
    x = np.arange(len(categories))
    width = 0.24
    
    fig, ax = plt.subplots(figsize=(4.5, 2.6))
    rects1 = ax.bar(x - width, mode_a_f1, width, label='Mode A F1', color='#2b5c8f', edgecolor='black', linewidth=0.5)
    rects2 = ax.bar(x, mode_b_f1, width, label='Mode B F1', color='#e07a5f', edgecolor='black', linewidth=0.5)
    rects3 = ax.bar(x + width, pr_auc, width, label='PR-AUC', color='#81b29a', edgecolor='black', linewidth=0.5)
    
    ax.set_ylabel('Metric (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 105)
    ax.legend(loc='upper right', framealpha=0.95)
    ax.set_title('Impact of Concept Drift & 68.9% OOV on BGL')
    
    for rects in [rects1, rects2, rects3]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 2),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=7.5)
                        
    plt.savefig(os.path.join(OUT_DIR, "chronological_comparison.pdf"))
    plt.savefig(os.path.join(OUT_DIR, "chronological_comparison.png"), dpi=300)
    plt.close()
    print("Saved chronological_comparison.pdf")

def generate_prevalence_plot():
    # Natural (10.07%) vs Balanced View (50.0%)
    categories = ['Natural (10.07% Anom)', 'Balanced View (50% Anom)']
    prec = [93.76, 99.26]
    rec = [82.61, 82.61]
    f1 = [87.80, 90.16]
    
    x = np.arange(len(categories))
    width = 0.24
    
    fig, ax = plt.subplots(figsize=(4.5, 2.6))
    rects1 = ax.bar(x - width, prec, width, label='Precision', color='#3d405b', edgecolor='black', linewidth=0.5)
    rects2 = ax.bar(x, rec, width, label='Recall', color='#e07a5f', edgecolor='black', linewidth=0.5)
    rects3 = ax.bar(x + width, f1, width, label='F1-Score', color='#81b29a', edgecolor='black', linewidth=0.5)
    
    ax.set_ylabel('Metric (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(70, 105)
    ax.legend(loc='lower left', framealpha=0.95)
    ax.set_title('Base-Rate Effect: Precision Inflation from 10% to 50% Prevalence')
    
    for rects in [rects1, rects2, rects3]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 2),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=7.5)
                        
    plt.savefig(os.path.join(OUT_DIR, "prevalence_impact.pdf"))
    plt.savefig(os.path.join(OUT_DIR, "prevalence_impact.png"), dpi=300)
    plt.close()
    print("Saved prevalence_impact.pdf")

if __name__ == '__main__':
    generate_ablation_plot()
    generate_chronological_plot()
    generate_prevalence_plot()
    print("All figures successfully generated in paper/figures/")

