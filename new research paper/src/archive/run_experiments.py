"""
Main Experiment Runner — ARCHIVE (predecessor of run_baseline.py + run_improvements.py).
New Research Paper: BertGCN Log-Based Anomaly Detection (Alsalmi et al., CMC 2025/2026)

This script was the original combined runner BEFORE the codebase was split into:
  - src/baseline_replication/run_baseline.py    (Baseline BertGCN replication)
  - src/our_contributions/run_improvements.py   (Our enhanced EW-GAT contributions)

It is kept here for reference and reproducibility auditing.
The ACTIVE scripts to run are the two above — not this file.

Previously this used generate_benchmark_sequences() with SYNTHETIC data.
It now loads REAL Zenodo HDFS and BGL datasets via real_data_loader.py,
matching the methodology of the active run_baseline.py and run_improvements.py.
"""
import os
import sys
import torch
import numpy as np
import random
from typing import List

# ── Path setup ────────────────────────────────────────────────────────────────
# This file lives at: new research paper/src/archive/run_experiments.py
# We need:  new research paper/src/  on sys.path for package imports.
current_dir = os.path.dirname(os.path.abspath(__file__))   # archive/
src_dir     = os.path.dirname(current_dir)                 # src/
sys.path.insert(0, src_dir)

from baseline_replication.data.preprocess import LogPreprocessor
from baseline_replication.data.graph_builder import GraphBuilder, LogGraph
from baseline_replication.data.real_data_loader import load_hdfs_real_sequences, load_bgl_real_sequences
from baseline_replication.models.bert_embedder import BertEmbedder
from baseline_replication.models.gcn import BertGCN
from our_contributions.models.enhanced_gnn import EnhancedLogGNN
from baseline_replication.train import train_model, evaluate_model, oversample_graphs
from baseline_replication.evaluate import print_metrics_table


def run_experiment_on_dataset(
    dataset_name: str = "HDFS",
    epochs: int = 15,
    device: str = None,
    max_samples: int = 1000,
    window_size: int = 20,
):
    print(f"\n{'#'*80}")
    print(f" EXPERIMENT ON REAL ZENODO DATASET: {dataset_name}")
    print(f"{'#'*80}")

    # 1. Initialize Preprocessor and Embedder
    preprocessor = LogPreprocessor()
    embedder = BertEmbedder(device=device)
    builder = GraphBuilder(embedder)

    # 2. Load REAL Zenodo sequences
    print(f"Loading REAL Zenodo sequences for {dataset_name}...")
    if dataset_name.upper() == "HDFS":
        raw_sequences = load_hdfs_real_sequences(max_blocks=max_samples)
    elif dataset_name.upper() == "BGL":
        raw_sequences = load_bgl_real_sequences(window_size=window_size, max_windows=max_samples)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Expected 'HDFS' or 'BGL'.")

    # 3. Clean and construct Log Graphs
    print("Constructing directed weighted graphs with BERT contextual embeddings...")
    all_graphs: List[LogGraph] = []
    for seq, label in raw_sequences:
        cleaned_seq = [preprocessor.clean_log_message(msg) for msg in seq]
        graph = builder.build_graph_from_sequence(cleaned_seq, label)
        all_graphs.append(graph)

    # 4. 70:30 Train-Test split
    split_idx = int(len(all_graphs) * 0.7)
    train_graphs_raw = all_graphs[:split_idx]
    test_graphs = all_graphs[split_idx:]

    print(f"Total Graphs: {len(all_graphs)} | Train: {len(train_graphs_raw)} | Test: {len(test_graphs)}")
    train_anoms = sum(1 for g in train_graphs_raw if g.label == 1)
    test_anoms = sum(1 for g in test_graphs if g.label == 1)
    print(f"Train Anomalies: {train_anoms}/{len(train_graphs_raw)} | Test Anomalies: {test_anoms}/{len(test_graphs)}")

    # ── Experiment 1: Baseline BertGCN WITHOUT Oversampling (Table 3 of paper) ──
    print("\n>>> [1/3] Training Baseline BertGCN WITHOUT Oversampling...")
    model_baseline_no_os = BertGCN(in_dim=768, hidden_dim=128, num_classes=2, num_layers=2, readout="mean")
    train_model(model_baseline_no_os, list(train_graphs_raw), epochs=epochs, lr=0.001, device=device)
    metrics_no_os = evaluate_model(model_baseline_no_os, test_graphs, device=device)
    print_metrics_table(f"Baseline BertGCN (Before Oversampling) on {dataset_name}", metrics_no_os)

    # ── Experiment 2: Baseline BertGCN WITH Oversampling (Paper Replication) ───
    print("\n>>> [2/3] Training Baseline BertGCN WITH Oversampling...")
    train_graphs_os = oversample_graphs(train_graphs_raw)
    print(f"Train set size after oversampling: {len(train_graphs_os)} (Balanced 50/50)")

    model_baseline_os = BertGCN(in_dim=768, hidden_dim=128, num_classes=2, num_layers=2, readout="mean")
    train_model(model_baseline_os, train_graphs_os, epochs=epochs, lr=0.001, device=device)
    metrics_os = evaluate_model(model_baseline_os, test_graphs, device=device)
    print_metrics_table(f"Baseline BertGCN (After Oversampling) on {dataset_name}", metrics_os)

    # ── Experiment 3: Our Enhanced Architecture ──────────────────────────────
    print("\n>>> [3/3] Training Our Proposed Enhanced Architecture (EW-GAT + Multi-Readout)...")
    enhanced_model = EnhancedLogGNN(in_dim=768, hidden_dim=128, num_classes=2, num_layers=2, dropout=0.2)
    train_model(enhanced_model, train_graphs_os, epochs=epochs, lr=0.001, use_focal_loss=True, device=device)
    metrics_enhanced = evaluate_model(enhanced_model, test_graphs, threshold=0.45, device=device)
    print_metrics_table(f"Our Enhanced Model (EW-GAT + Multi-Readout) on {dataset_name}", metrics_enhanced)

    return {
        "dataset": dataset_name,
        "baseline_no_os": metrics_no_os,
        "baseline_os": metrics_os,
        "enhanced": metrics_enhanced
    }


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on Device: {device}")
    print("NOTE: This is an ARCHIVE script. Use run_baseline.py and run_improvements.py instead.\n")

    # Run on both HDFS and BGL using real Zenodo data
    hdfs_results = run_experiment_on_dataset("HDFS", epochs=15, device=device, max_samples=1000)
    bgl_results  = run_experiment_on_dataset("BGL",  epochs=15, device=device, max_samples=1000, window_size=20)

    print("\n" + "="*80)
    print(" SUMMARY BENCHMARK COMPARISON TABLE (REAL ZENODO DATA)")
    print("="*80)
    print(f"{'Dataset':<8} | {'Model':<38} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<8} | {'F1-Score':<8}")
    print("-" * 92)
    for res in [hdfs_results, bgl_results]:
        ds   = res["dataset"]
        m_no = res["baseline_no_os"]
        m_os = res["baseline_os"]
        m_en = res["enhanced"]
        print(f"{ds:<8} | {'BertGCN (Before Oversampling)':<38} | {m_no['accuracy']*100:6.2f}%    | {m_no['precision']*100:6.2f}%    | {m_no['recall']*100:5.2f}%  | {m_no['f1']*100:5.2f}%")
        print(f"{ds:<8} | {'BertGCN (Paper Replication)':<38} | {m_os['accuracy']*100:6.2f}%    | {m_os['precision']*100:6.2f}%    | {m_os['recall']*100:5.2f}%  | {m_os['f1']*100:5.2f}%")
        print(f"{ds:<8} | {'Our Enhanced Architecture (EW-GAT)':<38} | {m_en['accuracy']*100:6.2f}%    | {m_en['precision']*100:6.2f}%    | {m_en['recall']*100:5.2f}%  | {m_en['f1']*100:5.2f}%")
        print("-" * 92)
