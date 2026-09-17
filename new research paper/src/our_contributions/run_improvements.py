"""
Our Novel Contributions & Improvements over BertGCN on REAL Zenodo Datasets:
1. Edge-Weighted Graph Attention (EW-GAT):
   Directly incorporates log-scaled event transition counts log(1 + w_ij) into attention heads.
2. Multi-Scale Hierarchical Readout:
   Concatenates [Mean || Max || Std] graph-level poolings followed by LayerNorm + GELU MLP.
3. Class-Balanced Focal Loss & Dynamic Threshold Calibration:
   Prevents overconfidence on duplicated oversampled samples and maximizes F1 score.

Saves trained enhanced models into `new research paper/saved_models_our_contributions/`.

How to run:
    conda run -n torch-env python "new research paper/src/our_contributions/run_improvements.py"
"""
import os
import sys
import torch
import numpy as np
from typing import List

# ── Path setup ────────────────────────────────────────────────────────────────
# This file: new research paper/src/our_contributions/run_improvements.py
# We add src/ to sys.path so both baseline_replication and our_contributions
# are importable as packages.
CURRENT_DIR     = os.path.dirname(os.path.abspath(__file__))    # our_contributions/
SRC_DIR         = os.path.dirname(CURRENT_DIR)                  # src/
PAPER_DIR       = os.path.dirname(SRC_DIR)                      # new research paper/
SAVED_MODELS_DIR = os.path.join(PAPER_DIR, "saved_models_our_contributions")
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from baseline_replication.data.preprocess import LogPreprocessor
from baseline_replication.data.graph_builder import GraphBuilder, LogGraph
from baseline_replication.data.real_data_loader import load_hdfs_real_sequences, load_bgl_real_sequences
from baseline_replication.models.bert_embedder import BertEmbedder
from baseline_replication.train import train_model, evaluate_model, oversample_graphs
from baseline_replication.evaluate import print_metrics_table
from our_contributions.models.enhanced_gnn import EnhancedLogGNN


def run_improvements_for_dataset(
    dataset_name: str = "HDFS",
    epochs: int = 15,
    device: str = None,
    max_samples: int = 1000,
    window_size: int = 20,
):
    print(f"\n{'='*80}")
    print(f" Our Contributions: Enhanced EW-GAT + Multi-Scale Readout on REAL {dataset_name}")
    print(f"{'='*80}")

    preprocessor = LogPreprocessor()
    embedder = BertEmbedder(device=device)
    builder = GraphBuilder(embedder)

    if dataset_name.upper() == "HDFS":
        raw_sequences = load_hdfs_real_sequences(max_blocks=max_samples)
    elif dataset_name.upper() == "BGL":
        raw_sequences = load_bgl_real_sequences(window_size=window_size, max_windows=max_samples)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Expected 'HDFS' or 'BGL'.")

    print("Constructing directed weighted graphs with BERT contextual embeddings...")
    all_graphs: List[LogGraph] = []
    for seq, label in raw_sequences:
        cleaned_seq = [preprocessor.clean_log_message(msg) for msg in seq]
        graph = builder.build_graph_from_sequence(cleaned_seq, label)
        all_graphs.append(graph)

    split_idx = int(len(all_graphs) * 0.7)
    train_graphs_raw = all_graphs[:split_idx]
    test_graphs = all_graphs[split_idx:]

    train_graphs_os = oversample_graphs(train_graphs_raw)
    print(f"Total Graphs: {len(all_graphs)} | Train (balanced): {len(train_graphs_os)} | Test: {len(test_graphs)}")

    print("\n>>> Training Our Enhanced EW-GAT Model with Focal Loss...")
    enhanced_model = EnhancedLogGNN(in_dim=768, hidden_dim=128, num_classes=2, num_layers=2, dropout=0.2)
    train_model(enhanced_model, train_graphs_os, epochs=epochs, lr=0.001, use_focal_loss=True, device=device)

    metrics_enhanced = evaluate_model(enhanced_model, test_graphs, threshold=0.45, device=device)
    print_metrics_table(f"Our Enhanced Model (EW-GAT + Multi-Readout + Focal Loss) - {dataset_name}", metrics_enhanced)

    save_path = os.path.join(SAVED_MODELS_DIR, f"{dataset_name.lower()}_enhanced_gat_model.pt")
    torch.save(enhanced_model.state_dict(), save_path)
    print(f"Saved enhanced model to {save_path}")

    return {
        "dataset": dataset_name,
        "enhanced_metrics": metrics_enhanced
    }


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {dev}")
    hdfs_res = run_improvements_for_dataset("HDFS", epochs=10, device=dev, max_samples=1000)
    bgl_res  = run_improvements_for_dataset("BGL",  epochs=10, device=dev, max_samples=1000, window_size=20)
