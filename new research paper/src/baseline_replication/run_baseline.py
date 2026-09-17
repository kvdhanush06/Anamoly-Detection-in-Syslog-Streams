"""
Baseline Replication for New Research Paper:
BertGCN: Log-Based Anomaly Detection of System Logs Using Graph Neural Network
(Eman Alsalmi, Abeer Alhuzali, Areej Alhothali - CMC Dec 2025 / 2026)

Runs on the REAL Zenodo HDFS & BGL datasets extracted in `data/`:
1. Performs parsing-free cleaning & WordPiece tokenization (Sec 4.2).
2. Computes contextual BERT embeddings (768-dim) for log events (Sec 4.4).
3. Constructs directed, weighted session transition graphs with self-loops on starting events (Sec 4.3).
4. Evaluates baseline BertGCN with and without minority oversampling (Table 3 replication).
5. Saves trained baseline models into `new research paper/saved_models/`.

How to run:
    conda run -n torch-env python "new research paper/src/baseline_replication/run_baseline.py"
"""
import os
import sys
import torch
import numpy as np
import random
from typing import List

# ── Path setup ────────────────────────────────────────────────────────────────
# This file: new research paper/src/baseline_replication/run_baseline.py
# We add src/ to sys.path so that package imports work correctly.
CURRENT_DIR     = os.path.dirname(os.path.abspath(__file__))   # baseline_replication/
SRC_DIR         = os.path.dirname(CURRENT_DIR)                 # src/
PAPER_DIR       = os.path.dirname(SRC_DIR)                     # new research paper/
SAVED_MODELS_DIR = os.path.join(PAPER_DIR, "saved_models")
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

# Insert src/ so `baseline_replication.*` is importable as a package
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from baseline_replication.data.preprocess import LogPreprocessor
from baseline_replication.data.graph_builder import GraphBuilder, LogGraph
from baseline_replication.data.real_data_loader import load_hdfs_real_sequences, load_bgl_real_sequences
from baseline_replication.models.bert_embedder import BertEmbedder
from baseline_replication.models.gcn import BertGCN
from baseline_replication.train import train_model, evaluate_model, oversample_graphs
from baseline_replication.evaluate import print_metrics_table


def run_baseline_for_dataset(
    dataset_name: str = "HDFS",
    epochs: int = 15,
    device: str = None,
    max_samples: int = 1000,
    window_size: int = 20,
):
    print(f"\n{'='*80}")
    print(f" Baseline Replication: BertGCN on REAL {dataset_name} Dataset")
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

    print(f"Total Graphs: {len(all_graphs)} | Train: {len(train_graphs_raw)} | Test: {len(test_graphs)}")
    train_anoms = sum(1 for g in train_graphs_raw if g.label == 1)
    print(f"Train Anomalies: {train_anoms}/{len(train_graphs_raw)}")

    # ── 1. Baseline WITHOUT Oversampling (Table 3 — before oversampling) ──────
    print("\n>>> [1/2] Training Baseline BertGCN BEFORE Oversampling...")
    model_no_os = BertGCN(in_dim=768, hidden_dim=128, num_classes=2, num_layers=2, readout="mean")
    train_model(model_no_os, list(train_graphs_raw), epochs=epochs, lr=0.001, device=device)
    metrics_no_os = evaluate_model(model_no_os, test_graphs, device=device)
    print_metrics_table(f"Baseline BertGCN (Before Oversampling) - {dataset_name}", metrics_no_os)

    # ── 2. Baseline WITH Oversampling (Table 3 — after oversampling / benchmark) ─
    print("\n>>> [2/2] Training Baseline BertGCN AFTER Oversampling (Paper Replication)...")
    train_graphs_os = oversample_graphs(train_graphs_raw)
    model_os = BertGCN(in_dim=768, hidden_dim=128, num_classes=2, num_layers=2, readout="mean")
    train_model(model_os, train_graphs_os, epochs=epochs, lr=0.001, device=device)
    metrics_os = evaluate_model(model_os, test_graphs, device=device)
    print_metrics_table(f"Baseline BertGCN (After Oversampling) - {dataset_name}", metrics_os)

    # Save trained baseline model
    save_path = os.path.join(SAVED_MODELS_DIR, f"{dataset_name.lower()}_bert_gcn_baseline.pt")
    torch.save(model_os.state_dict(), save_path)
    print(f"Saved baseline model to {save_path}")

    return {
        "dataset": dataset_name,
        "before_oversampling": metrics_no_os,
        "after_oversampling": metrics_os
    }


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {dev}")
    hdfs_res = run_baseline_for_dataset("HDFS", epochs=10, device=dev, max_samples=1000)
    bgl_res  = run_baseline_for_dataset("BGL",  epochs=10, device=dev, max_samples=1000, window_size=20)
