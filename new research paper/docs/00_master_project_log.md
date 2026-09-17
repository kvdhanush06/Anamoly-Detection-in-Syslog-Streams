# Master Project Log: BertGCN Log-Based Anomaly Detection Using Graph Neural Networks
**Base Paper:** "Log-Based Anomaly Detection of System Logs Using Graph Neural Network" (Eman Alsalmi, Abeer Alhuzali, Areej Alhothali - CMC Dec 2025 / 2026)

This document tracks the methodology, experimental setups, replication benchmarks, and novel contributions.

---

## 1. Methodology Breakdown

### A. Baseline Reproduction (`src/baseline_replication/`)
- **Parsing-Free Preprocessing (`data/preprocess.py`)**: Eliminates log parsers by stripping non-character tokens, removing numerical parameters, converting to lowercase, and performing BERT WordPiece tokenization.
- **Contextual Semantic Embeddings (`models/bert_embedder.py`)**: Encodes unique log events into dense 768-dimensional contextual vectors via `bert-base-uncased`.
- **Directed, Weighted Graph Construction (`data/graph_builder.py`)**: Models each session as a directed graph $G = (V, E, X_v, F_w)$ with transition count edge weights and self-loops on the initial node.
- **GCN Message Passing & Readout (`models/gcn.py`)**: Symmetric normalized adjacency convolution:
  $$H^{(l+1)} = \sigma(\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H^{(l)} W^{(l)})$$
  followed by global readout pooling and Softmax binary classification.
- **Imbalance Handling (`train.py`)**: Random minority oversampling to balance normal and anomalous training sessions.
- **Execution Script**: `src/baseline_replication/run_baseline.py`

### B. Target Paper Numbers (Table 3 & Table 4/6)
| Dataset | Oversampling | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|---|
| **HDFS** | Before | 94.12% | 95.02% | 87.31% | 90.98% |
| **HDFS** | After | **97.01%** | **95.07%** | **95.00%** | **95.64%** |
| **BGL**  | Before | 91.84% | 93.15% | 84.27% | 88.49% |
| **BGL**  | After | **94.72%** | **92.88%** | **91.60%** | **92.23%** |

---

## 2. Our Novel Contributions (`src/our_contributions/`)
1. **Edge-Weighted Graph Attention (EW-GAT)**: Injects log-scaled transition frequencies $\log(1 + w_{ij})$ as edge priors into attention heads, avoiding noise from high-frequency repetitive transitions.
2. **Multi-Scale Hierarchical Readout**: Aggregates $[\text{mean}(H) \,\|\, \text{max}(H) \,\|\, \text{std}(H)]$ through a LayerNorm + GELU projection head.
3. **Class-Balanced Focal Loss**: Prevents overconfidence on duplicated oversampled samples.
4. **Execution Script**: `src/our_contributions/run_improvements.py`
