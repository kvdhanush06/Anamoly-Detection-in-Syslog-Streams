"""
# Project Spec: BertGCN Log-Based Anomaly Detection Using Graph Neural Networks

## Base Paper
- **Title:** Log-Based Anomaly Detection of System Logs Using Graph Neural Network
- **Authors:** Eman Alsalmi, Abeer Alhuzali, Areej Alhothali
- **Journal:** Computers, Materials and Continua (Dec 2025 / 2026)
- **DOI:** 10.32604/cmc.2025.071012

## Core Methodology
1. **Parsing-Free Preprocessing:** Non-character removal, lowercase conversion, and WordPiece tokenization.
2. **Contextual Node Embeddings:** BERT 768-dimensional contextual representations for each unique log event.
3. **Directed Weighted Graph Construction:** $G = (V, E, X_v, F_w)$ with transition count edge weights and starting event self-loops.
4. **Graph Convolutional Network (GCN):** Message passing with normalized adjacency matrices and graph readout pooling.
5. **Class Imbalance:** Minority oversampling (random duplication) during training.
6. **Our Contributions:**
   - Edge-Weighted Graph Attention (EW-GAT)
   - Multi-Scale Readout Pooling (Mean + Max + Std)
   - Class-Balanced Focal Loss & Dynamic Calibration
"""
