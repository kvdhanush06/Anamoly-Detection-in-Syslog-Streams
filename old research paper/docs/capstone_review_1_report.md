# Anomaly Detection in Syslog Streams: A Comparative Evaluation of Linear and Deep Learning Architectures

**Capstone Review 1 Document**

**Team Members:**
- Venkata Dhanush Kakarlamudi (23BCE8172)
- Veera Nidhish Gondimalla (23BCE7421)
- Settipalli Raja Rushi Praneeth Reddy (23BCE7425)
- Evvala Venkata Lakshmi Narasimha Veereswar (23BCE7439)

---

## Table of Contents
- [1. Abstract](#1-abstract)
- [2. Introduction & Problem Statement](#2-introduction--problem-statement)
- [3. Progress Till Review 1: Methodology & Experiments](#3-progress-till-review-1-methodology--experiments)
  - [Phase 1: Baseline Replication (Count-Based Linear Models)](#phase-1-baseline-replication-count-based-linear-models)
  - [Phase 2: Sequence Order Embeddings (Semantic NLP)](#phase-2-sequence-order-embeddings-semantic-nlp)
  - [Phase 3: Structural Node-Based Grouping](#phase-3-structural-node-based-grouping)
  - [Phase 4: Deep Learning Autoencoder on Semantics](#phase-4-deep-learning-autoencoder-on-semantics)
  - [Phase 5: Re-Evaluating the Baseline with Dynamic Peaks-Over-Threshold (POT)](#phase-5-re-evaluating-the-baseline-with-dynamic-peaks-over-threshold-pot)
- [4. Conclusion of Phase 1](#4-conclusion-of-phase-1)
- [5. Future Scope & Roadmap (For Review 2 & 3)](#5-future-scope--roadmap-for-review-2--3)

---

## 1. Abstract
The primary goal of this project is to develop, evaluate, and iteratively improve robust architectures for log anomaly detection across different complex system log datasets. The initial phases of this project were anchored in replicating and analyzing a 2025 IEEE ICSRS base paper ("Lightweight Optimization based Log-file Anomaly Detection" by Fält et al.), which proposed standard frequency-based linear models such as Principal Component Analysis (PCA) and Robust Principal Component Analysis (RPCA). We tested these models on structured logs from the Hadoop Distributed File System (HDFS_v1) and complex, real-world interleaved logs from the Blue Gene/L (BGL) supercomputer. Our rigorous reproduction established the brittleness of traditional count-based models on complex data, revealing critical data-geometry assumptions and proving that linear approaches collapse when applied to highly interleaved systems like BGL. Subsequently, the project explored Semantic Natural Language Processing (NLP) embeddings (Word2Vec) and Deep Learning architectures (Multi-Layer Perceptron Autoencoders) to capture contextual execution paths and non-linear data distributions. This pivot significantly improved anomaly detection performance and robustness. This report details the comprehensive progression of these methodologies, the mathematical insights gained, and the experiments conducted up to Review 1.

## 2. Introduction & Problem Statement
System logs serve as a fundamental data source for monitoring the health, security, and operational status of large-scale distributed systems. However, effectively detecting anomalies within these logs is an increasingly complex challenge due to the massive volume of data, the unstructured nature of raw log messages, and the intricate interleaving of concurrent processes across multiple nodes. 

Traditional anomaly detection approaches in this domain often rely on count-based matrices (such as term-frequency or TF-IDF) of discrete log events. While these methods perform admirably for highly structured, low-rank data (like HDFS logs which cleanly project onto a rank-2 plane), they fundamentally fail to capture the sequential order and semantic context of log execution paths. Our core problem statement focuses on overcoming the limitations of these count-based linear models when applied to complex, high-variance datasets like BGL. In such environments, simple event frequencies are mathematically insufficient to establish a distinct normal operational subspace, causing standard models to collapse. This project aims to bridge that gap by transitioning from simplistic frequency counting to deep semantic sequence modeling.

## 3. Progress Till Review 1: Methodology & Experiments
Our workflow up to Review 1 has been meticulously broken down into five major experimental phases. Each phase was designed to iteratively diagnose architectural limitations and develop targeted improvements.

### Phase 1: Baseline Replication (Count-Based Linear Models)
*   **Objective & Method:** We sought to replicate the baseline methods from the base paper, which construct a term-frequency matrix of discrete log events and apply linear projection techniques (PCA, RPCA) to detect anomalies via reconstruction error. We analyzed the HDFS_v1 dataset, containing ~11.1 million log lines grouped by `block_id`.
*   **HDFS Dataset Insights:** Through rigorous testing, we discovered that taking the raw event counts and applying an undocumented L2-normalization step mapped all normal sequences tightly onto a rank-2 plane on a unit sphere. With this correction, our PCA implementation achieved an F1-score of **94.65%**, perfectly matching the paper's claimed 94.64%. However, the RPCA implementation suffered a severe "rank-collapse" on this data geometry, peaking at 87.16% F1 (short of the paper's 90.55% target) and demonstrating that RPCA's formulation underperforms standard PCA when fitting data lacks contamination.
*   **BGL Dataset Evaluation:** To test generalizability, we parsed and sequenced 4.7 million real-world logs from the BGL dataset. When evaluated under strict protocols, the PCA baseline collapsed entirely to **30.18% F1**. Even with an oracle threshold sweep to find the absolute theoretical maximum, the linear model reached a strict ceiling of **~65% F1**. This proved definitively that the linear approach is fundamentally dataset-dependent and inadequate for highly interleaved, complex logs where the intrinsic rank is much higher (e.g., rank 239).

### Phase 2: Sequence Order Embeddings (Semantic NLP)
*   **Objective & Method:** To resolve the rank-collapse issue and capture the temporal transition meaning of logs (e.g., preserving the order `[Login, Error, Logout]`), we replaced sparse count matrices with dense, continuous semantic vectors using Word2Vec (`gensim`).
*   **Results & Diagnosis:** On the heavily structured HDFS dataset, replacing counts with continuous dense vectors successfully resolved the rank-collapse, allowing the RPCA model's performance to recover significantly to **84.96% F1**. On BGL, however, performance initially degraded (dropping to 44.28% F1). We diagnosed that the arbitrary 5-minute time window extraction method created a "semantic soup," where temporally interleaved logs from entirely independent server nodes were being incorrectly averaged together, destroying coherent execution paths.

### Phase 3: Structural Node-Based Grouping
*   **Objective & Method:** We refactored the BGL log extraction process to correct the semantic soup issue. Instead of grouping arbitrarily by time, we grouped logs structurally and logically by their originating `Node` identifier combined with a 6-hour sliding `Window_ID`.
*   **Results & Diagnosis:** This structural alignment mapped the sequences to true physical execution paths before Word2Vec vectorization. Consequently, linear PCA performance on these structurally grouped embeddings leaped from the previous strict 65% ceiling to **75.00% F1**. This breakthrough proved that semantic transitions contain critical anomaly indicators that raw frequency counts entirely omit, provided the sequences are logically extracted.

### Phase 4: Deep Learning Autoencoder on Semantics
*   **Objective & Method:** While semantic embeddings improved linear model performance, Word2Vec embeddings exist in a continuous, non-linear space where linear projections like PCA are sub-optimal. We replaced the linear detectors with a multi-layer perceptron (MLP) Deep Learning Autoencoder designed to reconstruct the 32-dimensional semantic vectors.
*   **Results & Diagnosis:** Under a rigorous 5-fold cross-validation on over 1.15 million sequences, the Autoencoder completely shattered the limits on the complex BGL dataset, achieving an average **87.46% F1-score**. The model successfully navigated the high variance and non-linear data distributions, proving that an Autoencoder paired with Node-Based Semantic Embeddings is a vastly superior architecture for messy, real-world systems.

### Phase 5: Re-Evaluating the Baseline with Dynamic Peaks-Over-Threshold (POT)
*   **Objective & Method:** The base paper relied on a hardcoded 95th-percentile error threshold, an unrealistic heuristic for real-world deployment where the anomaly ratio is unknown. We re-evaluated this and discovered a "tie-inflation" bug in standard rank-based error metrics that skewed results. To fix this and automate thresholding, we implemented dynamic Extreme Value Theory via the Peaks-Over-Threshold (POT) algorithm.
*   **Results & Diagnosis:** Our dynamic POT threshold actively modeled the tail of the reconstruction error distribution rather than forcing a strict percentage cutoff. This raised the linear PCA baseline performance on HDFS to an impressive **95.95% F1**, effectively removing the reliance on oracle knowledge and making the algorithm deployable in real-world streaming scenarios.

## 4. Conclusion of Phase 1
The extensive experiments conducted up to Review 1 have definitively established a "Dataset-Dependency Rule" for log anomaly detection:
1. **For Perfectly Structured Logs (e.g., HDFS):** Linear PCA on lightweight L2-normalized counts represents the absolute mathematical ceiling. Introducing deep learning or complex semantic logic here is unnecessary and can even introduce false positives.
2. **For Complex Interleaved Logs (e.g., BGL):** Standard frequency-count linear models are brittle and mathematically collapse. Shifting to **Node-Based Semantic Embeddings paired with Deep Learning (Autoencoders)** allows the system to understand the actual contextual flow and non-linear execution paths, leading to highly robust anomaly detection that shatters the linear ceiling. 

While this architecture demonstrates immense potential, the deep learning model exhibits some sensitivity to threshold calibration (POT variance across folds) and potential data distribution shifts, which will be the focus of subsequent refinements.

## 5. Future Scope & Roadmap (For Review 2 & 3)
As we progress toward the next review cycles, the architecture will remain flexible for significant strategic pivots. Our planned explorations include:
1.  **Advanced Sequence Modeling:** Investigating whether deeper sequence models (such as Transformers, Long Short-Term Memory networks (LSTMs), or Small Language Models (SLMs)) can capture long-range dependencies better than simple Word2Vec aggregations.
2.  **Addressing Data Leakage & Feature Engineering:** Ensuring strict isolation between training embeddings and evaluation data (addressing identified embedding-level data leakage) to guarantee robust out-of-sample performance and true generalization.
3.  **Alternative Non-Linear Architectures:** Remaining open to entirely new paradigms, such as contrastive learning, Isolation Forests on embeddings, or self-supervised anomaly detection frameworks, should the current Autoencoder hit a performance plateau or demonstrate inference bottlenecks.
4.  **Real-Time Feasibility:** Evaluating and optimizing the inference time of the chosen deep learning architectures to ensure viability for deployment in live, high-throughput log streams.

