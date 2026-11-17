# Anomaly Detection in Syslog Streams
**Review 2 Presentation Draft**

---

## Slide 1: Title Slide
**Title:** Anomaly Detection in Syslog Streams: Implementing Lightweight Deep Semantic Architecture
**Subtitle:** Capstone Review 2
**Team Members:** Venkata Dhanush Kakarlamudi, Veera Nidhish Gondimalla, Settipalli Raja Rushi Praneeth Reddy, Evvala Venkata Lakshmi Narasimha Veereswar

---

## Slide 2: Recap of Review 1 (The Problem)
* **The Goal:** Detect anomalies in complex system logs.
* **The Problem Identified:** Traditional lightweight linear models (PCA/RPCA) work well on simple datasets (HDFS) but **collapse completely** (F1 ~65%) on complex, interleaved real-world logs like the BGL supercomputer.
* **The Reason:** Raw frequency counts fail to capture the sequential, semantic execution path of concurrent processes.

---

## Slide 3: Proposed Architecture (Where we left off)
* **The Pivot:** Shifting from Count-Based Linear Models to **Semantic NLP Models**.
* **Core Idea:** Use Word2Vec to convert discrete log events into continuous semantic vectors.
* **Deep Learning Engine:** Replace simple PCA with a Deep Learning Autoencoder capable of handling non-linear data distributions.
* **Adaptive Thresholding:** Replace static, hardcoded thresholds with Dynamic Extreme Value Theory (POT).

---

## Slide 4: Implementation - Node-Based Semantic Grouping
* **The Challenge:** Extracting BGL logs by pure time-windows (e.g., 5 minutes) creates a "semantic soup" of interleaved, unrelated server logs.
* **Our Solution:** Grouping logs structurally by **Node Identifier** and a 6-hour sliding window.
* **The Impact:** This perfectly aligned the sequences to their true physical execution paths, giving our NLP model accurate context and immediately boosting baseline performance.

---

## Slide 5: Addressing the Data Leakage Vulnerability (Our Key Contribution)
* **The Flaw in Existing Literature:** Many log anomaly papers implicitly leak data by training their embedding models (Word2Vec/TF-IDF) on the entire dataset *before* the Train/Test split.
* **Our Strict Isolation Fix:** We refactored our pipeline to strictly isolate Word2Vec training. The model is dynamically trained **only on the normal training fold** and then applies mean pooling to test data, handling unseen logs as Out-Of-Vocabulary (OOV).
* **Why it Matters:** This guarantees that our evaluation is 100% scientifically honest and proves our model generalizes to entirely unseen data in real-world deployments.

---

## Slide 6: Deep Learning Autoencoder on Semantics
* **The Model:** A 3-layer Multi-Layer Perceptron (MLP) Autoencoder (16-8-16 architecture).
* **The Workflow:** 
  1. Input: 32-dimensional L2-normalized Word2Vec vectors.
  2. Bottleneck: Compresses data to learn the "normal" structural manifold.
  3. Reconstruction: High reconstruction error flags the sequence as an anomaly.
* **Advantage:** Navigates the high variance and non-linear distribution of semantic embeddings far better than linear SVD/PCA.

---

## Slide 7: Dynamic Peaks-Over-Threshold (POT)
* **The Limitation of Base Papers:** Relying on a hardcoded 95th-percentile error cutoff requires "oracle" knowledge of how many anomalies exist.
* **Our Implementation:** We implemented Extreme Value Theory via the **POT Algorithm**.
* **How it works:** It dynamically models the tail of the reconstruction error distribution to adaptively set thresholds without prior knowledge.
* **Result:** Makes the algorithm fully deployable in real-time streaming environments.

---

## Slide 8: Experimental Results & Performance
**HDFS Dataset (Structured):**
* **PCA + POT:** Achieved **95.95% F1 Score**.
* *Conclusion:* For highly structured logs, linear modeling remains the most efficient ceiling.

**BGL Dataset (Complex & Interleaved):**
* **Our Leak-Free Node-Based Autoencoder:**
  * **Precision:** 97.50%
  * **Recall:** 81.71%
  * **F1 Score:** **88.88%**
* *Conclusion:* Achieved a massive +23% F1 improvement over the original linear baselines on complex data.

---

## Slide 9: Comparative Analysis: Lightweight vs. Heavyweight
* **Heavyweight Models (e.g., BertGCN, LogBERT):** 
  * Achieve ~95-98% F1 on BGL.
  * *Drawbacks:* Require massive GPU clusters, millions of parameters, and slow inference.
* **Traditional Lightweight ML (K-Means, Isolation Forest):**
  * Typically achieve **75% - 85% F1** on BGL.
* **Our Model (Word2Vec + Autoencoder):**
  * Achieves **88.88% F1**.
  * *The Verdict:* Our architecture successfully bridges the gap, offering near-SOTA performance while remaining fully lightweight, unsupervised, and capable of running on standard CPUs in real-time.

---

## Slide 10: Conclusion & Future Scope for Review 3
* **Conclusion:** Structural data preparation (Node-Based grouping) combined with simple Deep Learning (Autoencoders) provides a highly robust, efficient, and leak-free anomaly detection pipeline.
* **Next Steps:** 
  * Implement an **LSTM (Long Short-Term Memory)** sequence model to capture long-range temporal dependencies.
  * Compare the LSTM's ability to close the final performance gap against heavyweight Transformer models.
