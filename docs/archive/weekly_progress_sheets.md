# Capstone Project Diary - Weekly Progress Sheets

**Project Title:** Anomaly Detection in Syslog Streams: Implementing Lightweight Deep Semantic Architecture
**Guide:** Dr. Mohinder Singh B (SCOPE)
**Team Members (Reg. No. - Name):**
- 23BCE8172 - Venkata Dhanush Kakarlamudi
- 23BCE7421 - Veera Nidhish Gondimalla
- 23BCE7425 - Settipalli Raja Rushi Praneeth Reddy
- 23BCE7439 - Evvala Venkata Lakshmi Narasimha Veereswar

---

## Week 1
**Period From:** 10/08/2026 **To:** 16/08/2026
**Date of Meeting:** 10/08/2026
**Members attended (Reg. No.):** 23BCE8172, 23BCE7421, 23BCE7425, 23BCE7439

**Tasks Completed:**
- Conducted literature survey on system log anomaly detection models.
- Analyzed the base paper "Lightweight Optimization based Log-file Anomaly Detection" (Fält et al., 2025).
- Formulated the problem statement focusing on the limitations of count-based linear architectures.
- Set up the initial project repository and downloaded the HDFS_v1 dataset.

**Tasks Planned Next Week:**
- Parse and preprocess the HDFS_v1 dataset.
- Implement the baseline PCA and RPCA models with TF-IDF features.
- Evaluate the baseline models using the 95th-percentile error threshold.

**Comments/Suggestion by the Guide:** 
<br><br><br>
**Signature of the Guide:** 

---

## Week 2
**Period From:** 17/08/2026 **To:** 23/08/2026
**Date of Meeting:** 17/08/2026
**Members attended (Reg. No.):** 23BCE8172, 23BCE7421, 23BCE7425, 23BCE7439

**Tasks Completed:**
- Preprocessed the HDFS_v1 dataset into discrete event count matrices.
- Implemented standard PCA and RPCA baseline algorithms.
- Encountered rank-collapse issues with RPCA and sub-optimal performance with PCA (~76% F1).
- Began investigating the mathematical data geometry to resolve the discrepancy with the base paper.

**Tasks Planned Next Week:**
- Debug the baseline implementation by applying L2-normalization.
- Parse the complex, interleaved BGL supercomputer dataset for generalization testing.
- Evaluate the baseline linear models on the BGL dataset.

**Comments/Suggestion by the Guide:** 
<br><br><br>
**Signature of the Guide:** 

---

## Week 3
**Period From:** 24/08/2026 **To:** 30/08/2026
**Date of Meeting:** 24/08/2026
**Members attended (Reg. No.):** 23BCE8172, 23BCE7421, 23BCE7425, 23BCE7439

**Tasks Completed:**
- Successfully matched the base paper’s PCA result (~94.65% F1) on HDFS by introducing L2-normalized raw counts.
- Parsed 4.7 million real-world logs from the BGL dataset using automated sequence extraction.
- Evaluated the linear baseline models on the BGL dataset.
- Discovered that traditional linear models collapse on complex, high-rank interleaved logs (F1 ~65%).

**Tasks Planned Next Week:**
- Document the limitations of count-based frequency matrices.
- Finalize the Review 1 presentation and report.
- Deliver the Capstone Review 1 presentation outlining the proposed architectural pivot.

**Comments/Suggestion by the Guide:** 
<br><br><br>
**Signature of the Guide:** 

---

## Week 4
**Period From:** 31/08/2026 **To:** 06/09/2026
**Date of Meeting:** 31/08/2026
**Members attended (Reg. No.):** 23BCE8172, 23BCE7421, 23BCE7425, 23BCE7439

**Tasks Completed:**
- Consolidated the Phase 1 findings into the `00_master_project_log.md` tracking document.
- Delivered the Review 1 presentation highlighting the dataset-dependency rule of linear architectures.
- Proposed the pivot from discrete event counting to Semantic NLP models (Word2Vec) for Review 2.
- Designed a structural log grouping strategy to resolve the "semantic soup" issue in the BGL dataset.

**Tasks Planned Next Week:**
- Implement Node-Based structural grouping with a 6-hour sliding window on BGL logs.
- Refactor the data pipeline to extract sequential semantic execution paths.
- Investigate and resolve potential data leakage in Word2Vec training.

**Comments/Suggestion by the Guide:** 
<br><br><br>
**Signature of the Guide:** 

---

## Week 5
**Period From:** 07/09/2026 **To:** 13/09/2026
**Date of Meeting:** 07/09/2026
**Members attended (Reg. No.):** 23BCE8172, 23BCE7421, 23BCE7425, 23BCE7439

**Tasks Completed:**
- Implemented Node-Based structural grouping (Node + Window_ID) on the BGL dataset.
- Converted parsed BGL log sequences into `bgl_sequences.pkl` to preserve execution paths.
- Identified an implicit data-leak vulnerability in standard global embedding approaches.
- Refactored the Word2Vec pipeline to strictly train only on the normal training fold and handle test-set logs as Out-Of-Vocabulary (OOV).

**Tasks Planned Next Week:**
- Implement the deep learning Multi-Layer Perceptron (MLP) Autoencoder architecture.
- Train the Autoencoder on the 32-D L2-normalized Word2Vec mean-pooled vectors.
- Implement Dynamic Peaks-Over-Threshold (POT) to automate anomaly thresholding.

**Comments/Suggestion by the Guide:** 
<br><br><br>
**Signature of the Guide:** 

---

## Week 6
**Period From:** 14/09/2026 **To:** 20/09/2026
**Date of Meeting:** 14/09/2026
**Members attended (Reg. No.):** 23BCE8172, 23BCE7421, 23BCE7425, 23BCE7439

**Tasks Completed:**
- Developed a 3-layer (16-8-16) MLP Autoencoder to capture non-linear semantic manifolds.
- Integrated the leak-free Word2Vec mean-pooling function directly into the K-Fold evaluation loop (`run_improvements_leak_fixed.py`).
- Implemented the Extreme Value Theory-based Peaks-Over-Threshold (POT) algorithm.
- Replaced the hardcoded 95th-percentile heuristic with dynamic POT across all models.

**Tasks Planned Next Week:**
- Evaluate the final models and compile the F1, Precision, and Recall metrics.
- Perform a comparative analysis against standard lightweight and heavyweight models.
- Prepare the presentation and documentation for Capstone Review 2.

**Comments/Suggestion by the Guide:** 
<br><br><br>
**Signature of the Guide:** 

---

## Week 7
**Period From:** 21/09/2026 **To:** 27/09/2026
**Date of Meeting:** 21/09/2026
**Members attended (Reg. No.):** 23BCE8172, 23BCE7421, 23BCE7425, 23BCE7439

**Tasks Completed:**
- Finalized HDFS evaluation, achieving a 95.95% F1 score using PCA + POT.
- Completed 5-fold cross-validation on the BGL dataset with the leak-free Autoencoder, yielding Precision: 97.50%, Recall: 81.71%, and F1: 88.88%.
- Conducted a comparative analysis, proving our lightweight semantic approach effectively bridges the gap to heavyweight Graph Neural Networks.
- Drafted and formatted the Capstone Review 2 presentation slides.

**Tasks Planned Next Week:**
- Research and design an LSTM sequence model to capture long-term temporal dependencies.
- Address the zero-padding dilution issues commonly found in variable-length log sequences.
- Compare the upcoming LSTM implementation against heavy Transformer models for Review 3.

**Comments/Suggestion by the Guide:** 
<br><br><br>
**Signature of the Guide:** 
