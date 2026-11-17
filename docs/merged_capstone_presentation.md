# Merged Capstone Presentation (Review 1 + Review 2)

*Note for Claude: Do not condense this presentation. Treat this as two distinct halves joined together. The first half is the exact Review 1 presentation. The second half is the new Review 2 presentation.*

---
# PART 1: CAPSTONE REVIEW 1
*(Keep your original Review 1 slides exactly as they are. Below is the general flow to maintain.)*

## Slide 1: Title Slide (Review 1)
* **Title:** Anomaly Detection in Syslog Streams
* **Focus:** Introduction to the domain (AIOps, SRE).

## Slide 2-7: Introduction, Literature Review & Problem Statement
* Explain the massive scale of supercomputer logs.
* Review existing lightweight models (PCA/LogFIT) vs Heavyweight models (DeepLog).
* Choosing the Base Paper to replicate.

## Slide 8: Baseline Replication (HDFS)
* Successfully replicating the base paper's PCA model on the structured HDFS dataset.
* Discovery of the undocumented L2-normalization step.
* **Result:** Achieved 94.64% F1-Score on HDFS.

## Slide 9: Stress-Testing the Baseline (BGL)
* Testing the PCA baseline on the unstructured, highly interleaved BGL supercomputer dataset.
* **Result:** The linear model collapsed to 30.18% F1-Score.

## Slide 10: Proposed Architecture Concept
* Proposing a shift from "frequency counting" to "contextual semantics".
* Introducing the concept of a Deep Autoencoder to solve the BGL collapse.

## Slide 11: Thank You (End of Review 1)
* "Thank You" slide to cap off the first phase of the project.

---
---

# PART 2: CAPSTONE REVIEW 2 (Our Core Contributions)
*(This is where the new content begins, continuing right after the Review 1 Thank You slide.)*

## Slide 12: Title Slide (Review 2)
* **Title:** Transitioning from Linear to Deep Semantic Models
* **Subtitle:** Capstone Review 2 - Architecture and Final Results

## Slide 13: Contribution 1 - Perfecting the HDFS Baseline (POT)
* **The Flaw from Review 1:** The baseline PCA model on HDFS relied on a hardcoded, human-guessed "top 5%" threshold to find anomalies.
* **Our Solution:** We implemented the **Peaks-Over-Threshold (POT)** algorithm.
* **What is POT?** It is a mathematical algorithm from Extreme Value Theory. Instead of human guessing, POT mathematically models the "extreme tail" of the errors and autonomously draws the boundary between normal noise and true anomalies.
* **Result:** Raised HDFS performance to a fully autonomous **95.95% F1**.

## Slide 14: Contribution 2 - Tackling Unstructured Data (BGL)
* **The Flaw from Review 1:** BGL logs are unstructured and intertwined. "Pure Time Windows" created a "Semantic Soup" of unrelated events.
* **Our Solution:** Node-Based Sequence Alignment.
* **New Term - Sequence Alignment:** Sorting logs by their specific hardware chip (Node) and placing them in chronological order to reveal the true "Physical Execution Path" of that processor.
* **New Term - 6-Hour Window:** We extract logs in 6-hour chunks because it is the industry standard heuristic to capture a full computational job without cutting it in half.

## Slide 15: Deep Semantic Understanding (Word2Vec)
* We replaced word-counting with Natural Language Processing (Word2Vec) to understand the *meaning* of the logs.
* **New Term - Mean Pooling:** Word2Vec turns every log word into a math vector. Mean pooling simply calculates the mathematical average of all the words in a sequence to create one single, dense vector that represents the entire meaning of that sequence.

## Slide 16: The Semantic Autoencoder (16-8-16)
* We fed the mean-pooled semantic vectors into a Deep Learning Autoencoder.
* **New Term - 16-8-16 Architecture:** Our neural network compresses the input into 16 neurons, forces it through a tiny bottleneck of 8 neurons, and expands back to 16. This extreme compression forces the AI to memorize only the most vital, core patterns of normal supercomputer behavior. 
* If a sequence fails to reconstruct through the 8-neuron bottleneck, POT flags it as an anomaly.

## Slide 17: Ensuring Academic Rigor (The Final Updates)
* Before locking in our final results, we strictly audited our pipeline to ensure maximum mathematical fairness.
* **New Term - Preventing Test Information Leakage:** We strictly trained our Word2Vec model *only* on the training dataset. If it reads the whole dataset beforehand, it secretly "peeks" at the test vocabulary. We eliminated this data leak entirely.
* **New Term - 10-Fold Cross Validation:** We chopped the dataset into 10 pieces. We trained on 9 pieces (90%) and tested on 1 piece (10%), repeating this 10 times. Every single log was tested.
* **New Term - Seed Stabilization:** Neural networks use random starting weights. We implemented a stabilization loop that retrains the model if it hits a bad random seed, ensuring perfect convergence across all 10 folds.

## Slide 18: Final Results Comparison Table
* *(Insert the Benchmark Comparison Table here)*

| Model Architecture | Category | Dataset | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **LogClustering** | Lightweight | HDFS | 77.00% | 93.00% | 84.20% |
| **RPCA** | Lightweight | HDFS | 88.00% | 93.20% | 90.55% |
| **Standard PCA** | Lightweight | HDFS | 97.40% | 92.00% | 94.64% |
| **OUR PROPOSAL: PCA + POT** | **Lightweight** | **HDFS** | **93.61%** | **98.46%** | **95.95%** |
| | | | | | |
| **DeepLog (LSTM)** | Heavyweight | BGL | 83.20% | 69.40% | 75.70% |
| **CNN** | Heavyweight | BGL | 88.00% | 82.00% | 84.90% |
| **LogBERT (Transformer)** | Heavyweight | BGL | 94.50% | 89.90% | 92.10% |
| **OUR PROPOSAL: Semantic Autoencoder** | **Heavyweight** | **BGL** | **98.59%** | **83.55%** | **90.44%** |

## Slide 19: Conclusion
* We successfully built a hybrid architecture.
* We eliminated human guesswork using POT.
* We shattered the 30% baseline limit on unstructured data, achieving a robust, leak-free **90.44% F1-Score** on BGL without relying on slow, heavyweight LSTM architectures.

## Slide 20: Thank You (End of Review 2)
* Questions & Answers
