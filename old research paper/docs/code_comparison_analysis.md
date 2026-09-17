# Code Comparison: Baseline Replications vs. Our Contributions

This document outlines the specific code differences between the baseline implementation (`src/baseline_replication/run_baseline.py`) and our enhanced contributions (`src/our_contributions/run_improvements.py`). These differences are the direct mathematical and architectural reasons why our contributions achieve a significantly higher F1 score compared to the baseline models.

---

## 1. HDFS Dataset: The Thresholding Improvement

### Baseline (`run_hdfs_paper_exact_protocol`)
In the baseline replication, the code attempts to flag anomalies by strictly assuming a known **5% anomaly ratio**. It uses a hardcoded rank-based selection on the testing errors:

```python
et = fold_model.predict_errors(Xt)

# Hardcoded rank-based top 5% selection
n_flag = int(np.ceil(0.05 * len(et)))
flags = np.argsort(et)[-n_flag:]
y_pred = np.zeros(len(et), dtype=int)
y_pred[flags] = 1
```
**Why it fails to generalize:** This approach relies on "oracle knowledge" (knowing exactly how many anomalies exist in the test set). While it avoids the "tie-inflation" bug in PCA, it is rigid and cannot adapt to real-world streaming data where the anomaly ratio fluctuates.

### Our Contribution (`run_hdfs_pot_improvement`)
We replaced the hardcoded ranking with a dynamic **Extreme Value Theory (POT)** algorithm. Instead of just looking at the test errors, we evaluate the errors on the clean fitting data (`ef`) to model the tail distribution, and calculate a dynamic threshold (`theta`):

```python
ef = fold_model.predict_errors(Xf)
et = fold_model.predict_errors(Xt)

# Dynamic POT Thresholding
theta = compute_threshold_pot(ef, q=0.90, target_fpr=1e-3)
y_pred = (et > theta).astype(int)
```
**Why it improves F1:** The POT algorithm actively models the boundary of normal behavior based on the specific fold's training data. It flags anomalies dynamically if their error exceeds the calculated threshold, removing the reliance on a fixed percentage and naturally raising the F1 score by catching edge-case anomalies that the strict 5% cutoff misses.

---

## 2. BGL Dataset: The Deep Learning & Semantics Breakthrough

### Baseline (`run_bgl_baseline`)
The baseline approach attempts to use simple linear PCA on basic event-count matrices (L2-normalized) for the highly complex BGL dataset:

```python
# Uses basic count-based features
BGL_FEATURES_COUNT = "data/BGL/preprocessed/features.npz"
X_raw = data["X_counts"].astype(np.float32)
X = l2_normalize(X_raw)

# Uses linear PCA Model
model = PCAModel(k=6)
```
**Why it fails to generalize:** BGL contains heavily interleaved logs from multiple concurrent nodes. Simple frequency counts mapped onto a linear PCA model collapse because they cannot capture the sequence order or context, leading to an F1 score ceiling of ~65% (often crashing to ~30% in blind tests).

### Our Contribution (`run_bgl_autoencoder`)
To solve the linear collapse on complex data, our code fundamentally shifts the architecture in two major ways:

**A. Node-Based Semantic Word2Vec Features:**
Instead of raw counts, we load dense semantic embeddings (Word2Vec) that were structurally grouped by Node ID, preserving the actual sequence execution paths.
```python
# Uses dense Semantic NLP embeddings
BGL_FEATURES_W2V = "data/BGL/preprocessed/features_w2v.npz"
data = np.load(BGL_FEATURES_W2V)
X = data["X_l2"]
```

**B. Multi-Layer Perceptron (MLP) Autoencoder:**
Because the Word2Vec features exist in a complex non-linear space, linear PCA is insufficient. We replace it with a Deep Learning Autoencoder capable of learning non-linear reconstructions.
```python
# Imports Deep Learning Autoencoder
from models.deep_learning import AutoencoderModel

# Uses an MLP Autoencoder instead of linear PCA
model = AutoencoderModel(hidden_layer_sizes=(16, 8, 16))
```

**Why it improves F1:** 
1. The **Word2Vec features** capture the contextual meaning of the logs (e.g., a specific sequence of errors).
2. The **Autoencoder** maps the non-linear distributions of these embeddings, identifying anomalies that would overlap with normal data in a linear PCA subspace. 
3. Paired with the **POT Threshold**, this architecture shatters the baseline's 65% ceiling, raising the F1 score significantly on complex datasets like BGL.
