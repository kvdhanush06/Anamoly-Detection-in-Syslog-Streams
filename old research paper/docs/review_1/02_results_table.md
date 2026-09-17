# Results Summary

The following table presents the validated results of our models across different datasets.

| Setup / Model | Precision | Recall | F1 Score | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **HDFS PCA (Original Protocol)** | 92.34% | 96.96% | 94.60% | Replicates the paper's claimed 94.64% F1 |
| **BGL PCA (Baseline Application)** | - | - | ~65.00% (Ceiling) | Linear model baseline; paper did not test this dataset |
| **BGL Autoencoder (Word2Vec)** | 97.27% ± 1.08% | 80.76% ± 13.84% | 87.46% ± 9.17% | Uses node-based log grouping |

*Note: All numbers are derived from 5-fold cross-validation unless stated otherwise.*
*Note: embedding-level data leakage identified in Word2Vec training (see Limitation 6) - true out-of-sample performance is likely somewhat lower than the reported figure.*

## Autoencoder Footprint Benchmark
*   **Parameters:** ~1384
*   **Memory Footprint:** 26.58 KB
*   **Inference Time:** ~0.0002ms per sample (Note: Measures model forward-pass only, excluding preprocessing and Word2Vec lookup overhead).
