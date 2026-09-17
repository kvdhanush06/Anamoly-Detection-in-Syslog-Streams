# Limitations and Future Work

### 1. No Automatic Model Selection
*   **What it is:** The pipeline requires a user to manually select between the PCA model (for HDFS) and the Autoencoder model (for BGL) based on prior knowledge of the dataset's characteristics. There is no automated routing mechanism for genuinely new, unlabeled datasets.
*   **Why it matters:** This is the most critical limitation, as it dictates whether the project functions as an adaptive "framework" or just a collection of static models.
*   **Future Fix:** Implement an automatic complexity check that evaluates dataset variance (e.g., determining how much variance a few PCA components explain on the raw feature matrix without utilizing labels). This check would automatically route data to the appropriate model, and must be validated on a third, unseen dataset.

### 2. Arbitrary Time Window Grouping (BGL)
*   **What it is:** The BGL feature extraction pipeline divides logs into fixed 6-hour windows.
*   **Why it matters:** A fixed window can split a single continuous anomaly across two distinct time periods (e.g., spanning from hour 5:59 to 6:01), potentially causing the model to miss the contextual sequence of the failure.
*   **Future Fix:** Implement a sliding window approach that advances continuously rather than relying on static chronological boundaries.

### 3. Full Pipeline Latency Not Measured
*   **What it is:** While the raw Autoencoder model was benchmarked and found to be extremely fast (0.0002ms per sample), the timing metrics for the complete pipeline were not evaluated.
*   **Why it matters:** The overhead introduced by text preprocessing and Word2Vec embedding lookups could create bottlenecks in real-time streaming environments, masking the true operational latency.
*   **Future Fix:** Benchmark the complete end-to-end processing pipeline, from raw string ingestion to final anomaly prediction, to determine true operational latency.

### 4. Autoencoder Threshold Variance
*   **What it is:** During 5-fold cross-validation, the BGL Autoencoder's recall fluctuated significantly (80.76% ± 13.84%). In Fold 1, recall dropped to 53.20%.
*   **Why it matters:** The variance is caused by the POT (Extreme Value Theory) threshold calibrating differently based on the distribution of normal data in the fitting set. If the training data distribution shifts slightly, the threshold becomes overly conservative, leading to missed anomalies.
*   **Future Fix:** Investigate and implement more stable threshold calibration methods that are less sensitive to minor distribution shifts in the normal data.

### 5. Word2Vec Out-of-Vocabulary (OOV) Blindspot
*   **What it is:** Word2Vec models only generate meaningful embeddings for tokens encountered during the training phase.
*   **Why it matters:** If the system generates a genuinely new type of error containing an unseen word, the model lacks an embedding for it, effectively rendering the anomaly invisible to the Autoencoder.
*   **Future Fix:** Transition to subword or character-level embeddings (e.g., FastText), or implement a dedicated fallback detection path that triggers specifically when unknown tokens are encountered.

### 6. Word2Vec Embedding-Level Data Leakage
*   **What it is:** Word2Vec is trained once on the full BGL dataset (all sequences across all folds) before the cross-validation split, rather than being re-fit separately inside each fold using only that fold's fitting data.
*   **Why it matters:** Even though Word2Vec never sees anomaly labels, it does learn vocabulary and sequence structure from what later becomes test data. This means the reported BGL Autoencoder F1 (87.46%) likely somewhat overestimates true out-of-sample performance on genuinely unseen data. A properly isolated pipeline would also correctly produce out-of-vocabulary behavior for genuinely new error words in the real world, which the current pipeline cannot demonstrate.
*   **Future Fix:** Refit Word2Vec separately inside each cross-validation fold, using only that fold's fitting sequences, and generate test-fold embeddings using that fold-specific model.
