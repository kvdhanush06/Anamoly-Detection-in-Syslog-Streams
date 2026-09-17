# Debugging Journey

### 1. The Mean-Centering Bug
*   **What broke:** Our initial PCA reproduction attempt on the HDFS dataset yielded a recall stuck at approximately 42%, severely underperforming the expected baseline.
*   **How we found it:** We systematically audited our implementation against the mathematics described in Section III-B of the original paper.
*   **How we fixed it:** We identified that we were not properly mean-centering the data matrix before calculating the principal components. Implementing explicit mean-centering aligned our logic with the paper and resolved the recall deficit.

### 2. The Tie-Inflation Threshold Bug
*   **What broke:** When attempting to apply the paper's 95th-percentile anomaly threshold, our model flagged approximately 9% of the dataset instead of the targeted 5%.
*   **How we found it:** Debugging the reconstruction errors revealed that there were only ~198 unique error values across 21,000 test samples. Using a simple value-based threshold (`error > 95th_percentile_value`) caused massive ties, forcing the algorithm to over-flag.
*   **How we fixed it:** We switched the thresholding logic to a rank-based top-5% selection. By strictly taking the top 5% of samples by rank, we eliminated tie-inflation, which perfectly corrected the False Positive Rate and allowed us to match the paper's 94.64% F1 claim.

### 3. The BGL "Semantic Soup" Issue
*   **What broke:** Applying Word2Vec to the BGL dataset initially resulted in a 44% F1 score, demonstrating a failure to capture log semantics.
*   **How we found it:** Analysis of the feature extraction pipeline showed that grouping logs by arbitrary fixed time windows mixed completely unrelated operations from different server nodes into the same sequence.
*   **How we fixed it:** We refactored the pipeline to group logs by their specific Node identity before applying any windowing. This maintained the causal sequence of events for each physical machine, resolving the semantic overlap and allowing the model to detect anomalies accurately.
