# BGL Node-Grouping Label Logic and Leakage Analysis

## The Labeling Rule
In the Blue Gene/L (BGL) dataset, logs are extracted and grouped by `Node` identity and a 6-hour `Window_ID` (as implemented in `src/features/extract_embeddings.py`).

**Sequence Label Assignment:**
The label for an entire 6-hour sequence for a given Node is determined by a **Maximum / Any-Anomalous rule**.
If *any single log line* within that Node's sequence during that 6-hour window has a label other than `-` (the normal indicator in BGL), the sequence label is `1` (Anomaly). If all logs in the sequence are `-`, the sequence label is `0` (Normal).
This is achieved via: `y = grouped['Label_Bin'].max().values`

## Leakage Risk Analysis
**Is there label leakage?**
No. There is zero label leakage in this architecture for the following reasons:

1. **Features Exclude Node Identity**: The `Node` string itself (e.g., `R02-M1-N0-C:J12-U11`) is used *strictly* for pandas `.groupby()` structural grouping. It is **never** passed into the `Word2Vec` model, nor is it vectorized into the Autoencoder. The only features fed to the models are the raw transitions of the `EventId` tokens (e.g., `[E12, E34, E12]`).
2. **Features Exclude Timestamps**: The `Window_ID` is strictly structural. It ensures the Word2Vec model isn't trying to learn sequences that span across days/months, keeping execution transitions localized.
3. **Labels are strictly post-processing targets**: The target `y` vector is completely disjoint from the sequence embedding generation. The model (PCA or Autoencoder) only sees the `EventId` sequence representations during training (on normal data) and evaluation.

Therefore, the model cannot trivially correlate a Node identity or a timeframe to an anomaly because the model never sees Node identities or timeframes—it only evaluates the semantic validity of the execution path.
