"""
Feature extraction for HDFS_v1 log anomaly detection.

Builds the TF-IDF feature vectors defined in Equation (2) of the paper,
starting from the raw event count vectors (E1-E29) in Event_occurrence_matrix.csv.

Run from project root with venv activated:
    python src/feature_extraction.py
"""

import numpy as np
import pandas as pd

LABEL_PATH = "data/HDFS_v1/preprocessed/anomaly_label.csv"
MATRIX_PATH = "data/HDFS_v1/preprocessed/Event_occurrence_matrix.csv"

OUTPUT_PATH = "data/HDFS_v1/preprocessed/features.npz"


def load_and_merge():
    labels_df = pd.read_csv(LABEL_PATH)
    matrix_df = pd.read_csv(MATRIX_PATH)

    # merge on BlockId so every row has both the ground-truth label and the
    # event count columns, with an explicit suffix so we can tell the two
    # "Label" columns apart
    merged = matrix_df.merge(
        labels_df, on="BlockId", how="inner", suffixes=("_matrix", "_truth")
    )
    print(f"Merged shape: {merged.shape}")
    return merged


def check_label_agreement(merged):
    """
    anomaly_label.csv gives Label_truth = Normal/Anomaly (our ground truth).
    Event_occurrence_matrix.csv gives Label_matrix = Success/Fail.
    Confirm these actually agree before trusting either.
    """
    print(f"\n{'=' * 50}")
    print("LABEL AGREEMENT CHECK")
    print(f"{'=' * 50}")

    # expected mapping: Success -> Normal, Fail -> Anomaly
    expected = merged["Label_matrix"].map({"Success": "Normal", "Fail": "Anomaly"})
    mismatch = merged["Label_truth"] != expected
    n_mismatch = mismatch.sum()

    print(f"Rows checked: {len(merged):,}")
    print(f"Mismatches:   {n_mismatch:,}")

    if n_mismatch == 0:
        print("  OK: Label_matrix (Success/Fail) and Label_truth (Normal/Anomaly) agree perfectly.")
    else:
        pct = 100 * n_mismatch / len(merged)
        print(f"  WARNING: {pct:.2f}% mismatch between the two label columns.")
        print("  Example mismatched rows:")
        print(merged.loc[mismatch, ["BlockId", "Label_matrix", "Label_truth"]].head(5))
        print("  We will still use Label_truth (from anomaly_label.csv) as ground truth,")
        print("  since that file is the hand-labeled one described in the paper.")


def extract_event_count_matrix(merged):
    """
    Pulls out the E1..E29 columns as the raw event count matrix X,
    corresponding to Equation (1) in the paper: x_i = count(v_i in s)
    """
    event_cols = [c for c in merged.columns if c.startswith("E") and c[1:].isdigit()]
    event_cols_sorted = sorted(event_cols, key=lambda c: int(c[1:]))
    print(f"\nFound {len(event_cols_sorted)} event columns: {event_cols_sorted}")

    X_counts = merged[event_cols_sorted].to_numpy(dtype=float)
    block_ids = merged["BlockId"].to_numpy()
    y = (merged["Label_truth"] == "Anomaly").astype(int).to_numpy()

    return X_counts, y, block_ids, event_cols_sorted


def compute_tfidf(X_counts):
    """
    Implements Equation (2) from the paper directly (not sklearn's TF-IDF,
    since the paper's idf formula uses a specific smoothing: log(N / (1+df))
    without the "+1" on the log itself, which differs slightly from sklearn's
    default). This keeps us matching the paper exactly.

    tf(v_i, s)  = count(v_i in s) / sum_j count(v_j in s)
    idf(v_i)    = log( N / (1 + df(v_i)) )
    x_i         = tf(v_i, s) * idf(v_i)
    """
    n_sequences, n_events = X_counts.shape

    # tf: normalize each row (sequence) by its total event count
    row_sums = X_counts.sum(axis=1, keepdims=True)
    # avoid division by zero for any all-zero sequence (shouldn't happen, but be safe)
    row_sums[row_sums == 0] = 1
    tf = X_counts / row_sums

    # df: for each event column, how many sequences contain it at least once
    df = (X_counts > 0).sum(axis=0)

    # idf: log(N / (1 + df))
    idf = np.log(n_sequences / (1 + df))

    tfidf = tf * idf  # broadcasts idf across all rows

    print(f"\nTF-IDF matrix shape: {tfidf.shape}")
    print(f"TF-IDF value range: [{tfidf.min():.4f}, {tfidf.max():.4f}]")

    return tfidf


def main():
    merged = load_and_merge()
    check_label_agreement(merged)
    X_counts, y, block_ids, event_cols = extract_event_count_matrix(merged)
    X_tfidf = compute_tfidf(X_counts)

    print(f"\n{'=' * 50}")
    print("SAVING OUTPUT")
    print(f"{'=' * 50}")
    np.savez(
        OUTPUT_PATH,
        X_counts=X_counts,
        X_tfidf=X_tfidf,
        y=y,
        block_ids=block_ids,
        event_cols=np.array(event_cols),
    )
    print(f"Saved to {OUTPUT_PATH}")
    print(f"  X_counts: {X_counts.shape} (raw event count vectors, Eq. 1)")
    print(f"  X_tfidf:  {X_tfidf.shape} (TF-IDF vectors, Eq. 2 - used for PCA/RPCA)")
    print(f"  y:        {y.shape} (1 = anomaly, 0 = normal)")
    print(f"  Anomaly ratio: {y.mean() * 100:.2f}%")


if __name__ == "__main__":
    main()