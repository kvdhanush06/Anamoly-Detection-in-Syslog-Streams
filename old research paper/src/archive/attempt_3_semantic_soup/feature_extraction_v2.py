"""
Extended feature extraction for HDFS_v1 — v2.

Produces features_v2.npz containing:
  X_unigram  (n, 29)   — L2-normalised raw event counts (same as v1 X_counts after norm)
  X_bigram   (n, B)    — L2-normalised bigram (adjacent-pair) transition counts
  X_combined (n, 29+B+4) — unigram + bigram + time-interval stats, L2-normalised
  X_time     (n, 4)    — time-interval aggregates (mean, std, max, total)
  y          (n,)      — 1=anomaly, 0=normal
  block_ids  (n,)      — BlockId strings
  bigram_cols          — list of "Ei->Ej" labels for the bigram columns

Source files used:
  data/HDFS_v1/preprocessed/Event_occurrence_matrix.csv  (unigram counts)
  data/HDFS_v1/preprocessed/Event_traces.csv             (ordered seqs + intervals)
  data/HDFS_v1/preprocessed/anomaly_label.csv            (ground truth)

Run from project root with venv activated:
    python src/feature_extraction_v2.py
"""

import ast
import re
import numpy as np
import pandas as pd
from itertools import combinations
from collections import defaultdict

LABEL_PATH  = "data/HDFS_v1/preprocessed/anomaly_label.csv"
MATRIX_PATH = "data/HDFS_v1/preprocessed/Event_occurrence_matrix.csv"
TRACES_PATH = "data/HDFS_v1/preprocessed/Event_traces.csv"
OUTPUT_PATH = "data/HDFS_v1/preprocessed/features_v2.npz"


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def l2_normalize(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


def parse_event_list(s):
    """
    Parse a string like "[E5,E22,E5,E11]" into ["E5","E22","E5","E11"].
    Handles both quoted and unquoted formats.
    """
    s = s.strip()
    if s.startswith("["):
        # remove brackets, split on commas, strip whitespace/quotes
        inner = s[1:-1]
        parts = [p.strip().strip("'\"") for p in inner.split(",")]
        return [p for p in parts if p]
    return []


def parse_interval_list(s):
    """
    Parse a string like "[0.0, 1.0, 0.0, ...]" into a numpy array of floats.
    """
    s = s.strip()
    if s.startswith("["):
        inner = s[1:-1]
        try:
            vals = [float(v.strip()) for v in inner.split(",") if v.strip()]
            return np.array(vals, dtype=np.float64)
        except ValueError:
            return np.array([], dtype=np.float64)
    return np.array([], dtype=np.float64)


# ---------------------------------------------------------------------------
# Step 1 — Load unigram counts + labels (same as feature_extraction.py)
# ---------------------------------------------------------------------------

def load_unigram_and_labels():
    print("Loading unigram counts and labels...")
    labels_df = pd.read_csv(LABEL_PATH)
    matrix_df = pd.read_csv(MATRIX_PATH)

    merged = matrix_df.merge(labels_df, on="BlockId", how="inner",
                              suffixes=("_matrix", "_truth"))
    print(f"  merged shape: {merged.shape}")

    event_cols = sorted(
        [c for c in merged.columns if c.startswith("E") and c[1:].isdigit()],
        key=lambda c: int(c[1:])
    )
    print(f"  event columns ({len(event_cols)}): {event_cols}")

    X_counts  = merged[event_cols].to_numpy(dtype=float)
    y         = (merged["Label_truth"] == "Anomaly").astype(int).to_numpy()
    block_ids = merged["BlockId"].to_numpy()

    return X_counts, y, block_ids, event_cols, merged


# ---------------------------------------------------------------------------
# Step 2 — Parse Event_traces.csv to build bigrams + time features
# ---------------------------------------------------------------------------

def build_bigram_and_time_features(block_ids, event_cols):
    """
    Reads Event_traces.csv row-by-row and builds:
      bigram_counts : dict {block_id: Counter of (Ei, Ej) adjacent pairs}
      time_feats    : dict {block_id: [mean_interval, std_interval, max_interval, latency]}

    Returns:
      bigram_matrix  (n, B) — raw bigram counts aligned to block_ids order
      time_matrix    (n, 4)
      bigram_labels  list of "Ei->Ej" strings (length B)
    """
    print("\nParsing Event_traces.csv for bigrams and time features...")
    print("  (this may take ~30s for 575k rows)")

    # index block_ids for fast lookup
    bid_to_idx = {bid: i for i, bid in enumerate(block_ids)}
    n = len(block_ids)

    # We'll collect all observed bigram pairs first, then build the matrix
    bigram_counts_raw = defaultdict(lambda: defaultdict(int))  # idx → pair → count
    time_raw = {}   # idx → [mean, std, max, latency]

    traces_df = pd.read_csv(TRACES_PATH)
    print(f"  loaded {len(traces_df):,} rows")

    # Discover all bigram types in one pass
    all_bigrams = set()

    for _, row in traces_df.iterrows():
        bid = row["BlockId"]
        if bid not in bid_to_idx:
            continue
        idx = bid_to_idx[bid]

        # --- parse event sequence ---
        feats_str = str(row.get("Features", "[]"))
        events = parse_event_list(feats_str)

        # bigrams
        for i in range(len(events) - 1):
            pair = (events[i], events[i+1])
            bigram_counts_raw[idx][pair] += 1
            all_bigrams.add(pair)

        # --- parse time intervals ---
        ti_str  = str(row.get("TimeInterval", "[]"))
        intervals = parse_interval_list(ti_str)
        latency   = float(row.get("Latency", 0) or 0)

        if len(intervals) > 0:
            time_raw[idx] = [
                float(np.mean(intervals)),
                float(np.std(intervals)),
                float(np.max(intervals)),
                latency,
            ]
        else:
            time_raw[idx] = [0.0, 0.0, 0.0, latency]

    # Sort bigram pairs for stable column ordering
    all_bigrams_sorted = sorted(all_bigrams)
    B = len(all_bigrams_sorted)
    bigram_to_col = {pair: j for j, pair in enumerate(all_bigrams_sorted)}
    bigram_labels = [f"{a}->{b}" for a, b in all_bigrams_sorted]
    print(f"  unique bigrams found: {B}")

    # Build dense matrices
    bigram_matrix = np.zeros((n, B), dtype=np.float32)
    time_matrix   = np.zeros((n, 4),  dtype=np.float32)

    for idx, pair_counts in bigram_counts_raw.items():
        for pair, cnt in pair_counts.items():
            bigram_matrix[idx, bigram_to_col[pair]] = cnt

    for idx, tfeats in time_raw.items():
        time_matrix[idx] = tfeats

    # Rows with no trace data (shouldn't happen, but guard)
    missing = n - len(time_raw)
    if missing > 0:
        print(f"  WARNING: {missing} blocks have no trace data — time features = 0")

    return bigram_matrix, time_matrix, bigram_labels


# ---------------------------------------------------------------------------
# Step 3 — Assemble combined feature matrix
# ---------------------------------------------------------------------------

def main():
    # --- unigrams ---
    X_counts, y, block_ids, event_cols, _ = load_unigram_and_labels()

    # --- bigrams + time ---
    X_bigram_raw, X_time_raw, bigram_labels = build_bigram_and_time_features(
        block_ids, event_cols
    )

    # --- normalise time features to [0,1] per column (robust to scale) ---
    time_max = X_time_raw.max(axis=0)
    time_max[time_max == 0] = 1.0
    X_time_norm = X_time_raw / time_max

    # --- L2-normalise each feature type independently ---
    X_unigram_l2  = l2_normalize(X_counts.astype(np.float32))
    X_bigram_l2   = l2_normalize(X_bigram_raw)

    # combined: [unigram | bigram | time]
    X_combined_raw = np.hstack([X_counts.astype(np.float32),
                                 X_bigram_raw,
                                 X_time_norm])
    X_combined_l2  = l2_normalize(X_combined_raw)

    print(f"\n{'=' * 55}")
    print("FEATURE SHAPES")
    print(f"{'=' * 55}")
    print(f"  X_unigram_l2  : {X_unigram_l2.shape}")
    print(f"  X_bigram_raw  : {X_bigram_raw.shape}   ({len(bigram_labels)} bigram types)")
    print(f"  X_bigram_l2   : {X_bigram_l2.shape}")
    print(f"  X_time_norm   : {X_time_norm.shape}")
    print(f"  X_combined_l2 : {X_combined_l2.shape}")
    print(f"  y             : {y.shape}   anomaly ratio={y.mean()*100:.2f}%")

    # Explained-variance check: how many unigram dims vs bigram dims matter?
    print(f"\n  Top-10 most frequent bigrams (overall count):")
    bigram_totals = X_bigram_raw.sum(axis=0)
    top10 = np.argsort(bigram_totals)[::-1][:10]
    for j in top10:
        print(f"    {bigram_labels[j]:15s}  total={int(bigram_totals[j]):>12,}")

    print(f"\n{'=' * 55}")
    print("SAVING")
    print(f"{'=' * 55}")
    np.savez_compressed(
        OUTPUT_PATH,
        X_unigram_l2=X_unigram_l2,
        X_bigram_l2=X_bigram_l2,
        X_bigram_raw=X_bigram_raw,
        X_time_norm=X_time_norm,
        X_combined_l2=X_combined_l2,
        y=y,
        block_ids=block_ids,
        bigram_labels=np.array(bigram_labels),
        event_cols=np.array(event_cols),
    )
    print(f"  Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
