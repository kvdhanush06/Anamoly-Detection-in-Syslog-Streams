"""
Real Zenodo Dataset Loader for BertGCN.

Loads the actual HDFS and BGL log datasets downloaded from Zenodo
(https://zenodo.org/record/3227177) and stored in:
  - data/HDFS_v1/HDFS.log              (raw log lines)
  - data/HDFS_v1/anomaly_label.csv     (BlockId, Label columns)
  - data/BGL/BGL.log                   (raw log lines; label in col 0)

Both datasets are the standard Loghub benchmarks used across all log anomaly
detection research including Alsalmi et al. (CMC 2025/2026).

Dataset Notes:
  HDFS:
    - Session unit: Block (identified by blk_<id> in log line).
    - anomaly_label.csv maps BlockId → "Anomaly" or "Normal".
    - Returns: List[(messages_for_block, label)] — one entry per block.

  BGL:
    - Session unit: Fixed-size sliding window of `window_size` consecutive lines.
    - A window is anomalous if ANY line in it has a non-"-" label in column 0.
    - Returns: List[(messages_in_window, label)] — one entry per window.

The paths are resolved relative to this file's location:
  this_file → src/baseline_replication/data/real_data_loader.py
  ROOT      → <repo_root>/
  DATA_DIR  → <repo_root>/data/
"""
import os
import re
import csv
from typing import List, Tuple

# ── Path resolution ────────────────────────────────────────────────────────────
_THIS_DIR   = os.path.dirname(os.path.abspath(__file__))         # data/
_BASELINE_DIR = os.path.dirname(_THIS_DIR)                        # baseline_replication/
_SRC_DIR    = os.path.dirname(_BASELINE_DIR)                     # src/
_PAPER_DIR  = os.path.dirname(_SRC_DIR)                          # new research paper/
_ROOT_DIR   = os.path.dirname(_PAPER_DIR)                        # repo root
_DATA_DIR   = os.path.join(_ROOT_DIR, "data")                   # data/

HDFS_LOG_PATH    = os.path.join(_DATA_DIR, "HDFS_v1", "HDFS.log")
HDFS_LABEL_PATH  = os.path.join(_DATA_DIR, "HDFS_v1", "anomaly_label.csv")
BGL_LOG_PATH     = os.path.join(_DATA_DIR, "BGL", "BGL.log")


# ── HDFS Loader ───────────────────────────────────────────────────────────────

def _extract_block_ids(line: str) -> List[str]:
    """Extracts all blk_<id> identifiers from a single log line."""
    return re.findall(r"(blk_-?\d+)", line)


def load_hdfs_real_sequences(
    max_blocks: int = None,
) -> List[Tuple[List[str], int]]:
    """
    Loads HDFS sessions (blocks) from the real Zenodo log files.

    Each HDFS block is a session: all log lines referencing blk_<id>
    form that block's sequence. Labels come from anomaly_label.csv.

    Args:
        max_blocks: If set, limits the number of blocks returned (useful for
                    faster prototyping). None = load all blocks.

    Returns:
        List of (log_message_list, label) tuples.
        label: 0 = Normal, 1 = Anomaly.

    Raises:
        FileNotFoundError: If HDFS.log or anomaly_label.csv are missing.
    """
    if not os.path.isfile(HDFS_LOG_PATH):
        raise FileNotFoundError(
            f"HDFS log not found at: {HDFS_LOG_PATH}\n"
            "Download from https://zenodo.org/record/3227177 and extract to data/HDFS_v1/"
        )
    if not os.path.isfile(HDFS_LABEL_PATH):
        raise FileNotFoundError(
            f"HDFS labels not found at: {HDFS_LABEL_PATH}\n"
            "Download anomaly_label.csv and place in data/HDFS_v1/"
        )

    # 1. Load ground-truth labels
    block_labels: dict = {}
    with open(HDFS_LABEL_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            bid = row["BlockId"].strip()
            lbl = 1 if row["Label"].strip() == "Anomaly" else 0
            block_labels[bid] = lbl

    # 2. Group log lines by block ID
    block_messages: dict = {}
    with open(HDFS_LOG_PATH, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            bids = _extract_block_ids(line)
            for bid in bids:
                if bid not in block_messages:
                    block_messages[bid] = []
                block_messages[bid].append(line)

    # 3. Build (sequence, label) pairs — only for blocks with known labels
    sequences: List[Tuple[List[str], int]] = []
    for bid, msgs in block_messages.items():
        if bid not in block_labels:
            continue
        sequences.append((msgs, block_labels[bid]))
        if max_blocks is not None and len(sequences) >= max_blocks:
            break

    print(
        f"[HDFS Loader] Loaded {len(sequences)} blocks "
        f"({sum(1 for _, l in sequences if l == 1)} anomalous, "
        f"{sum(1 for _, l in sequences if l == 0)} normal)"
    )
    return sequences


# ── BGL Loader ────────────────────────────────────────────────────────────────

def load_bgl_real_sequences(
    window_size: int = 20,
    step_size: int = None,
    max_windows: int = None,
) -> List[Tuple[List[str], int]]:
    """
    Loads BGL log sessions as fixed-size sliding windows from the real Zenodo log file.

    BGL label convention:
      - Column 0 of each line: "-" = Normal, anything else = Anomaly.
      - A WINDOW is labelled Anomaly if ANY line in it is anomalous.

    Args:
        window_size: Number of consecutive lines per session (default: 20).
        step_size:   Sliding step between windows (default: window_size, i.e., non-overlapping).
        max_windows: Optional limit on the total number of windows returned.

    Returns:
        List of (log_message_list, label) tuples.
        label: 0 = Normal, 1 = Anomaly.

    Raises:
        FileNotFoundError: If BGL.log is missing.
    """
    if not os.path.isfile(BGL_LOG_PATH):
        raise FileNotFoundError(
            f"BGL log not found at: {BGL_LOG_PATH}\n"
            "Download from https://zenodo.org/record/3227177 and extract to data/BGL/"
        )

    if step_size is None:
        step_size = window_size  # non-overlapping windows by default

    # 1. Read all lines and extract (label_flag, full_message) pairs
    raw_lines: List[Tuple[int, str]] = []
    with open(BGL_LOG_PATH, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) == 0:
                continue
            label_col = parts[0].strip()
            message = parts[1].strip() if len(parts) > 1 else label_col
            is_anomaly = 0 if label_col == "-" else 1
            raw_lines.append((is_anomaly, message))

    # 2. Slide window over all lines
    sequences: List[Tuple[List[str], int]] = []
    total_lines = len(raw_lines)

    i = 0
    while i + window_size <= total_lines:
        window = raw_lines[i: i + window_size]
        window_label = 1 if any(lbl == 1 for lbl, _ in window) else 0
        messages = [msg for _, msg in window]
        sequences.append((messages, window_label))

        if max_windows is not None and len(sequences) >= max_windows:
            break
        i += step_size

    print(
        f"[BGL Loader] Loaded {len(sequences)} windows (size={window_size}, step={step_size}) "
        f"from {total_lines} total lines — "
        f"{sum(1 for _, l in sequences if l == 1)} anomalous, "
        f"{sum(1 for _, l in sequences if l == 0)} normal"
    )
    return sequences
