import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

DATA_PATH = "data/BGL/preprocessed/bgl_sequences.pkl"
STRUCTURED_CSV = "data/BGL/preprocessed/BGL.log_structured.csv"

def load_data():
    """Loads BGL sequences and binary labels strictly read-only."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Cannot find {DATA_PATH}")
    with open(DATA_PATH, "rb") as f:
        data = pickle.load(f)
    sequences = np.array(data["sequences"], dtype=object)
    y = np.array(data["y"], dtype=int)
    return sequences, y

def get_10fold_splits(y, seed=42):
    """
    Creates a 10-fold split done SEPARATELY on normal windows and on anomalous windows,
    so each fold's test set has natural prevalence (~10% anomalies), and every anomaly
    is tested exactly once across the 10 folds.
    
    Inside non-test portion:
      - TRAIN: 90% of non-test normals (model never sees anomalies)
      - VALIDATION: remaining 10% of non-test normals + random subsample (max 20,000) of non-test anomalies
    """
    normal_idx = np.where(y == 0)[0]
    anom_idx = np.where(y == 1)[0]
    
    kf_norm = KFold(n_splits=10, shuffle=True, random_state=seed)
    kf_anom = KFold(n_splits=10, shuffle=True, random_state=seed)
    
    splits = []
    
    norm_splits = list(kf_norm.split(normal_idx))
    anom_splits = list(kf_anom.split(anom_idx))
    
    for fold in range(10):
        train_norm_fold, test_norm_fold = norm_splits[fold]
        train_anom_fold, test_anom_fold = anom_splits[fold]
        
        test_norm_idx = normal_idx[test_norm_fold]
        test_anom_idx = anom_idx[test_anom_fold]
        test_idx = np.concatenate([test_norm_idx, test_anom_idx])
        
        non_test_norm_idx = normal_idx[train_norm_fold]
        non_test_anom_idx = anom_idx[train_anom_fold]
        
        # 90% train normals, 10% val normals
        kf_inner = KFold(n_splits=10, shuffle=True, random_state=seed + fold + 1)
        train_sub_idx, val_norm_sub_idx = next(kf_inner.split(non_test_norm_idx))
        
        train_idx = non_test_norm_idx[train_sub_idx]
        val_norm_idx = non_test_norm_idx[val_norm_sub_idx]
        
        # Validation anomalies: max 20,000 from non_test_anom_idx
        rng = np.random.default_rng(seed + fold + 100)
        n_val_anom = min(20000, len(non_test_anom_idx))
        val_anom_idx = rng.choice(non_test_anom_idx, n_val_anom, replace=False)
        
        val_idx = np.concatenate([val_norm_idx, val_anom_idx])
        
        # Strict Disjointness Assertions
        assert len(set(train_idx) & set(val_idx)) == 0, f"Fold {fold+1}: Train and Val overlap!"
        assert len(set(train_idx) & set(test_idx)) == 0, f"Fold {fold+1}: Train and Test overlap!"
        assert len(set(val_idx) & set(test_idx)) == 0, f"Fold {fold+1}: Val and Test overlap!"
        assert np.all(y[train_idx] == 0), f"Fold {fold+1}: Anomalies present in Train set!"
        assert np.all(y[val_norm_idx] == 0), f"Fold {fold+1}: Anomalies present in Val normal set!"
        
        splits.append({
            "fold": fold + 1,
            "train_idx": train_idx,
            "val_norm_idx": val_norm_idx,
            "val_anom_idx": val_anom_idx,
            "val_idx": val_idx,
            "test_norm_idx": test_norm_idx,
            "test_anom_idx": test_anom_idx,
            "test_idx": test_idx
        })
        
    return splits

def get_chronological_split(y):
    """
    Sorts windows by timestamp; trains on earliest 70% of normals,
    validates on next 10% of normals (+ subsample anomalies),
    tests on latest 20% (all its normals and anomalies).
    """
    print("Loading timestamps for chronological split...")
    df = pd.read_csv(STRUCTURED_CSV, usecols=["Timestamp", "Node"])
    window_size = 6 * 3600
    df["Window_ID"] = df["Timestamp"] // window_size
    grouped = df.groupby(["Node", "Window_ID"], sort=True)
    
    # Each sequence corresponds to one (Node, Window_ID) group in sorted order
    keys = list(grouped.groups.keys())
    window_ids = np.array([k[1] for k in keys], dtype=np.int32)
    
    # Sort order by window_id (timestamp)
    sort_order = np.argsort(window_ids, kind="stable")
    
    # Extract normal and anomaly indices in chronological order
    y_sorted = y[sort_order]
    
    normal_sorted = sort_order[np.where(y_sorted == 0)[0]]
    anom_sorted = sort_order[np.where(y_sorted == 1)[0]]
    
    n_norm = len(normal_sorted)
    cut1 = int(0.70 * n_norm)
    cut2 = int(0.80 * n_norm)
    
    train_idx = normal_sorted[:cut1]
    val_norm_idx = normal_sorted[cut1:cut2]
    test_norm_idx = normal_sorted[cut2:]
    
    # For anomalies, find the timestamp threshold for cut2 (the latest 20% time period)
    last_val_time = window_ids[val_norm_idx[-1]]
    
    # Anomalies occurring after last_val_time are in test; before are eligible for val
    test_anom_mask = window_ids[anom_sorted] > last_val_time
    test_anom_idx = anom_sorted[test_anom_mask]
    non_test_anom_idx = anom_sorted[~test_anom_mask]
    
    rng = np.random.default_rng(42)
    n_val_anom = min(20000, len(non_test_anom_idx))
    val_anom_idx = rng.choice(non_test_anom_idx, n_val_anom, replace=False) if n_val_anom > 0 else np.array([], dtype=int)
    
    val_idx = np.concatenate([val_norm_idx, val_anom_idx])
    test_idx = np.concatenate([test_norm_idx, test_anom_idx])
    
    assert len(set(train_idx) & set(val_idx)) == 0
    assert len(set(train_idx) & set(test_idx)) == 0
    assert len(set(val_idx) & set(test_idx)) == 0
    assert np.all(y[train_idx] == 0)
    
    return {
        "train_idx": train_idx,
        "val_norm_idx": val_norm_idx,
        "val_anom_idx": val_anom_idx,
        "val_idx": val_idx,
        "test_norm_idx": test_norm_idx,
        "test_anom_idx": test_anom_idx,
        "test_idx": test_idx
    }

