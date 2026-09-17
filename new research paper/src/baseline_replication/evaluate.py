"""
Evaluation utilities: Precision, Recall, F1-Score, and Accuracy.
Section 4.5 & 5.3 of Alsalmi et al.
"""
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from typing import Dict, Any

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes standard anomaly classification metrics.
    Positive class is 1 (Anomaly).
    """
    prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn)
    }

def print_metrics_table(title: str, metrics: Dict[str, float]):
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")
    print(f" Accuracy : {metrics['accuracy'] * 100:.2f}%")
    print(f" Precision: {metrics['precision'] * 100:.2f}%")
    print(f" Recall   : {metrics['recall'] * 100:.2f}%")
    print(f" F1-Score : {metrics['f1'] * 100:.2f}%")
    if 'tp' in metrics:
        print(f" Confusion: TP={metrics['tp']}, FP={metrics['fp']}, TN={metrics['tn']}, FN={metrics['fn']}")
    print(f"{'='*60}\n")
