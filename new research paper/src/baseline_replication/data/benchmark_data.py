"""
Benchmark Data Generator — DEPRECATED / Archive Reference Only.

This module previously generated SYNTHETIC log sequences for quick smoke-testing
before the real Zenodo datasets were integrated.

STATUS: This module is NO LONGER USED by any active script. Both run_baseline.py
and run_improvements.py now load the REAL Zenodo HDFS and BGL datasets via
real_data_loader.py.

This file is kept in the data/ package solely so that the archived
src/archive/run_experiments.py (which references it) does not cause ImportErrors
when inspected. The archive script itself redirects to the real loader.

DO NOT use generate_benchmark_sequences() for any new code.
"""
import warnings
from typing import List, Tuple

from .real_data_loader import load_hdfs_real_sequences, load_bgl_real_sequences


def generate_benchmark_sequences(
    dataset: str = "HDFS",
    num_normal: int = 500,
    num_anomalies: int = 60,
    random_state: int = 42,
    window_size: int = 20,
    max_samples: int = None,
) -> List[Tuple[List[str], int]]:
    """
    [DEPRECATED] Previously generated synthetic sequences for smoke-testing.

    Now transparently redirects to the real Zenodo dataset loaders so that
    any code calling this function automatically uses real data.

    Args:
        dataset:      "HDFS" or "BGL".
        num_normal:   (ignored) Kept for backward compatibility.
        num_anomalies:(ignored) Kept for backward compatibility.
        random_state: (ignored) Kept for backward compatibility.
        window_size:  Window size for BGL (default 20, as in paper).
        max_samples:  Optional cap on total sequences returned.

    Returns:
        Real dataset sequences as List[(messages, label)].
    """
    warnings.warn(
        "generate_benchmark_sequences() is deprecated. "
        "Real Zenodo data is now loaded directly via real_data_loader.py. "
        "This call will transparently redirect to the real loader.",
        DeprecationWarning,
        stacklevel=2,
    )

    if dataset.upper() == "HDFS":
        return load_hdfs_real_sequences(max_blocks=max_samples)
    elif dataset.upper() == "BGL":
        return load_bgl_real_sequences(window_size=window_size, max_windows=max_samples)
    else:
        raise ValueError(f"Unknown dataset '{dataset}'. Expected 'HDFS' or 'BGL'.")
