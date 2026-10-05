"""
Real feature extraction logic, matching the v2 model (10 features).
Used both for training (via data/load_and_extract_v2.py duplicating
this logic) and for live prediction in app.py.
"""

import numpy as np
from scipy.stats import skew, kurtosis

FEATURE_COLS = ["smv_max", "smv_min", "smv_std", "smv_mean", "jerk_max",
                "smv_skew", "smv_kurtosis", "x_std", "y_std", "z_std"]


def extract_window_features(window_df) -> dict:
    """
    window_df: a pandas DataFrame with columns AccX, AccY, AccZ,
    covering one window of motion (ideally 512 rows to match training,
    though the model will still run on other lengths -- just less
    reliably, since it wasn't trained on that).
    """
    window = window_df[["AccX", "AccY", "AccZ"]].to_numpy()
    smv = np.linalg.norm(window, axis=1)

    features = {
        "smv_max": smv.max(),
        "smv_min": smv.min(),
        "smv_std": smv.std(),
        "smv_mean": smv.mean(),
        "jerk_max": np.diff(smv).max(),
        "smv_skew": skew(smv),
        "smv_kurtosis": kurtosis(smv),
        "x_std": window[:, 0].std(),
        "y_std": window[:, 1].std(),
        "z_std": window[:, 2].std(),
    }
    return features
