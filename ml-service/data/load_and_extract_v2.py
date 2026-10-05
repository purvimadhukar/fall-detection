"""
v2: adds per-axis std, skewness, and kurtosis on top of the original
5 features, to see if they sharpen the fall vs pre-fall vs non-fall boundary.
"""

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
import os

DATA_DIR = "SisFall_dataset/Three Classes"
WINDOW_LEN = 512
N_AXES = 3


def load_split(split_name: str):
    x_path = os.path.join(DATA_DIR, f"x_{split_name}_3")
    y_path = os.path.join(DATA_DIR, f"y_{split_name}_3")

    raw_x = np.fromfile(x_path, dtype=np.float32)
    raw_y = np.fromfile(y_path, dtype=np.uint8)

    n_samples = raw_y.shape[0] // 3

    X = raw_x.reshape(n_samples, WINDOW_LEN, N_AXES)
    Y_onehot = raw_y.reshape(n_samples, 3)
    y_class = np.argmax(Y_onehot, axis=1)

    return X, y_class


def extract_features_for_window(window: np.ndarray) -> dict:
    smv = np.linalg.norm(window, axis=1)

    features = {
        "smv_max": smv.max(),
        "smv_min": smv.min(),
        "smv_std": smv.std(),
        "smv_mean": smv.mean(),
        "jerk_max": np.diff(smv).max(),
        # NEW: how lopsided is the shape? (fall = calm + one spike = lopsided)
        "smv_skew": skew(smv),
        # NEW: how peaked/extreme is the shape? (fall = one extreme outlier)
        "smv_kurtosis": kurtosis(smv),
    }

    # NEW: per-axis std (direction-specific info, not just combined magnitude)
    features["x_std"] = window[:, 0].std()
    features["y_std"] = window[:, 1].std()
    features["z_std"] = window[:, 2].std()

    return features


def build_feature_dataframe(X: np.ndarray, y_class: np.ndarray) -> pd.DataFrame:
    rows = []
    for i in range(X.shape[0]):
        f = extract_features_for_window(X[i])
        f["label"] = y_class[i]
        rows.append(f)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        print(f"Loading {split}...")
        X, y_class = load_split(split)
        df = build_feature_dataframe(X, y_class)
        out_path = f"features_{split}_v2.csv"
        df.to_csv(out_path, index=False)
        print(f"  Saved {out_path} ({df.shape[0]} rows, {df.shape[1]} cols)")
        print(df.head(3))
        print()
