"""
Loads the SisFall Enhanced ("Three Classes") dataset and converts each
raw motion window into a small set of features a model can learn from.

File format (reverse-engineered -- dataset ships with zero documentation):
  - x_train_3 / x_val_3 / x_test_3: raw float32 values, no header.
    Reshape to (n_samples, 512, 3) -- 512 timesteps, 3 axes (X, Y, Z).
  - y_train_3 / y_val_3 / y_test_3: raw uint8 values, no header.
    Reshape to (n_samples, 3) -- one-hot encoded:
      column 0 = non-fall, column 1 = pre-fall, column 2 = fall

THE CORE IDEA (in plain words):
  Each sample is a short recording of "how hard is the phone shaking"
  over time. Walking = gentle wiggle the whole time. Falling = calm,
  then one sudden violent spike (the impact). We turn that wiggly
  line into 5 simple numbers:
    1. Highest point?          -> smv_max
    2. Lowest point?           -> smv_min
    3. How jumpy overall?      -> smv_std
    4. Average level?          -> smv_mean
    5. Fastest single jump?    -> jerk_max
"""

import numpy as np
import pandas as pd
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

    smv_max = smv.max()
    smv_min = smv.min()
    smv_std = smv.std()
    smv_mean = smv.mean()

    jerk = np.diff(smv)
    jerk_max = jerk.max()

    return {
        "smv_max": smv_max,
        "smv_min": smv_min,
        "smv_std": smv_std,
        "smv_mean": smv_mean,
        "jerk_max": jerk_max,
    }


def build_feature_dataframe(X: np.ndarray, y_class: np.ndarray) -> pd.DataFrame:
    rows = []
    for i in range(X.shape[0]):
        features = extract_features_for_window(X[i])
        features["label"] = y_class[i]
        rows.append(features)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        print(f"Loading {split}...")
        X, y_class = load_split(split)
        print(f"  {split}: {X.shape[0]} samples")

        print(f"  Extracting features for {split}...")
        df = build_feature_dataframe(X, y_class)

        out_path = f"features_{split}.csv"
        df.to_csv(out_path, index=False)
        print(f"  Saved {out_path}")
        print(df.head())
        print()
