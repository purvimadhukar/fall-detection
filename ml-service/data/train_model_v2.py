"""
Retrains using the expanded 8-feature set (v2) and compares against
the original 5-feature model.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

FEATURE_COLS = ["smv_max", "smv_min", "smv_std", "smv_mean", "jerk_max",
                "smv_skew", "smv_kurtosis", "x_std", "y_std", "z_std"]

train_df = pd.read_csv("features_train_v2.csv")
val_df = pd.read_csv("features_val_v2.csv")

X_train, y_train = train_df[FEATURE_COLS], train_df["label"]
X_val, y_val = val_df[FEATURE_COLS], val_df["label"]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("Training done.")

val_preds = model.predict(X_val)
print("\n--- VALIDATION RESULTS (v2, 10 features) ---")
print(classification_report(y_val, val_preds, target_names=["non-fall", "pre-fall", "fall"]))

print("\n--- CONFUSION MATRIX ---")
print("Rows = actual class, Columns = predicted class")
print("Order: [non-fall, pre-fall, fall]")
print(confusion_matrix(y_val, val_preds))

print("\n--- FEATURE IMPORTANCE ---")
importances = sorted(zip(FEATURE_COLS, model.feature_importances_), key=lambda x: -x[1])
for name, importance in importances:
    print(f"{name}: {importance:.3f}")

joblib.dump(model, "../models/fall_detector_v2.joblib")
print("\nModel saved.")
