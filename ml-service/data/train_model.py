"""
Trains a Random Forest to classify motion windows as non-fall, pre-fall, or fall.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

FEATURE_COLS = ["smv_max", "smv_min", "smv_std", "smv_mean", "jerk_max"]

# Step 1: load the already-split data
train_df = pd.read_csv("features_train.csv")
val_df = pd.read_csv("features_val.csv")
test_df = pd.read_csv("features_test.csv")

X_train, y_train = train_df[FEATURE_COLS], train_df["label"]
X_val, y_val = val_df[FEATURE_COLS], val_df["label"]
X_test, y_test = test_df[FEATURE_COLS], test_df["label"]

# Step 2: train the "100 voters" model
# n_estimators=100 means literally 100 trees voting
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("Training done.")

# Step 3: check it on validation data (our "practice exam before the real one")
val_preds = model.predict(X_val)
print("\n--- VALIDATION RESULTS ---")
print(classification_report(y_val, val_preds, target_names=["non-fall", "pre-fall", "fall"]))

# Step 4: which features did the model actually rely on?
print("\n--- FEATURE IMPORTANCE ---")
for name, importance in zip(FEATURE_COLS, model.feature_importances_):
    print(f"{name}: {importance:.3f}")

# Step 5: save the trained model to disk
joblib.dump(model, "../models/fall_detector.joblib")
print("\nModel saved to ../models/fall_detector.joblib")
