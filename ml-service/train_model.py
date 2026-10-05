"""
Train the fall detection classifier.

Once feature_extraction.py gives you a clean DataFrame of features + labels,
this script trains and evaluates a model.

WHY Random Forest (or SVM) instead of a neural net for v1:
  - Small-ish dataset (SisFall isn't huge) -- simpler models generalize
    better with less data
  - Interpretable: you can inspect feature_importances_ and literally
    show, in an interview, "here's proof smv_max and jerk_max are the
    most predictive features" -- ties your ML choice back to the domain
    reasoning, which is a strong signal of understanding vs. just
    calling a library
  - Fast to train/iterate, runs fine on any laptop, no GPU needed

You can add a neural net as a "phase 2 comparison" later if you want to
show you explored both -- but don't start there.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib


def load_feature_dataset(path: str) -> pd.DataFrame:
    """TODO: load the CSV/parquet you saved from build_feature_dataset()."""
    raise NotImplementedError


def train_and_evaluate(df: pd.DataFrame):
    """
    TODO:
    1. Split into X (features) and y (label column)
    2. train_test_split -- IMPORTANT: split by TRIAL/subject, not by
       individual window. If windows from the same trial end up in both
       train and test sets, your test accuracy will be misleadingly high
       (data leakage) because adjacent windows are highly similar.
       Think about how to do a "grouped" split (hint: look into
       sklearn's GroupShuffleSplit, grouping by subject ID or trial ID).
    3. Train a RandomForestClassifier
    4. Evaluate with classification_report and confusion_matrix

    IMPORTANT ON METRICS: for this problem, a missed fall (false negative)
    is far worse than a false alarm (false positive) -- a missed fall
    could mean someone doesn't get help. So don't just optimize for
    accuracy. Look at RECALL on the fall class specifically, and consider
    whether you should bias the model/threshold toward catching more
    falls even at the cost of more false alarms. This is a real product
    decision you should be able to justify in an interview.
    5. Save the trained model with joblib.dump()
    """
    raise NotImplementedError


if __name__ == "__main__":
    # df = load_feature_dataset("data/features.csv")
    # train_and_evaluate(df)
    pass
