"""
Flask ML service. Serves the trained fall-detection model over HTTP.

This file is mostly plumbing (already written for you) -- the real work
is in feature_extraction.py and train_model.py. Once you have a trained
model saved via joblib, this will serve predictions.

Run with: python app.py
Test with: curl -X POST http://localhost:5001/predict -H "Content-Type: application/json" \
  -d '{"accel_window": [[0.1,0.2,9.8], [0.1,0.2,9.7], ...]}'
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import joblib
import os

from feature_extraction import extract_window_features

app = Flask(__name__)
CORS(app)  # allows the React frontend / Node backend to call this service

MODEL_PATH = os.path.join("models", "fall_detector_v2.joblib")
model = None


def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(f"Model loaded from {MODEL_PATH}")
    else:
        print(f"WARNING: no model found at {MODEL_PATH}. Train one first "
              f"with train_model.py, or /predict will fail.")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects JSON: {"accel_window": [[x,y,z], [x,y,z], ...]}
    (a window of ~1-2 seconds of accelerometer readings)

    Returns: {"fall_detected": bool, "confidence": float}
    """
    if model is None:
        return jsonify({"error": "Model not loaded. Train a model first."}), 503

    data = request.get_json()
    if not data or "accel_window" not in data:
        return jsonify({"error": "Missing 'accel_window' in request body"}), 400

    window_array = np.array(data["accel_window"])
    if window_array.shape[1] != 3:
        return jsonify({"error": "Each row must have [x, y, z]"}), 400

    window_df = pd.DataFrame(window_array, columns=["AccX", "AccY", "AccZ"])

    # This calls the function YOU implement in feature_extraction.py --
    # once that's done, this endpoint just works.
    features = extract_window_features(window_df)
    features_df = pd.DataFrame([features])

    prediction = model.predict(features_df)[0]
    confidence = model.predict_proba(features_df)[0].max()

    return jsonify({
        "fall_detected": bool(prediction),
        "confidence": float(confidence)
    })


if __name__ == "__main__":
    load_model()
    app.run(host="0.0.0.0", port=5001, debug=True)
