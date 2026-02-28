from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import os

app = Flask(__name__)
CORS(app)

print("Loading model...")

# Load model from same directory as app.py
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

print("Model loaded successfully.")


# -------------------------
# Health Check Route
# -------------------------
@app.route("/")
def home():
    return jsonify({
        "status": "success",
        "message": "Fake News Detection API Running"
    })


# -------------------------
# Single Prediction
# -------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data or "text" not in data:
            return jsonify({"error": "No text provided"}), 400

        text = data["text"]

        if not isinstance(text, str) or not text.strip():
            return jsonify({"error": "Invalid or empty text"}), 400

        # Use full sklearn pipeline directly
        prediction = model.predict([text])[0]
        probabilities = model.predict_proba([text])[0]

        confidence = round(float(np.max(probabilities)) * 100, 2)

        result = "Real News" if prediction == 1 else "Fake News"

        return jsonify({
            "prediction": result,
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({
            "error": "Prediction failed",
            "details": str(e)
        }), 500


# -------------------------
# Batch Prediction
# -------------------------
@app.route("/batch_predict", methods=["POST"])
def batch_predict():
    try:
        data = request.get_json()

        if not data or "texts" not in data:
            return jsonify({"error": "Provide list of texts"}), 400

        texts = data["texts"]

        if not isinstance(texts, list) or len(texts) == 0:
            return jsonify({"error": "Invalid texts list"}), 400

        predictions = model.predict(texts)
        probabilities = model.predict_proba(texts)

        results = []

        for i in range(len(texts)):
            label = "Real News" if predictions[i] == 1 else "Fake News"
            confidence = round(float(np.max(probabilities[i])) * 100, 2)

            results.append({
                "text": texts[i],
                "prediction": label,
                "confidence": confidence
            })

        return jsonify(results)

    except Exception as e:
        return jsonify({
            "error": "Batch prediction failed",
            "details": str(e)
        }), 500


# -------------------------
# Run Locally
# -------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)