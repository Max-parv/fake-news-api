from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)

print("Loading model...")

with open("../model.pkl", "rb") as f:
    model = pickle.load(f)

print("Model loaded successfully.")

@app.route("/")
def home():
    return jsonify({"message": "Fake News Detection API Running"})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    text = data["text"]

    if not text.strip():
        return jsonify({"error": "Empty text"}), 400

    # Use FULL pipeline directly
    prediction = model.predict([text])[0]
    probabilities = model.predict_proba([text])[0]

    confidence = round(float(np.max(probabilities)) * 100, 2)

    result = "Real News" if prediction == 1 else "Fake News"

    return jsonify({
        "prediction": result,
        "confidence": confidence
    })


if __name__ == "__main__":
    app.run(debug=True)