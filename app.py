import pickle
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

print("Loading models...")

# Load vectorizer
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Load multiple models
models = {
    "passive_aggressive": pickle.load(open("model.pkl", "rb")),
    # Optional additional models:
    # "logistic": pickle.load(open("log_model.pkl", "rb")),
}

print("Models loaded successfully.")


# -------------------------
# Single Prediction
# -------------------------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")
    model_name = data.get("model", "passive_aggressive")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    if model_name not in models:
        return jsonify({"error": "Model not found"}), 400

    model = models[model_name]
    vectorized = vectorizer.transform([text])

    prediction = model.predict(vectorized)[0]
    proba = model.predict_proba(vectorized)[0]
    confidence = round(float(np.max(proba)) * 100, 2)

    result = "Real News" if prediction == 1 else "Fake News"

    return jsonify({
        "model_used": model_name,
        "prediction": result,
        "confidence": confidence
    })


# -------------------------
# Batch Prediction
# -------------------------
@app.route("/batch_predict", methods=["POST"])
def batch_predict():
    data = request.get_json()
    texts = data.get("texts", [])

    if not texts or not isinstance(texts, list):
        return jsonify({"error": "Provide list of texts"}), 400

    vectorized = vectorizer.transform(texts)
    model = models["passive_aggressive"]

    predictions = model.predict(vectorized)
    probabilities = model.predict_proba(vectorized)

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


# -------------------------
# Explanation Endpoint
# -------------------------
@app.route("/explain", methods=["POST"])
def explain():
    data = request.get_json()
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    vectorized = vectorizer.transform([text])
    model = models["passive_aggressive"]

    feature_names = vectorizer.get_feature_names_out()
    coefficients = model.base_estimator.coef_[0]

    # Get top influential words
    sorted_indices = np.argsort(coefficients)
    top_fake = [feature_names[i] for i in sorted_indices[:10]]
    top_real = [feature_names[i] for i in sorted_indices[-10:]]

    return jsonify({
        "top_fake_indicators": top_fake,
        "top_real_indicators": top_real
    })


@app.route("/")
def home():
    return jsonify({"message": "Advanced Fake News API Running"})


if __name__ == "__main__":
    app.run(debug=True)