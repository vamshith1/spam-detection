"""Load the saved model and classify new messages.

Usage:
    python src/predict.py "Congratulations! You won a free iPhone"
"""
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import clean_text  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")


def load_artifacts():
    with open(os.path.join(MODELS_DIR, "spam_model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "vectorizer.pkl"), "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer


def predict(message: str, model=None, vectorizer=None):
    """Return (label, spam_probability) for a single message."""
    if model is None or vectorizer is None:
        model, vectorizer = load_artifacts()
    features = vectorizer.transform([clean_text(message)])
    spam_prob = float(model.predict_proba(features)[0][1])
    label = "Spam" if spam_prob >= 0.5 else "Ham"
    return label, spam_prob


if __name__ == "__main__":
    msg = " ".join(sys.argv[1:]) or "Congratulations! You have won a free iPhone. Click here to claim now!"
    label, prob = predict(msg)
    print(f"Message: {msg}\nPrediction: {label} (spam probability {prob:.2%})")
