"""Load a saved model and classify new messages.

Usage:
    python src/predict.py "Your account is suspended, verify now"            # email model
    python src/predict.py --type sms "WINNER! Call 09061701461 to claim"     # SMS model
"""
import argparse
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import clean_text  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")
TYPES = ("email", "sms")


def load_artifacts(kind: str = "email"):
    """Return (model, vectorizer) for the given message type."""
    if kind not in TYPES:
        raise ValueError(f"kind must be one of {TYPES}")
    folder = os.path.join(MODELS_DIR, kind)
    with open(os.path.join(folder, "spam_model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(folder, "vectorizer.pkl"), "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer


def predict(message: str, kind: str = "email", model=None, vectorizer=None):
    """Return (label, spam_probability) for a single message."""
    if model is None or vectorizer is None:
        model, vectorizer = load_artifacts(kind)
    features = vectorizer.transform([clean_text(message)])
    spam_prob = float(model.predict_proba(features)[0][1])
    label = "Spam" if spam_prob >= 0.5 else "Ham"
    return label, spam_prob


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", choices=TYPES, default="email")
    parser.add_argument("message", nargs="*")
    args = parser.parse_args()
    msg = " ".join(args.message) or "Congratulations! You have won a free iPhone. Click here to claim now!"
    label, prob = predict(msg, args.type)
    print(f"[{args.type}] Message: {msg}\nPrediction: {label} (spam probability {prob:.2%})")
