"""Train and compare Naive Bayes and Logistic Regression spam classifiers.

Usage (from the project root):
    python src/train.py
    python src/train.py --data dataset/spam.csv
"""
import argparse
import json
import os
import pickle
import sys

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import clean_text  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA = os.path.join(ROOT, "dataset", "spam.csv")
MODELS_DIR = os.path.join(ROOT, "models")


def load_dataset(path: str) -> pd.DataFrame:
    """Load the dataset into two columns: label (0=ham, 1=spam) and text.

    Supports the Kaggle SMS Spam Collection (columns v1, v2, latin-1 encoded)
    and any CSV with label/text style columns.
    """
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1")

    cols = {c.lower().strip(): c for c in df.columns}
    if "v1" in cols and "v2" in cols:
        df = df[[cols["v1"], cols["v2"]]]
    else:
        label_col = next((cols[c] for c in ("label", "category", "class", "spam") if c in cols), None)
        text_col = next((cols[c] for c in ("text", "message", "email", "body", "content") if c in cols), None)
        if label_col is None or text_col is None:
            raise ValueError(f"Could not find label/text columns in {list(df.columns)}")
        df = df[[label_col, text_col]]
    df.columns = ["label", "text"]

    df["label"] = df["label"].astype(str).str.strip().str.lower().map(
        {"ham": 0, "spam": 1, "0": 0, "1": 1}
    )
    return df


def evaluate(name, model, X_test, y_test):
    pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    }
    print(f"\n===== {name} =====")
    for k in ("accuracy", "precision", "recall", "f1"):
        print(f"{k:>10}: {metrics[k]:.4f}")
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred, target_names=["ham", "spam"]))
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DEFAULT_DATA)
    args = parser.parse_args()

    if not os.path.exists(args.data):
        sys.exit(
            f"Dataset not found at {args.data}\n"
            "Download the SMS Spam Collection from Kaggle and save it as dataset/spam.csv"
        )

    # 1. Load and clean data
    df = load_dataset(args.data)
    print(f"Loaded {len(df)} rows")
    df = df.dropna(subset=["label", "text"])
    before = len(df)
    df = df.drop_duplicates(subset="text")
    print(f"Removed {before - len(df)} duplicates -> {len(df)} rows")
    print("Class balance:\n", df["label"].map({0: "ham", 1: "spam"}).value_counts())

    # 2. Preprocess text
    df["clean"] = df["text"].apply(clean_text)

    # 3. Train-test split (stratified keeps the spam ratio equal in both sets)
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"], df["label"].astype(int), test_size=0.2, random_state=42, stratify=df["label"]
    )

    # 4. TF-IDF features - fit ONLY on training data to avoid leakage
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # 5. Train both models
    models = {
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.1),
        "Logistic Regression": LogisticRegression(max_iter=1000, C=10, class_weight="balanced"),
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train_vec, y_train)
        results[name] = evaluate(name, model, X_test_vec, y_test)

    # 6. Pick the best model by F1 (balances precision and recall on the minority spam class)
    best_name = max(results, key=lambda n: results[n]["f1"])
    print(f"\nBest model: {best_name} (F1 = {results[best_name]['f1']:.4f})")

    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(os.path.join(MODELS_DIR, "spam_model.pkl"), "wb") as f:
        pickle.dump(models[best_name], f)
    with open(os.path.join(MODELS_DIR, "vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)
    with open(os.path.join(MODELS_DIR, "metrics.json"), "w") as f:
        json.dump({"best_model": best_name, "results": results}, f, indent=2)
    print(f"Saved model, vectorizer and metrics to {MODELS_DIR}")


if __name__ == "__main__":
    main()
