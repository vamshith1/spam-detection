"""Streamlit interface for the spam detector.

Run:  streamlit run app.py
"""
import json
import os
import sys

import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
from predict import load_artifacts, predict  # noqa: E402

st.set_page_config(page_title="Spam Detector", page_icon="📧", layout="centered")


@st.cache_resource
def get_model():
    return load_artifacts()


st.title("📧 Spam Detector")
st.caption("Machine Learning Powered Message Classification")

if not os.path.exists(os.path.join(ROOT, "models", "spam_model.pkl")):
    st.error("No trained model found. Run `python src/train.py` first.")
    st.stop()

model, vectorizer = get_model()

tab_predict, tab_metrics = st.tabs(["Check a message", "Model performance"])

with tab_predict:
    examples = {
        "— pick an example —": "",
        "Prize scam": "Congratulations! You have won a free iPhone. Click here to claim your prize now!",
        "Normal message": "Hey, are we still meeting for lunch tomorrow at 1pm?",
        "Promo with number": "URGENT! Your mobile number has been awarded £2000. Call 09061701461 to claim.",
    }
    choice = st.selectbox("Try an example", list(examples))
    message = st.text_area(
        "Enter your email or message",
        value=examples[choice],
        height=160,
        placeholder="Paste your message here... e.g. Congratulations! You won a free gift...",
    )

    if st.button("Predict Message", type="primary"):
        if not message.strip():
            st.warning("Please enter a message first.")
        else:
            label, prob = predict(message, model, vectorizer)
            if label == "Spam":
                st.error(f"🚫 **Spam detected** — spam probability {prob:.1%}")
            else:
                st.success(f"✅ **Ham / Not Spam** — spam probability {prob:.1%}")
            st.progress(prob)

with tab_metrics:
    metrics_path = os.path.join(ROOT, "models", "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            data = json.load(f)
        st.write(f"**Model in use:** {data['best_model']}")
        table = pd.DataFrame(
            {name: {k: v for k, v in r.items() if k != "confusion_matrix"} for name, r in data["results"].items()}
        ).T
        st.dataframe(table.style.format("{:.4f}"))
        for name, r in data["results"].items():
            st.write(f"**{name} — confusion matrix**")
            cm = pd.DataFrame(
                r["confusion_matrix"], index=["Actual Ham", "Actual Spam"], columns=["Pred Ham", "Pred Spam"]
            )
            st.table(cm)
    else:
        st.info("Metrics appear here after running `python src/train.py`.")
