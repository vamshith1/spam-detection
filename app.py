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

TYPES = {"Email": "email", "SMS / text message": "sms"}

EXAMPLES = {
    "email": {
        "Phishing email": (
            "Subject: Your account will be suspended\n\nDear valued customer, we detected unusual "
            "activity on your account. To avoid suspension, verify your billing information within "
            "24 hours by clicking the secure link below. Limited time offer: get 50% off your next "
            "order when you confirm today! http://secure-verify-account.biz/login"
        ),
        "Work email": (
            "Subject: Q3 budget review\n\nHi Sara, I've attached the updated Q3 budget spreadsheet. "
            "Could you review the travel and contractor lines before Thursday's meeting? Let me know "
            "if anything looks off and I'll update it. Thanks, Mark"
        ),
    },
    "sms": {
        "Prize scam": "Congratulations! You have won a free iPhone. Click here to claim your prize now!",
        "Promo with number": "URGENT! Your mobile number has been awarded £2000. Call 09061701461 to claim.",
        "Normal message": "Hey, are we still meeting for lunch tomorrow at 1pm?",
    },
}


def model_dir(kind):
    return os.path.join(ROOT, "models", kind)


@st.cache_resource
def get_model(kind):
    return load_artifacts(kind)


st.title("📧 Spam Detector")
st.caption("Machine Learning Powered Message Classification")

available = {label: k for label, k in TYPES.items() if os.path.exists(os.path.join(model_dir(k), "spam_model.pkl"))}
if not available:
    st.error("No trained model found. Run `python src/train.py` first.")
    st.stop()

tab_predict, tab_metrics = st.tabs(["Check a message", "Model performance"])

with tab_predict:
    type_label = st.radio("Message type", list(available), horizontal=True)
    kind = available[type_label]
    model, vectorizer = get_model(kind)

    examples = {"— pick an example —": "", **EXAMPLES[kind]}
    choice = st.selectbox("Try an example", list(examples), key=f"example_{kind}")
    message = st.text_area(
        "Enter your email or message",
        value=examples[choice],
        height=200 if kind == "email" else 120,
        placeholder="Paste the subject and body of the email here..."
        if kind == "email"
        else "Paste your text message here...",
        key=f"text_{kind}_{choice}",
    )

    if st.button("Predict Message", type="primary"):
        if not message.strip():
            st.warning("Please enter a message first.")
        else:
            label, prob = predict(message, kind, model, vectorizer)
            if label == "Spam":
                st.error(f"🚫 **Spam detected** — spam probability {prob:.1%}")
            else:
                st.success(f"✅ **Ham / Not Spam** — spam probability {prob:.1%}")
            st.progress(prob)
            st.caption(f"Checked with the {type_label.lower()} model.")

with tab_metrics:
    keys = ("accuracy", "precision", "recall", "f1")
    for type_label, kind in available.items():
        path = os.path.join(model_dir(kind), "metrics.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)
        st.subheader(f"{type_label} model")
        st.caption(
            f"Dataset: {data['dataset']} — {data['rows']:,} messages "
            f"({data['train_size']:,} train / {data['test_size']:,} test). "
            f"Model in use: **{data['best_model']}**"
        )
        table = pd.DataFrame({name: {k: r[k] for k in keys} for name, r in data["results"].items()}).T
        st.dataframe(table.style.format("{:.4f}"))
        cols = st.columns(len(data["results"]))
        for col, (name, r) in zip(cols, data["results"].items()):
            with col:
                st.write(f"**{name}**")
                st.table(
                    pd.DataFrame(
                        r["confusion_matrix"],
                        index=["Actual Ham", "Actual Spam"],
                        columns=["Pred Ham", "Pred Spam"],
                    )
                )
