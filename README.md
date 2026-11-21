# Email Spam Detection Using Machine Learning

A machine learning app that classifies **emails** and **SMS / text messages** as **Spam** or **Ham** (not spam) using NLP preprocessing, TF-IDF features, and Naive Bayes / Logistic Regression, with a Streamlit web interface.

**Live demo:** https://vn-spam-detector.streamlit.app

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://vn-spam-detector.streamlit.app)

## Highlights

- Trained on **30,000+ real emails** (Enron-Spam corpus) and 5,000+ SMS messages
- **98.7% accuracy** on unseen emails, **98.8%** on unseen SMS
- Separate model per message type, since emails and texts use very different language
- Streamlit app with an Email / SMS switch, spam probability and model performance dashboard

## Results

Each model is tested on a held-out 20% of its dataset that it never saw during training.

### Email model: Enron-Spam (30,138 emails after removing duplicates; 15,590 ham / 14,548 spam)

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| **Multinomial Naive Bayes** (selected) | **98.74%** | **98.63%** | 98.76% | **0.9870** |
| Logistic Regression | 98.71% | 98.23% | **99.11%** | 0.9867 |

| Test set: 6,028 emails | NB: Pred Ham | NB: Pred Spam | LR: Pred Ham | LR: Pred Spam |
|---|---|---|---|---|
| Actual Ham | 3,078 | 40 | 3,066 | 52 |
| Actual Spam | 36 | 2,874 | 26 | 2,884 |

The two are almost tied. Naive Bayes wins on F1 by making fewer false alarms (40 vs 52 real emails wrongly flagged), which matters more to users than catching a few extra spam emails.

### SMS model: SMS Spam Collection (5,171 messages after removing duplicates; 4,518 ham / 653 spam)

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Multinomial Naive Bayes | 98.45% | 97.52% | 90.08% | 0.9365 |
| **Logistic Regression** (selected) | **98.84%** | 96.12% | **94.66%** | **0.9538** |

| Test set: 1,035 messages | NB: Pred Ham | NB: Pred Spam | LR: Pred Ham | LR: Pred Spam |
|---|---|---|---|---|
| Actual Ham | 901 | 3 | 899 | 5 |
| Actual Spam | 13 | 118 | 7 | 124 |

Only 13% of SMS messages are spam, so accuracy alone is misleading here. Logistic Regression (with balanced class weights) misses about half as much spam (7 vs 13).

### Why two models instead of one?

A single model trained on both datasets reached 98.3% on emails but only **0.83 F1 on SMS**: about 1 in 22 normal texts got flagged as spam. The 30,000 long emails dominated the vocabulary, and short, slangy texts looked unusual by comparison. Even giving SMS 5× weight only reached 0.90 F1. Separate models keep each one at its best (0.987 email, 0.954 SMS).

## Project structure

```
├── dataset/
│   ├── emails.csv.gz       # Enron-Spam emails (label, text), gzip-compressed
│   └── spam.csv            # SMS Spam Collection (label, text)
├── models/
│   ├── email/              # spam_model.pkl, vectorizer.pkl, metrics.json
│   └── sms/                # spam_model.pkl, vectorizer.pkl, metrics.json
├── src/
│   ├── preprocess.py       # text cleaning
│   ├── train.py            # train + evaluate + save (one model per type)
│   └── predict.py          # load a model and predict
├── app.py                  # Streamlit UI
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

The datasets and trained models are included, so the app runs straight after cloning.

## Run

```bash
python src/train.py                  # retrain both models (or: --type email / --type sms)
python src/predict.py "Subject: Verify your account now at http://secure-login.biz"
python src/predict.py --type sms "WINNER! Call 09061701461 to claim your prize"
streamlit run app.py
```

Use it from Python:

```python
import sys; sys.path.insert(0, "src")
from predict import predict

label, prob = predict("Subject: Team lunch on Friday...", kind="email")   # ('Ham', 0.001)
```

## How it works

1. **Load & clean data**: drop missing rows and duplicates, map labels to 0/1.
2. **Preprocess**: strip HTML tags and the `Subject:` header, lowercase, replace URLs / email addresses / numbers with tokens (`urltoken`, `numtoken`), remove punctuation and stop words, stem words.
3. **Split**: 80/20 stratified train-test split.
4. **TF-IDF**: unigrams + bigrams, fitted on training data only (no leakage). 30,000 features for email, 5,000 for SMS.
5. **Train**: Multinomial Naive Bayes and Logistic Regression for each message type.
6. **Evaluate**: accuracy, precision, recall, F1, confusion matrix. The best model per type is chosen by F1.
7. **Predict**: the Streamlit app shows Spam / Ham with the spam probability.

## Limitations

- The app only reads the **text** you paste. It doesn't check the sender, headers, real link targets or attachments.
- The Enron emails are from 2000–2005. Modern spam styles (crypto scams, AI-written phishing) may be caught less reliably.
- Treat the result as a second opinion. Don't click links just because a message was classified as Ham.

## Dataset & credits

- **Enron-Spam** (`dataset/emails.csv.gz`): V. Metsis, I. Androutsopoulos and G. Paliouras, *"Spam Filtering with Naive Bayes – Which Naive Bayes?"*, CEAS 2006. Preprocessed version from the [authors' site](https://www2.aueb.gr/users/ion/data/enron-spam/). The six `enron1`–`enron6` folders were merged into one `label,text` CSV.
- **SMS Spam Collection** (`dataset/spam.csv`): Tiago A. Almeida and José María Gómez Hidalgo, [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/228/sms+spam+collection), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Converted from tab-separated to CSV.
