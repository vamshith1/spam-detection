# Email Spam Detection Using Machine Learning

A machine learning app that classifies emails / text messages as **Spam** or **Ham** (not spam) using NLP preprocessing, TF-IDF features, and Naive Bayes / Logistic Regression, with a Streamlit web interface.

## Project structure

```
├── dataset/spam.csv        # labelled messages (download separately)
├── models/                 # created by training
│   ├── spam_model.pkl      # best model
│   ├── vectorizer.pkl      # fitted TF-IDF vectorizer
│   └── metrics.json        # evaluation results for both models
├── src/
│   ├── preprocess.py       # text cleaning
│   ├── train.py            # train + evaluate + save
│   └── predict.py          # load model and predict
├── app.py                  # Streamlit UI
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

Download the **SMS Spam Collection Dataset** from Kaggle and save it as `dataset/spam.csv`.
Any CSV with a `label` (spam/ham) column and a `text`/`message` column also works.

Optional: `python -m nltk.downloader stopwords` (otherwise scikit-learn's stop-word list is used).

## Run

```bash
python src/train.py          # trains both models, prints metrics, saves the best one
python src/predict.py "You won a free prize, call now!"
streamlit run app.py
```

## How it works

1. **Load & clean data** – drop missing rows and duplicates, map labels to 0/1.
2. **Preprocess** – lowercase, replace URLs/emails/numbers with tokens, remove punctuation and stop words, stem words.
3. **Split** – 80/20 stratified train-test split.
4. **TF-IDF** – unigrams + bigrams, fitted on training data only (no leakage).
5. **Train** – Multinomial Naive Bayes and Logistic Regression.
6. **Evaluate** – accuracy, precision, recall, F1, confusion matrix; best model chosen by F1.
7. **Predict** – Streamlit app shows Spam / Ham with spam probability.

## Results

Trained on the SMS Spam Collection (5,574 messages → 5,171 after removing duplicates; 4,518 ham / 653 spam). Test set: 1,035 messages (20%).

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Multinomial Naive Bayes | 98.45% | 97.52% | 90.08% | 0.9365 |
| **Logistic Regression** (selected) | **98.84%** | 96.12% | **94.66%** | **0.9538** |

Confusion matrices (rows = actual, columns = predicted):

| | NB: Pred Ham | NB: Pred Spam | LR: Pred Ham | LR: Pred Spam |
|---|---|---|---|---|
| Actual Ham | 901 | 3 | 899 | 5 |
| Actual Spam | 13 | 118 | 7 | 124 |

Logistic Regression catches more spam (7 missed vs 13) at the cost of 2 extra false alarms, giving it the higher F1.

## Dataset & credits

`dataset/spam.csv` is the [SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) by Tiago A. Almeida and José María Gómez Hidalgo, from the UCI Machine Learning Repository, licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). It was converted from tab-separated to CSV with `label,text` columns.

The trained model (`models/`) is committed so the app runs straight after cloning; run `python src/train.py` to retrain.

The SMS Spam Collection contains SMS messages, not emails. For true email spam detection, train on an email dataset such as the Enron spam corpus — the code works the same.
