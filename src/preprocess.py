"""Text cleaning for spam detection.

Steps: strip HTML and "Subject:" -> lowercase -> replace URLs/emails/numbers with tokens -> strip punctuation
-> tokenize -> drop stop words -> stem each word.
"""
import re
import string

from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

try:
    # NLTK's list is used if it has been downloaded (python -m nltk.downloader stopwords)
    from nltk.corpus import stopwords

    STOP_WORDS = set(stopwords.words("english"))
except LookupError:
    # Fallback that needs no download
    STOP_WORDS = set(ENGLISH_STOP_WORDS)

_stemmer = PorterStemmer()
_HTML_RE = re.compile(r"<[^>]+>")
_SUBJECT_RE = re.compile(r"^\s*subject\s*:", re.IGNORECASE)
_URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
_EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
_NUM_RE = re.compile(r"\d+")
_PUNCT_TABLE = str.maketrans(string.punctuation, " " * len(string.punctuation))


def clean_text(text: str) -> str:
    """Return a cleaned, stemmed version of a raw message."""
    if not isinstance(text, str):
        return ""
    # Emails: drop HTML tags and the leading "Subject:" header word, so the model
    # learns from the words themselves rather than from email formatting.
    text = _HTML_RE.sub(" ", text)
    text = _SUBJECT_RE.sub(" ", text)
    text = text.lower()
    # Keep the *fact* that a link/email/number was present - a strong spam signal
    text = _URL_RE.sub(" urltoken ", text)
    text = _EMAIL_RE.sub(" emailtoken ", text)
    text = _NUM_RE.sub(" numtoken ", text)
    text = text.translate(_PUNCT_TABLE)
    tokens = text.split()
    tokens = [_stemmer.stem(t) for t in tokens if t not in STOP_WORDS and len(t) > 1]
    return " ".join(tokens)


if __name__ == "__main__":
    sample = "Congratulations! You WON a FREE iPhone. Click www.prize.com now or call 09061701461"
    print("Raw:    ", sample)
    print("Cleaned:", clean_text(sample))
