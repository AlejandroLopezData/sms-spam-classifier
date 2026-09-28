"""Text cleaning and hand-crafted features.

Everything here is either stateless or fitted inside an sklearn Pipeline, so no information
from the test set leaks into training.
"""
from __future__ import annotations

import re

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w.-]+@[\w.-]+\.\w+\b")
PHONE_RE = re.compile(r"\b\d{7,}\b")
CURRENCY_RE = re.compile(r"[£$€]")
SYMBOL_RE = re.compile(r"[$£€%&@#*]")
NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")
WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Normalise a message for TF-IDF.

    URLs, emails, phone numbers and currency symbols are replaced by placeholder tokens
    instead of being deleted, so the vectorizer can still use them as signal.
    """
    text = text.lower()
    text = URL_RE.sub(" urltoken ", text)
    text = EMAIL_RE.sub(" emailtoken ", text)
    text = PHONE_RE.sub(" phonetoken ", text)
    text = CURRENCY_RE.sub(" currencytoken ", text)
    text = NON_ALNUM_RE.sub(" ", text)
    return WHITESPACE_RE.sub(" ", text).strip()


class TextStats(BaseEstimator, TransformerMixin):
    """Hand-crafted numeric features computed from the RAW message text."""

    FEATURE_NAMES = (
        "msg_length",
        "has_url",
        "has_email",
        "has_phone",
        "num_digits",
        "num_uppercase",
        "num_exclamations",
        "num_symbols",
    )

    def __init__(self, log_transform: bool = True):
        self.log_transform = log_transform

    def fit(self, X, y=None):
        return self

    @staticmethod
    def _stats(text: str) -> list[float]:
        return [
            len(text),
            int(bool(URL_RE.search(text))),
            int(bool(EMAIL_RE.search(text))),
            int(bool(PHONE_RE.search(text))),
            sum(c.isdigit() for c in text),
            sum(c.isupper() for c in text),
            text.count("!"),
            len(SYMBOL_RE.findall(text)),
        ]

    def transform(self, X):
        features = np.array([self._stats(text) for text in X], dtype=float)
        # log1p tames heavy-tailed counts (e.g. very long messages)
        return np.log1p(features) if self.log_transform else features

    def get_feature_names_out(self, input_features=None):
        return np.array(self.FEATURE_NAMES, dtype=object)