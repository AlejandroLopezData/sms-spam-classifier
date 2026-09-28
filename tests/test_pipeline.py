import numpy as np
import pytest

from spam_detection.evaluate import select_threshold
from spam_detection.models import MODEL_NAMES, build_pipeline

SPAM = [
    "WIN a FREE prize now call 08001234567 £1000",
    "URGENT! claim your free cash at www.win-cash.com",
    "Congratulations you have won a £500 voucher, text WIN to 80086",
    "FREE entry to our weekly competition, reply now!!!",
    "You have been selected for a cash prize, call now 09061234567",
    "Claim your free ringtone today, click http://ring.tones",
    "URGENT your mobile number has won £2000 bonus caller prize",
    "Free msg: get 5 free tickets, text STOP to opt out",
]
HAM = [
    "ok see you at the station in ten minutes",
    "can you pick up some milk on your way home",
    "I will call you later tonight, dinner was great",
    "happy birthday, hope you have a lovely day",
    "are we still meeting for lunch tomorrow",
    "thanks for helping me with the homework yesterday",
    "running late, start without me please",
    "did you see the game last night, it was amazing",
]
TFIDF = {"min_df": 1, "max_df": 1.0, "max_features": 200, "ngram_range": [1, 1], "stop_words": None}


@pytest.mark.parametrize("manual", [True, False])
@pytest.mark.parametrize("model", MODEL_NAMES)
def test_pipeline_fit_predict(model, manual):
    X = SPAM + HAM
    y = [1] * len(SPAM) + [0] * len(HAM)
    pipe = build_pipeline(model, TFIDF, use_manual_features=manual, random_state=42)
    pipe.fit(X, y)
    proba = pipe.predict_proba(["free prize call now", "see you at lunch"])
    assert proba.shape == (2, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_select_threshold_respects_min_precision():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.6, 0.5, 0.7, 0.8, 0.9])
    thr = select_threshold(y_true, y_prob, min_precision=1.0)
    assert thr == pytest.approx(0.7)  # 0.5 would include the 0.6 ham message