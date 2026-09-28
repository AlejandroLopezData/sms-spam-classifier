"""Classify new messages with the trained model.

Usage:
    spam-predict "Congratulations! You won a FREE prize, call 08001234567 now"
    spam-predict --file messages.txt        # one message per line
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from spam_detection.config import load_config, resolve_path


def load_bundle(model_path: str | Path | None = None) -> dict:
    """Load the saved model bundle (pipeline + decision threshold)."""
    if model_path is None:
        model_path = resolve_path(load_config()["paths"]["model"])
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"{model_path} not found. Run `spam-train` first.")
    return joblib.load(model_path)


def predict_messages(messages: list[str], bundle: dict) -> list[dict]:
    """Return label and spam probability for each message."""
    probs = bundle["pipeline"].predict_proba(list(messages))[:, 1]
    return [
        {
            "message": msg,
            "spam_probability": float(p),
            "label": "spam" if p >= bundle["threshold"] else "ham",
        }
        for msg, p in zip(messages, probs)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify SMS messages as spam or ham.")
    parser.add_argument("messages", nargs="*", help="Messages to classify")
    parser.add_argument("--file", help="Text file with one message per line")
    parser.add_argument("--model", default=None, help="Path to a saved model (.joblib)")
    args = parser.parse_args()

    messages = list(args.messages)
    if args.file:
        messages += [
            line.strip()
            for line in Path(args.file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if not messages:
        parser.error("Provide at least one message or --file.")

    bundle = load_bundle(args.model)
    for r in predict_messages(messages, bundle):
        print(f"[{r['label'].upper():<4}] p(spam) = {r['spam_probability']:.3f} | {r['message']}")


if __name__ == "__main__":
    main()