"""Metrics, threshold selection and plots."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: figures are saved to disk
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def _save(fig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    """Classification metrics for the spam class at a given decision threshold."""
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def select_threshold(y_true, y_prob, min_precision: float = 0.98) -> float:
    """Highest-recall threshold whose precision is at least `min_precision`.

    Falls back to 0.5 if no threshold reaches the required precision.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    valid = precision[:-1] >= min_precision
    if not valid.any():
        return 0.5
    idx = int(np.argmax(np.where(valid, recall[:-1], -1.0)))
    return float(thresholds[idx])


def plot_roc_curves(y_true, probs: dict, path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, y_prob in probs.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc_score(y_true, y_prob):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curves (test set)")
    ax.legend(loc="lower right")
    _save(fig, path)


def plot_pr_curves(y_true, probs: dict, path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, y_prob in probs.items():
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ax.plot(recall, precision, label=f"{name} (AP = {average_precision_score(y_true, y_prob):.3f})")
    ax.set(
        xlabel="Recall",
        ylabel="Precision",
        title="Precision-Recall curves (test set)",
        ylim=(0.5, 1.02),
    )
    ax.legend(loc="lower left")
    _save(fig, path)


def plot_confusion(y_true, y_pred, title: str, path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["ham", "spam"], cmap="Blues", colorbar=False, ax=ax
    )
    ax.set_title(title)
    _save(fig, path)


def plot_top_features(pipeline, path, top_n: int = 15) -> None:
    """Most spam-indicative and ham-indicative features of a fitted logistic regression."""
    names = pipeline.named_steps["features"].get_feature_names_out()
    names = np.array([n.replace("tfidf__", "") for n in names])
    coefs = pipeline.named_steps["clf"].coef_.ravel()

    order = np.argsort(coefs)
    idx = np.concatenate([order[:top_n], order[-top_n:]])
    colors = ["tab:blue" if coefs[i] < 0 else "tab:red" for i in idx]

    fig, ax = plt.subplots(figsize=(8, 9))
    ax.barh(names[idx], coefs[idx], color=colors)
    ax.set(xlabel="Coefficient (red -> spam, blue -> ham)", title=f"Top {top_n} features per class")
    _save(fig, path)


def save_error_analysis(X_test, y_test, y_prob, threshold: float, path) -> pd.DataFrame:
    """Save misclassified test messages (false positives / false negatives) to CSV."""
    df = pd.DataFrame(
        {
            "text": np.asarray(X_test),
            "label": np.asarray(y_test),
            "spam_probability": np.asarray(y_prob),
        }
    )
    df["prediction"] = (df["spam_probability"] >= threshold).astype(int)
    errors = df[df["label"] != df["prediction"]].copy()
    errors["error_type"] = np.where(errors["label"] == 1, "false_negative", "false_positive")
    errors = errors.sort_values(["error_type", "spam_probability"])

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    errors.to_csv(path, index=False)
    return errors