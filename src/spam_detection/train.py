"""Train, tune and compare all configured experiments.

Usage:
    spam-train                       # or: python -m spam_detection.train
    spam-train --config configs/config.yaml
"""
from __future__ import annotations

import argparse
import json
import logging

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict

from spam_detection.config import load_config, resolve_path
from spam_detection.data import load_dataset, split_data
from spam_detection.evaluate import (
    compute_metrics,
    plot_confusion,
    plot_pr_curves,
    plot_roc_curves,
    plot_top_features,
    save_error_analysis,
    select_threshold,
)
from spam_detection.models import build_pipeline

logger = logging.getLogger("spam_detection.train")


def _prepare_grid(raw_grid: dict | None) -> dict:
    """YAML lists -> tuples where needed (e.g. ngram_range must be a tuple)."""
    grid = {}
    for key, values in (raw_grid or {}).items():
        grid[key] = [tuple(v) if isinstance(v, list) else v for v in values]
    return grid


def run_experiment(exp: dict, cfg: dict, splits, cv, seed: int) -> dict:
    X_train, X_test, y_train, y_test = splits
    name = exp["name"]
    logger.info("[%s] grid search...", name)

    pipeline = build_pipeline(
        model_name=exp["model"],
        tfidf_params=cfg["tfidf"],
        use_manual_features=exp["manual_features"],
        random_state=seed,
    )
    grid = _prepare_grid(cfg.get("param_grids", {}).get(exp["model"]))
    search = GridSearchCV(
        pipeline, grid, scoring=cfg["cv"]["scoring"], cv=cv, n_jobs=-1, refit=True
    )
    search.fit(X_train, y_train)
    best = search.best_estimator_
    logger.info("[%s] CV %s = %.4f | %s", name, cfg["cv"]["scoring"], search.best_score_, search.best_params_)

    # Threshold chosen on out-of-fold train predictions -> the test set is never used for tuning
    oof_prob = cross_val_predict(
        best, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1
    )[:, 1]
    threshold = select_threshold(y_train, oof_prob, cfg["threshold"]["min_precision"])

    y_prob = best.predict_proba(X_test)[:, 1]
    return {
        "exp": exp,
        "estimator": best,
        "cv_score": search.best_score_,
        "best_params": search.best_params_,
        "threshold": threshold,
        "y_prob": y_prob,
        "metrics_default": compute_metrics(y_test, y_prob, 0.5),
        "metrics_tuned": compute_metrics(y_test, y_prob, threshold),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and compare spam classifiers.")
    parser.add_argument("--config", default=None, help="Path to a YAML config file")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")

    cfg = load_config(args.config)
    seed = cfg["seed"]
    reports_dir = resolve_path(cfg["paths"]["reports"])
    figures_dir = resolve_path(cfg["paths"]["figures"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    # --- Data ---
    X, y = load_dataset(resolve_path(cfg["paths"]["raw_data"]))
    logger.info("Class distribution:\n%s", y.map({0: "ham", 1: "spam"}).value_counts().to_string())
    splits = split_data(X, y, cfg["data"]["test_size"], seed)
    X_train, X_test, y_train, y_test = splits
    logger.info("Train: %d | Test: %d", len(X_train), len(X_test))

    cv = StratifiedKFold(n_splits=cfg["cv"]["folds"], shuffle=True, random_state=seed)

    # --- Experiments ---
    results = {}
    for exp in cfg["experiments"]:
        results[exp["name"]] = run_experiment(exp, cfg, splits, cv, seed)

    # --- Summary table ---
    rows = []
    for name, res in results.items():
        d, t = res["metrics_default"], res["metrics_tuned"]
        rows.append(
            {
                "experiment": name,
                "model": res["exp"]["model"],
                "manual_features": res["exp"]["manual_features"],
                f"cv_{cfg['cv']['scoring']}": res["cv_score"],
                "test_roc_auc": d["roc_auc"],
                "test_pr_auc": d["pr_auc"],
                "precision@0.5": d["precision"],
                "recall@0.5": d["recall"],
                "f1@0.5": d["f1"],
                "tuned_threshold": t["threshold"],
                "precision_tuned": t["precision"],
                "recall_tuned": t["recall"],
                "f1_tuned": t["f1"],
                "fp_tuned": t["fp"],
                "fn_tuned": t["fn"],
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(reports_dir / "results.csv", index=False)
    with open(reports_dir / "best_params.json", "w", encoding="utf-8") as f:
        json.dump({n: r["best_params"] for n, r in results.items()}, f, indent=2, default=str)
    logger.info("\n%s", summary.round(4).to_string(index=False))

    # --- Best model: selected by CV score on train (NOT by test performance) ---
    best_name = max(results, key=lambda n: results[n]["cv_score"])
    best = results[best_name]
    logger.info("Best experiment by CV: %s", best_name)

    # --- Figures & error analysis ---
    probs = {n: r["y_prob"] for n, r in results.items()}
    plot_roc_curves(y_test, probs, figures_dir / "roc_curves.png")
    plot_pr_curves(y_test, probs, figures_dir / "pr_curves.png")

    y_pred_best = (best["y_prob"] >= best["threshold"]).astype(int)
    plot_confusion(
        y_test,
        y_pred_best,
        f"{best_name} (threshold = {best['threshold']:.2f})",
        figures_dir / "confusion_matrix.png",
    )
    save_error_analysis(
        X_test, y_test, best["y_prob"], best["threshold"], reports_dir / "errors.csv"
    )

    # Interpretability: coefficients of the best logistic regression experiment
    logreg_names = [
        n for n, r in results.items() if isinstance(r["estimator"].named_steps["clf"], LogisticRegression)
    ]
    if logreg_names:
        best_logreg = max(logreg_names, key=lambda n: results[n]["cv_score"])
        plot_top_features(results[best_logreg]["estimator"], figures_dir / "top_features.png")

    # --- Save the best model ---
    model_path = resolve_path(cfg["paths"]["model"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": best["estimator"],
            "threshold": best["threshold"],
            "experiment": best_name,
            "test_metrics": best["metrics_tuned"],
        },
        model_path,
    )
    logger.info("Model saved to %s", model_path)


if __name__ == "__main__":
    main()