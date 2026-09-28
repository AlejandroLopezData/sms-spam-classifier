"""Pipeline definitions: features + classifier."""
from __future__ import annotations

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MinMaxScaler

from spam_detection.features import TextStats, clean_text

MODEL_NAMES = ("naive_bayes", "logreg", "random_forest", "gradient_boosting")


def build_features(tfidf_params: dict, use_manual_features: bool = True) -> FeatureUnion:
    """TF-IDF on cleaned text, optionally stacked with scaled hand-crafted features."""
    params = dict(tfidf_params)
    if "ngram_range" in params:
        params["ngram_range"] = tuple(params["ngram_range"])

    transformers = [("tfidf", TfidfVectorizer(preprocessor=clean_text, **params))]
    if use_manual_features:
        stats = Pipeline(
            [
                ("extract", TextStats(log_transform=True)),
                # [0, 1] range keeps features non-negative (needed by MultinomialNB) and
                # comparable to TF-IDF values. clip=True protects against unseen extremes.
                ("scale", MinMaxScaler(clip=True)),
            ]
        )
        transformers.append(("stats", stats))
    return FeatureUnion(transformers)


def get_classifier(name: str, random_state: int = 42):
    """Return an untuned classifier by name."""
    if name == "naive_bayes":
        return MultinomialNB()
    if name == "logreg":
        return LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state)
    if name == "random_forest":
        # n_jobs=1 here: parallelism is handled by GridSearchCV
        return RandomForestClassifier(
            n_estimators=300, class_weight="balanced_subsample", n_jobs=1, random_state=random_state
        )
    if name == "gradient_boosting":
        return GradientBoostingClassifier(random_state=random_state)
    raise ValueError(f"Unknown model '{name}'. Choose from {MODEL_NAMES}.")


def build_pipeline(
    model_name: str,
    tfidf_params: dict,
    use_manual_features: bool = True,
    random_state: int = 42,
) -> Pipeline:
    """Full pipeline: raw text in, spam probability out."""
    return Pipeline(
        [
            ("features", build_features(tfidf_params, use_manual_features)),
            ("clf", get_classifier(model_name, random_state)),
        ]
    )