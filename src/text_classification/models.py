import logging

import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from text_classification.artifacts import load_model_bundle, load_training_run
from text_classification.config import select_models, select_versions
from text_classification.preprocessing import preprocess_text


logger = logging.getLogger(__name__)


def train_logistic_regression(X_train, y_train, seed, **model_params):

    params = {"max_iter": 1000, "random_state": seed}
    params.update(model_params)
    model = LogisticRegression(**params)
    model.fit(X_train, y_train)
    return model


def train_gradient_boosting(
    X_train, y_train, seed, svd_components=100, **model_params
):
    n_components = min(svd_components, X_train.shape[1], X_train.shape[0] - 1)
    params = {"random_state": seed}
    params.update(model_params)
    model = Pipeline([
        ("svd", TruncatedSVD(n_components=n_components, random_state=seed)),
        ("classifier", GradientBoostingClassifier(**params)),
    ])
    model.fit(X_train, y_train)
    return model


def evaluate_saved_models(run_dir, test, version_names=None, model_names=None):
    """Predict test labels using saved text preprocessing and model backends."""
    config, models_dir = load_training_run(run_dir)
    versions = select_versions(config, list(config["versions"]) if version_names is None else version_names)
    models = select_models(config, model_names)
    missing = {"id", "text"} - set(test.columns)
    if missing:
        raise ValueError(f"Test CSV is missing columns: {', '.join(sorted(missing))}.")
    entries = [entry for entry in config["artifacts"]
               if entry["version"] in versions and entry["model"] in models]
    expected = {(version, model) for version in versions for model in models}
    actual = {(entry["version"], entry["model"]) for entry in entries}
    if actual != expected or len(entries) != len(expected):
        raise ValueError("Saved run is missing or duplicates a requested version/model combination.")
    processed, results = {}, []
    for entry in entries:
        bundle = load_model_bundle(entry, config, models_dir)
        version = entry["version"]
        logger.info("Predicting test: %s / %s", version, entry["model"])
        if version not in processed:
            processed[version] = test["text"].map(
                lambda text: preprocess_text(text, **bundle["preprocessing"])
            )
        features = (bundle["vectorizer"].transform(processed[version])
                    if bundle.get("kind", "tfidf") == "tfidf" else processed[version])
        predictions = bundle["model"].predict(features)
        results.append({
            "version": version, "model": bundle["model_label"],
            "predictions_test": predictions,
        })
    return pd.DataFrame(results)
