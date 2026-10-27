from dataclasses import dataclass
from copy import deepcopy
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from text_classification.config import MODEL_KINDS, MODEL_LABELS, select_models, select_versions
from text_classification.data import split_indices
from text_classification.metrics import calculate_metrics
from text_classification.models import train_gradient_boosting, train_logistic_regression
from text_classification.preprocessing import prepare_text_frame
from text_classification.tfidf import build_tfidf_features


logger = logging.getLogger(__name__)


@dataclass
class ExperimentResult:
    results_df: pd.DataFrame
    vectorizers: dict
    trained_models: dict
    tfidf_dataset: dict
    train_idx: np.ndarray
    val_idx: np.ndarray
    validation: pd.DataFrame
    config: dict


def run_experiments(train, config, version_names, model_names=None):
    """Fit selected variants on train and compute validation predictions only."""
    versions = select_versions(config, version_names)
    models = select_models(config, model_names)
    missing = {"id", "text", "target"} - set(train.columns)
    if missing:
        raise ValueError(f"Training CSV is missing columns: {', '.join(sorted(missing))}.")
    if not train["target"].isin([0, 1]).all():
        raise ValueError("Training targets must be binary labels 0 and 1 without missing values.")
    train_idx, val_idx = split_indices(
        train, seed=config["seed"], test_size=config["validation_size"]
    )
    y_train = train["target"].iloc[train_idx]
    y_val = train["target"].iloc[val_idx]
    prepared_train = prepare_text_frame(train, versions)
    trainers = {
        "logistic_regression": train_logistic_regression,
        "gradient_boosting": train_gradient_boosting,
    }
    vectorizers, trained_models, datasets, results = {}, {}, {}, []
    for version in versions:
        logger.info("Version: %s", version)
        train_texts = prepared_train[version].iloc[train_idx]
        val_texts = prepared_train[version].iloc[val_idx]
        features = {}
        if any(MODEL_KINDS[name] == "tfidf" for name in models):
            vectorizer, features = build_tfidf_features(train_texts, val_texts, **config["tfidf"])
            vectorizers[version] = vectorizer
            datasets[version] = features
        trained_models[version] = {}
        for model_name, params in models.items():
            label, kind = MODEL_LABELS[model_name], MODEL_KINDS[model_name]
            logger.info("Training %s on %s", model_name, version)
            if kind == "tfidf":
                model = trainers[model_name](features["X_train"], y_train, seed=config["seed"], **params)
                pred_val = model.predict(features["X_val"])
            elif kind == "bge":
                from text_classification.neural import train_bge_classifier

                model = train_bge_classifier(train_texts, y_train, seed=config["seed"], **params)
                pred_val = model.predict(val_texts)
                model.offload()
            else:
                from text_classification.neural import train_transformer_classifier

                model = train_transformer_classifier(
                    train_texts, y_train, val_texts, y_val, seed=config["seed"],
                    work_dir=Path(config["models_dir"]) / ".training", **params,
                )
                pred_val = model.predict(val_texts)
                model.offload()
            trained_models[version][label] = model
            results.append({
                "version": version,
                "model": label,
                **calculate_metrics(y_val, pred_val),
                "predictions_val": pred_val,
            })

    results_df = pd.DataFrame(results).sort_values("f1", ascending=False).reset_index(drop=True)
    run_config = deepcopy(config)
    run_config.update(versions=versions, models=models,
                      selected_versions=list(versions), selected_models=list(models))
    return ExperimentResult(
        results_df, vectorizers, trained_models, datasets, train_idx, val_idx,
        train.iloc[val_idx][["id", "target"]].reset_index(drop=True).copy(), run_config,
    )


def run_classic_experiments(train, config, version_names, model_names=None):
    """Backward-compatible entry point for existing notebooks."""
    return run_experiments(train, config, version_names, model_names)
