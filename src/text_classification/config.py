from copy import deepcopy
from pathlib import Path
import re

import yaml


MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "gradient_boosting": "Gradient Boosting",
    "bge_logistic_regression": "BGE Logistic Regression",
    "distilbert": "DistilBERT",
}
MODEL_NAMES = set(MODEL_LABELS)
MODEL_KINDS = {
    "logistic_regression": "tfidf",
    "gradient_boosting": "tfidf",
    "bge_logistic_regression": "bge",
    "distilbert": "transformer",
}


def _mapping(value, name):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a YAML mapping.")
    return value


def load_config(path):
    """Read experiment settings; resolve data/output paths relative to YAML."""
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as file:
        try:
            config = _mapping(yaml.safe_load(file), "Configuration")
        except yaml.YAMLError as error:
            raise ValueError(f"Invalid YAML in {path}: {error}") from error

    required = {"seed", "data_dir", "models_dir", "reports_dir", "validation_size", "versions", "tfidf", "models"}
    missing = required - config.keys()
    if missing:
        raise ValueError(f"Missing configuration fields: {', '.join(sorted(missing))}.")
    if type(config["seed"]) is not int or not 0 <= config["seed"] < 2**32:
        raise ValueError("seed must be an integer between 0 and 2**32 - 1.")
    size = config["validation_size"]
    if isinstance(size, bool) or not isinstance(size, (int, float)) or not 0 < size < 1:
        raise ValueError("validation_size must be a fraction between 0 and 1.")

    for name in ("data_dir", "models_dir", "reports_dir"):
        if not isinstance(config[name], str) or not config[name].strip():
            raise ValueError(f"{name} must be a nonempty path string.")
        config[name] = str((path.parent / config[name]).resolve())

    versions = _mapping(config["versions"], "versions")
    if not versions:
        raise ValueError("versions must contain at least one preprocessing variant.")
    allowed = {"remove_stopwords", "normalization", "lowercase", "mode"}
    for name, params in versions.items():
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name):
            raise ValueError(f"Invalid preprocessing version name: {name!r}.")
        _mapping(params, f"versions.{name}")
        unknown = params.keys() - allowed
        if unknown:
            raise ValueError(f"Unknown preprocessing parameters for {name}: {', '.join(sorted(unknown))}.")
        for flag in ("remove_stopwords", "lowercase"):
            if flag in params and type(params[flag]) is not bool:
                raise ValueError(f"versions.{name}.{flag} must be true or false.")
        if params.get("normalization") not in (None, "lemma", "stem"):
            raise ValueError(f"versions.{name}.normalization must be null, lemma, or stem.")
        if params.get("mode", "classic") not in {"classic", "minimal"}:
            raise ValueError(f"versions.{name}.mode must be classic or minimal.")
        if params.get("mode") == "minimal" and (params.get("remove_stopwords") or params.get("normalization")):
            raise ValueError(f"versions.{name}: minimal mode preserves stopwords and word forms.")

    tfidf = _mapping(config["tfidf"], "tfidf")
    if "ngram_range" in tfidf:
        ngrams = tfidf["ngram_range"]
        if (not isinstance(ngrams, (list, tuple)) or len(ngrams) != 2
                or any(type(value) is not int for value in ngrams)
                or not 1 <= ngrams[0] <= ngrams[1]):
            raise ValueError("tfidf.ngram_range must contain two ordered positive integers.")
        tfidf["ngram_range"] = tuple(ngrams)

    models = _mapping(config["models"], "models")
    if not models or models.keys() - MODEL_NAMES:
        raise ValueError(f"models must contain one or more of: {', '.join(sorted(MODEL_NAMES))}.")
    for name, params in models.items():
        _mapping(params, f"models.{name}")
        if "random_state" in params or "seed" in params:
            raise ValueError(f"Use the top-level seed instead of models.{name}.random_state/seed.")
        if name == "gradient_boosting" and "svd_components" in params:
            value = params["svd_components"]
            if type(value) is not int or value < 1:
                raise ValueError("models.gradient_boosting.svd_components must be positive.")
        if name in {"bge_logistic_regression", "distilbert"}:
            _validate_neural_params(name, params)
    return config


def _validate_neural_params(name, params):
    allowed = ({"encoder_name", "batch_size", "normalize_embeddings", "device", "classifier"}
               if name == "bge_logistic_regression" else
               {"model_name", "max_length", "inference_batch_size", "device", "training"})
    unknown = params.keys() - allowed
    if unknown:
        raise ValueError(f"Unknown parameters for {name}: {', '.join(sorted(unknown))}.")
    if params.get("device", "auto") not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"models.{name}.device must be auto, cpu, or cuda.")
    for key in ("batch_size", "inference_batch_size", "max_length"):
        if key in params and (type(params[key]) is not int or params[key] < 1):
            raise ValueError(f"models.{name}.{key} must be a positive integer.")
    source = "encoder_name" if name == "bge_logistic_regression" else "model_name"
    if source in params and (not isinstance(params[source], str) or not params[source].strip()):
        raise ValueError(f"models.{name}.{source} must be a nonempty model ID/path.")
    if name == "bge_logistic_regression":
        if "normalize_embeddings" in params and type(params["normalize_embeddings"]) is not bool:
            raise ValueError("normalize_embeddings must be true or false.")
        classifier = _mapping(params.get("classifier", {}), f"models.{name}.classifier")
        if "random_state" in classifier:
            raise ValueError("Use the top-level seed for the BGE classifier.")
    else:
        training = _mapping(params.get("training", {}), f"models.{name}.training")
        reserved = {"output_dir", "seed", "data_seed", "report_to", "push_to_hub"} & training.keys()
        if reserved:
            raise ValueError(f"Training settings managed by the runner: {', '.join(sorted(reserved))}.")


def _select(available, names, kind):
    if names is None or isinstance(names, str) or not names:
        raise ValueError(f"Select at least one {kind} explicitly.")
    names = list(dict.fromkeys(names))
    unknown = [name for name in names if name not in available]
    if unknown:
        raise ValueError(
            f"Unknown {kind}: {', '.join(unknown)}. Available: {', '.join(available)}."
        )
    return {name: deepcopy(available[name]) for name in names}


def select_versions(config, names):
    """Return only explicitly requested variants, preserving request order."""
    return _select(config["versions"], names, "preprocessing versions")


def select_models(config, names=None):
    """Select requested classifiers, or all classifiers declared in YAML."""
    return _select(config["models"], list(config["models"]) if names is None else names, "models")
