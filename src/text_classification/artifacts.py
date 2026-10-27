from copy import deepcopy
from pathlib import Path
import platform
from importlib.metadata import PackageNotFoundError, version as package_version
from uuid import uuid4

import joblib
import numpy as np
import sklearn
import yaml

from text_classification.config import MODEL_KINDS, MODEL_LABELS, load_config
from text_classification.reports import append_metrics_history, save_metrics, save_validation_reports


def save_training_run(experiment):
    """Persist fitted preprocessing/model bundles, validation reports, and manifest."""
    config = deepcopy(experiment.config)
    models_dir = Path(config["models_dir"])
    validation_dir = Path(config["reports_dir"]) / "validation"
    run_id = uuid4().hex
    libraries = {
        "python": platform.python_version(), "scikit-learn": sklearn.__version__,
        "numpy": np.__version__, "joblib": joblib.__version__,
    }
    for name in ("torch", "transformers", "sentence-transformers", "datasets", "accelerate"):
        try:
            libraries[name] = package_version(name)
        except PackageNotFoundError:
            pass
    artifacts = []
    for version in config["selected_versions"]:
        for name in config["selected_models"]:
            label = MODEL_LABELS[name]
            kind = MODEL_KINDS[name]
            relative_path = (Path(version) / f"{name}.joblib" if kind == "tfidf"
                             else Path(version) / name / "metadata.joblib")
            path = models_dir / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            bundle = {
                "artifact_format": 1, "run_id": run_id, "version": version,
                "model_name": name, "model_label": label,
                "preprocessing": deepcopy(config["versions"][version]),
                "kind": kind,
                "libraries": libraries,
            }
            model = experiment.trained_models[version][label]
            if kind == "tfidf":
                bundle.update(vectorizer=experiment.vectorizers[version], model=model)
            else:
                model.save(path.parent / "model")
                bundle["model_path"] = "model"
            joblib.dump(bundle, path, compress=3)
            artifacts.append({"version": version, "model": name, "kind": kind,
                              "file": relative_path.as_posix()})

    save_validation_reports(experiment.results_df, experiment.validation, validation_dir)
    save_metrics(experiment.results_df, validation_dir / "tfidf-validation-metrics.csv")
    config.update(artifact_format=1, run_id=run_id, libraries=libraries, artifacts=artifacts)
    manifest = models_dir / "classic-run.yaml"
    with manifest.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, allow_unicode=True, sort_keys=False)
    append_metrics_history(experiment.results_df, validation_dir / "metrics-history.csv", run_id)
    return manifest


def load_training_run(run_dir):
    """Load the snapshot accompanying saved models, independent of source YAML."""
    path = Path(run_dir).resolve()
    manifest = path if path.suffix in {".yaml", ".yml"} else path / "classic-run.yaml"
    config = load_config(manifest)
    if config.get("artifact_format") != 1 or not config.get("run_id"):
        raise ValueError("Unsupported or missing training artifact format/run ID.")
    artifacts = config.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("The training manifest contains no saved models.")
    return config, manifest.parent


def load_model_bundle(entry, config, models_dir):
    path = (models_dir / entry["file"]).resolve()
    if not path.is_relative_to(models_dir):
        raise ValueError("Model artifact must be located inside its run directory.")
    bundle = joblib.load(path)
    if (not isinstance(bundle, dict) or bundle.get("artifact_format") != 1
            or bundle.get("run_id") != config["run_id"]
            or bundle.get("version") != entry["version"]
            or bundle.get("model_name") != entry["model"]):
        raise ValueError(f"Model artifact does not match the saved run: {path}.")
    kind = bundle.get("kind", "tfidf")
    if kind != MODEL_KINDS[entry["model"]]:
        raise ValueError(f"Model kind does not match its manifest: {path}.")
    if kind != "tfidf":
        from text_classification.neural import load_neural_model

        model_path = (path.parent / bundle["model_path"]).resolve()
        if not model_path.is_relative_to(path.parent):
            raise ValueError("Neural weights must be inside the artifact directory.")
        bundle["model"] = load_neural_model(kind, model_path)
    return bundle
