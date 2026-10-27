from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def _write_csv(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _prediction_frame(test_ids, predictions):
    ids = pd.Series(test_ids).reset_index(drop=True)
    targets = pd.Series(predictions).reset_index(drop=True)
    if len(ids) != len(targets):
        raise ValueError("Each test ID must have exactly one prediction.")
    if not targets.isin([0, 1]).all():
        raise ValueError("Predictions must contain only binary labels 0 and 1.")

    return pd.DataFrame({"id": ids, "target": targets.astype("int64")})


def save_predictions(test_ids, predictions, path):
    """Save binary predictions in the input ID order and return the report."""
    report = _prediction_frame(test_ids, predictions)
    _write_csv(report, path)
    return report


def save_prediction_reports(results, test_ids, output_dir):
    """Save one prediction CSV per preprocessing variant and classifier."""
    output_dir = Path(output_dir)
    paths = []
    for row in results.itertuples(index=False):
        path = output_dir / f"{row.version}_{row.model}.csv"
        save_predictions(test_ids, row.predictions_test, path)
        paths.append(path)
    return paths


def save_validation_reports(results, validation, output_dir):
    """Save ID, true target, and predicted label for every validation example."""
    output_dir = Path(output_dir)
    targets = pd.Series(validation["target"]).reset_index(drop=True)
    if not targets.isin([0, 1]).all():
        raise ValueError("Validation targets must be binary labels 0 and 1.")
    paths = []
    for row in results.itertuples(index=False):
        report = _prediction_frame(validation["id"], row.predictions_val).rename(
            columns={"target": "prediction"}
        )
        report.insert(1, "target", targets.astype("int64"))
        path = output_dir / f"{row.version}_{row.model}_validation.csv"
        _write_csv(report, path)
        paths.append(path)
    return paths


def save_metrics(results, path):
    """Save experiment metrics without prediction arrays and return the table."""
    metrics = results.drop(
        columns=["predictions_val", "predictions_test"], errors="ignore"
    )
    _write_csv(metrics, path)
    return metrics


def append_metrics_history(results, path, run_id):
    """Append validation metrics for a completed training run without predictions."""
    history = results[["version", "model", "accuracy", "precision", "recall", "f1"]].copy()
    history["recorded_at"] = datetime.now(timezone.utc).isoformat()
    history["run_id"] = run_id
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    history.to_csv(path, mode="a", header=not path.exists() or path.stat().st_size == 0, index=False)


def save_metrics_ranking(history_paths, path, metric="f1"):
    """Keep the latest result per preprocessing/model pair and rank descending."""
    if metric not in {"accuracy", "precision", "recall", "f1"}:
        raise ValueError(f"Unknown quality metric: {metric}.")
    history_paths = list(history_paths)
    if not history_paths:
        raise ValueError("No metrics-history.csv files found. Run train.py first to collect metrics.")
    history = pd.concat([pd.read_csv(source) for source in history_paths], ignore_index=True)
    history["recorded_at"] = pd.to_datetime(history["recorded_at"], format="ISO8601", utc=True)
    ranking = (
        history.sort_values("recorded_at", kind="stable")
        .drop_duplicates(subset=["version", "model"], keep="last")
        .sort_values(metric, ascending=False, kind="stable")
        .reset_index(drop=True)
    )
    _write_csv(ranking, path)
    return ranking
