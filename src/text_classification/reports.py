from pathlib import Path

import pandas as pd


def _write_csv(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def save_predictions(test_ids, predictions, path):
    """Save binary predictions in the input ID order and return the report."""
    ids = pd.Series(test_ids).reset_index(drop=True)
    targets = pd.Series(predictions).reset_index(drop=True)
    if len(ids) != len(targets):
        raise ValueError("Each test ID must have exactly one prediction.")
    if not targets.isin([0, 1]).all():
        raise ValueError("Predictions must contain only binary labels 0 and 1.")

    report = pd.DataFrame({"id": ids, "target": targets.astype("int64")})
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


def save_metrics(results, path):
    """Save experiment metrics without prediction arrays and return the table."""
    metrics = results.drop(
        columns=["predictions_val", "predictions_test"], errors="ignore"
    )
    _write_csv(metrics, path)
    return metrics
