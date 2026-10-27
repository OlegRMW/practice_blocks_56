import argparse
from pathlib import Path

from text_classification.reports import save_metrics_ranking


def main(argv=None):
    parser = argparse.ArgumentParser(description="Rank the latest validation results per preprocessing/model pair.")
    parser.add_argument("--reports-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "data/reports",
                        help="Search this directory recursively for metrics-history.csv files.")
    parser.add_argument("--metric", choices=["f1", "accuracy", "precision", "recall"], default="f1")
    parser.add_argument("--output", type=Path, help="Defaults to reports-dir/metrics-ranking.csv.")
    args = parser.parse_args(argv)
    output = args.output or args.reports_dir / "metrics-ranking.csv"
    try:
        histories = sorted(args.reports_dir.rglob("metrics-history.csv"))
        ranking = save_metrics_ranking(histories, output, args.metric)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f"error: {error}\n")
    print(ranking[["version", "model", "accuracy", "precision", "recall", "f1"]].to_string(index=False))
    print(f"Saved {len(ranking)} combinations sorted by {args.metric}: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
