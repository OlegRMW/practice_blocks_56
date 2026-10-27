import argparse
import logging
from pathlib import Path

import pandas as pd

from text_classification.artifacts import load_training_run
from text_classification.config import select_models, select_versions
from text_classification.models import evaluate_saved_models
from text_classification.reports import save_prediction_reports


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate test reports from saved classifiers.")
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Directory containing classic-run.yaml and saved model bundles.")
    parser.add_argument("--test", type=Path, help="Test CSV; defaults to the saved data_dir/test.csv.")
    parser.add_argument("--output-dir", type=Path, help="Defaults to saved reports_dir/test.")
    parser.add_argument("--versions", nargs="+", metavar="VERSION")
    parser.add_argument("--models", nargs="+", metavar="MODEL")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        config, _ = load_training_run(args.run_dir)
        select_versions(config, list(config["versions"]) if args.versions is None else args.versions)
        select_models(config, args.models)
        test_path = args.test or Path(config["data_dir"]) / "test.csv"
        test = pd.read_csv(test_path)
        results = evaluate_saved_models(args.run_dir, test, args.versions, args.models)
        output_dir = args.output_dir or Path(config["reports_dir"]) / "test"
        paths = save_prediction_reports(results, test["id"], output_dir)
    except ImportError as error:
        parser.exit(2, f"error: {error}\nInstall the project dependencies with poetry install.\n")
    except KeyError as error:
        parser.exit(2, f"error: Missing data/artifact field: {error}\n")
    except LookupError as error:
        parser.exit(2, f"NLTK resource unavailable: {error}\n")
    except (OSError, ValueError, TypeError, EOFError) as error:
        parser.exit(2, f"error: {error}\n")
    print(f"Saved {len(paths)} test reports to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
