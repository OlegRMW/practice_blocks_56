import argparse
import logging
from pathlib import Path

from text_classification.config import load_config, select_models, select_versions


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Train selected TF-IDF, BGE, or transformer models with validation reports."
    )
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).resolve().parents[1] / "configs/classic.yaml")
    parser.add_argument("--versions", nargs="+", required=True, metavar="VERSION")
    parser.add_argument("--models", nargs="+", metavar="MODEL",
                        help="Optional classifier names; defaults to models declared in YAML.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        config = load_config(args.config)
        versions = select_versions(config, args.versions)
        models = select_models(config, args.models)

        from text_classification.artifacts import save_training_run
        from text_classification.data import load_train_data
        from text_classification.experiments import run_experiments

        train = load_train_data(config["data_dir"])
        experiment = run_experiments(train, config, list(versions), list(models))
        manifest = save_training_run(experiment)
        metrics = experiment.results_df.drop(columns=["predictions_val"])
    except ImportError as error:
        parser.exit(2, f"error: {error}\nInstall the project dependencies with poetry install.\n")
    except KeyError as error:
        parser.exit(2, f"error: Missing data/configuration field: {error}\n")
    except LookupError as error:
        parser.exit(2, f"NLTK resource unavailable: {error}\n"
                    "Install resources with: poetry run python -m nltk.downloader stopwords wordnet omw-1.4\n")
    except (OSError, ValueError, TypeError) as error:
        parser.exit(2, f"error: {error}\n")
    print(metrics.to_string(index=False))
    print(f"Saved {len(metrics)} trained models; run manifest: {manifest}")
    print(f"Validation reports: {Path(config['reports_dir']) / 'validation'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
