from copy import deepcopy
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

from text_classification.artifacts import save_training_run
from text_classification.config import load_config, select_models, select_versions
from text_classification.data import split_indices
from text_classification.experiments import run_classic_experiments
from text_classification.models import evaluate_saved_models
from text_classification.preprocessing import prepare_text_versions, preprocess_text


ROOT = Path(__file__).resolve().parents[1]


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory(prefix=".test-experiments-", dir=ROOT)
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name).resolve()
        self.assertTrue(self.folder.is_relative_to(ROOT))
        data = self.folder / "data"
        data.mkdir()
        self.train = pd.DataFrame({
            "id": np.arange(60),
            "text": ["Fire smoke rescue near rivers" if i % 2 else "Sunny picnics in green parks"
                     for i in range(60)],
            "target": [i % 2 for i in range(60)],
        })
        self.test = pd.DataFrame({
            "id": [102, 91, 73, 88],
            "text": ["Fire rescue", "Sunny parks", "Smoke near rivers", "Green picnics"],
        })
        self.train.to_csv(data / "train.csv", index=False)
        self.test.to_csv(data / "test.csv", index=False)
        self.raw_config = {
            "seed": 42,
            "data_dir": "data",
            "models_dir": "models",
            "reports_dir": "reports",
            "validation_size": 0.25,
            "versions": {
                "v1_basic": {},
                "v5_stem": {"normalization": "stem"},
                "unused_lemma": {"normalization": "lemma", "remove_stopwords": True},
            },
            "tfidf": {"ngram_range": [1, 2], "min_df": 1, "max_df": 1.0},
            "models": {
                "logistic_regression": {"C": 0.25, "max_iter": 100},
                "gradient_boosting": {"svd_components": 3, "n_estimators": 3, "max_depth": 2},
            },
        }
        self.path = self.folder / "experiment.yaml"
        self.write_config(self.raw_config)
        self.config = load_config(self.path)

    def write_config(self, config):
        self.path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    def cli(self, *args):
        return subprocess.run(
            [sys.executable, "-B", str(ROOT / "scripts/train.py"), "--config", str(self.path), *args],
            cwd=self.folder, text=True, capture_output=True, timeout=60,
        )

    def test_config_paths_and_selection(self):
        self.assertEqual(self.config["data_dir"], str(self.folder / "data"))
        self.assertEqual(self.config["tfidf"]["ngram_range"], (1, 2))
        selected = select_versions(self.config, ["v5_stem", "v1_basic", "v5_stem"])
        self.assertEqual(list(selected), ["v5_stem", "v1_basic"])
        selected["v1_basic"]["lowercase"] = False
        self.assertEqual(self.config["versions"]["v1_basic"], {})
        self.assertEqual(list(select_models(self.config, ["logistic_regression"])), ["logistic_regression"])

    def test_unknown_and_empty_selections_fail_before_processing(self):
        with patch("text_classification.experiments.prepare_text_frame") as prepare:
            for names in [[], ["unknown"], None]:
                with self.assertRaises(ValueError):
                    run_classic_experiments(self.train, self.config, names)
            with self.assertRaises(ValueError):
                run_classic_experiments(self.train, self.config, ["v1_basic"], ["unknown"])
            prepare.assert_not_called()

    def test_invalid_yaml_settings_are_rejected(self):
        for change in [
            {"validation_size": 2},
            {"seed": True},
            {"tfidf": {"ngram_range": [2, 1]}},
            {"versions": {"basic": {"lowercase": "false"}}},
            {"models": {"logistic_regression": {"random_state": 7}}},
        ]:
            config = deepcopy(self.raw_config)
            config.update(change)
            self.write_config(config)
            with self.assertRaises(ValueError):
                load_config(self.path)

    def test_prepares_only_selected_versions(self):
        versions = select_versions(self.config, ["v5_stem"])
        with patch("text_classification.preprocessing.preprocess_text", wraps=preprocess_text) as process:
            train, test = prepare_text_versions(self.train, self.test, versions)
        self.assertEqual(process.call_count, len(self.train) + len(self.test))
        self.assertTrue(all(call.kwargs == {"normalization": "stem"} for call in process.call_args_list))
        self.assertEqual(set(train) - set(self.train), {"v5_stem"})
        self.assertEqual(set(test) - set(self.test), {"v5_stem"})
        self.assertNotIn("v5_stem", self.train)

    def test_trains_only_selected_models_and_uses_yaml_params(self):
        with patch("text_classification.preprocessing._get_stop_words", side_effect=AssertionError("Unused stopwords loaded")), \
                patch("text_classification.experiments.train_gradient_boosting", side_effect=AssertionError("Unused model trained")):
            result = run_classic_experiments(
                self.train, self.config,
                ["v1_basic", "v5_stem", "v1_basic"], ["logistic_regression"],
            )
        self.assertEqual(set(result.results_df["version"]), {"v1_basic", "v5_stem"})
        self.assertEqual(len(result.results_df), 2)
        self.assertNotIn("predictions_test", result.results_df)
        self.assertTrue(all("X_test" not in features for features in result.tfidf_dataset.values()))
        self.assertEqual(set(result.vectorizers), {"v1_basic", "v5_stem"})
        expected_train, expected_val = split_indices(self.train, seed=42, test_size=0.25)
        np.testing.assert_array_equal(result.train_idx, expected_train)
        np.testing.assert_array_equal(result.val_idx, expected_val)
        for version in result.vectorizers:
            self.assertEqual(result.vectorizers[version].ngram_range, (1, 2))
            self.assertTrue(any(" " in word for word in result.vectorizers[version].vocabulary_))
            model = result.trained_models[version]["Logistic Regression"]
            self.assertEqual(model.C, 0.25)
            self.assertEqual(model.max_iter, 100)
            self.assertEqual(model.random_state, 42)

    def test_gradient_boosting_uses_yaml_params(self):
        result = run_classic_experiments(
            self.train, self.config, ["v1_basic"], ["gradient_boosting"]
        )
        model = result.trained_models["v1_basic"]["Gradient Boosting"]
        self.assertEqual(model.named_steps["svd"].n_components, 3)
        self.assertEqual(model.named_steps["classifier"].n_estimators, 3)
        self.assertEqual(model.named_steps["classifier"].max_depth, 2)

    def test_train_cli_saves_models_and_validation_without_test_csv(self):
        (self.folder / "data/test.csv").unlink()
        completed = self.cli("--versions", "v5_stem", "v1_basic", "--models", "logistic_regression")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        reports = self.folder / "reports/validation"
        self.assertEqual({p.name for p in reports.iterdir()}, {
            "v1_basic_Logistic Regression_validation.csv", "v5_stem_Logistic Regression_validation.csv",
            "tfidf-validation-metrics.csv",
        })
        _, val_idx = split_indices(self.train, seed=42, test_size=0.25)
        for name in ["v1_basic", "v5_stem"]:
            report = pd.read_csv(reports / f"{name}_Logistic Regression_validation.csv")
            self.assertEqual(report.columns.tolist(), ["id", "target", "prediction"])
            self.assertEqual(report["id"].tolist(), self.train["id"].iloc[val_idx].tolist())
            self.assertEqual(report["target"].tolist(), self.train["target"].iloc[val_idx].tolist())
            self.assertTrue(report["prediction"].isin([0, 1]).all())
            self.assertTrue((self.folder / "models" / name / "logistic_regression.joblib").is_file())
        metrics = pd.read_csv(reports / "tfidf-validation-metrics.csv")
        self.assertEqual(len(metrics), 2)
        self.assertNotIn("predictions_test", metrics)
        snapshot = yaml.safe_load((self.folder / "models/classic-run.yaml").read_text(encoding="utf-8"))
        self.assertEqual(snapshot["selected_versions"], ["v5_stem", "v1_basic"])
        self.assertEqual(list(snapshot["models"]), ["logistic_regression"])
        self.assertEqual(set(snapshot["versions"]), {"v5_stem", "v1_basic"})
        self.assertEqual(len(snapshot["artifacts"]), 2)
        self.assertFalse((self.folder / "reports/test").exists())

    def test_saved_predictions_match_memory_without_any_refitting(self):
        result = run_classic_experiments(
            self.train, self.config, ["v1_basic", "v5_stem"]
        )
        manifest = save_training_run(result)
        expected = {}
        for version, models in result.trained_models.items():
            texts = self.test["text"].map(
                lambda text: preprocess_text(text, **self.config["versions"][version])
            )
            features = result.vectorizers[version].transform(texts)
            for label, model in models.items():
                expected[(version, label)] = model.predict(features)
        # Neither changing the source YAML nor removing train.csv affects inference.
        self.raw_config["versions"]["v1_basic"] = {"lowercase": False}
        self.write_config(self.raw_config)
        (self.folder / "data/train.csv").unlink()
        with patch("sklearn.linear_model.LogisticRegression.fit", side_effect=AssertionError("Refit")), \
                patch("sklearn.ensemble.GradientBoostingClassifier.fit", side_effect=AssertionError("Refit")), \
                patch("sklearn.decomposition.TruncatedSVD.fit_transform", side_effect=AssertionError("Refit")), \
                patch("sklearn.feature_extraction.text.TfidfVectorizer.fit_transform", side_effect=AssertionError("Refit")):
            predictions = evaluate_saved_models(manifest, self.test)
        self.assertEqual(len(predictions), 4)
        for row in predictions.itertuples(index=False):
            np.testing.assert_array_equal(row.predictions_test, expected[(row.version, row.model)])

    def test_evaluate_cli_uses_saved_snapshot_and_preserves_test_order(self):
        trained = self.cli("--versions", "v1_basic", "v5_stem")
        self.assertEqual(trained.returncode, 0, trained.stderr)
        self.path.unlink()
        (self.folder / "data/train.csv").unlink()
        evaluated = subprocess.run([
            sys.executable, "-B", str(ROOT / "scripts/evaluate.py"),
            "--run-dir", str(self.folder / "models"),
            "--versions", "v5_stem", "--models", "logistic_regression",
        ], cwd=self.folder, capture_output=True, text=True, timeout=60)
        self.assertEqual(evaluated.returncode, 0, evaluated.stderr)
        reports = self.folder / "reports/test"
        self.assertEqual({p.name for p in reports.iterdir()}, {"v5_stem_Logistic Regression.csv"})
        report = pd.read_csv(reports / "v5_stem_Logistic Regression.csv")
        self.assertEqual(report.columns.tolist(), ["id", "target"])
        self.assertEqual(report["id"].tolist(), self.test["id"].tolist())

    def test_mismatched_artifact_is_rejected(self):
        import joblib

        result = run_classic_experiments(self.train, self.config, ["v1_basic"], ["logistic_regression"])
        manifest = save_training_run(result)
        artifact = self.folder / "models/v1_basic/logistic_regression.joblib"
        bundle = joblib.load(artifact)
        bundle["run_id"] = "another-run"
        joblib.dump(bundle, artifact)
        with self.assertRaisesRegex(ValueError, "does not match"):
            evaluate_saved_models(manifest, self.test)

    def test_cli_requires_versions_and_rejects_unknown_names(self):
        for args in [(), ("--versions", "unknown"), ("--versions", "v1_basic", "--models", "unknown")]:
            completed = self.cli(*args)
            self.assertEqual(completed.returncode, 2, completed.stderr)
            self.assertFalse((self.folder / "reports").exists())


if __name__ == "__main__":
    unittest.main()
