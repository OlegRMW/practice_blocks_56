from copy import deepcopy
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

from text_classification.artifacts import save_training_run
from text_classification.config import load_config
from text_classification.experiments import run_experiments
from text_classification.models import evaluate_saved_models
from text_classification.preprocessing import preprocess_text


ROOT = Path(__file__).resolve().parents[1]
NEURAL_INSTALLED = all(importlib.util.find_spec(name) for name in
                       ["torch", "transformers", "datasets", "sentence_transformers", "accelerate"])


class FakeEncoder:
    """A deterministic local encoder for orchestration/serialization tests."""
    calls = []

    def __init__(self, name, **kwargs):
        self.calls.append((name, kwargs))

    def encode(self, texts, **kwargs):
        features = np.asarray([[str(t).lower().count("fire"), str(t).lower().count("sunny"), 1.0]
                               for t in texts], dtype=float)
        if kwargs.get("normalize_embeddings"):
            features /= np.linalg.norm(features, axis=1, keepdims=True)
        return features

    def to(self, device):
        return self

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "encoder.json").write_text("{}", encoding="utf-8")


class FakeTransformer:
    def predict(self, texts):
        return np.asarray([int("fire" in str(text).lower()) for text in texts])

    def offload(self):
        pass

    def save(self, path):
        path.mkdir(parents=True, exist_ok=True)
        (path / "model.json").write_text("{}", encoding="utf-8")


class NeuralFixture(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory(prefix=".test-neural-", dir=ROOT)
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name).resolve()
        self.assertTrue(self.folder.is_relative_to(ROOT))
        self.train = pd.DataFrame({
            "id": range(16),
            "text": ["Sunny picnic!" if i % 2 == 0 else "Fire! Not safe." for i in range(16)],
            "target": [i % 2 for i in range(16)],
        })
        self.test = pd.DataFrame({"id": [81, 4], "text": ["Fire!", "Sunny picnic!"]})
        self.config = load_config(ROOT / "configs/neural.yaml")
        self.config.update(models_dir=str(self.folder / "models"),
                           reports_dir=str(self.folder / "reports"), validation_size=0.25)


class NeuralOrchestrationTests(NeuralFixture):
    def test_minimal_preprocessing_preserves_context(self):
        with patch("text_classification.preprocessing._get_stop_words", side_effect=AssertionError("stopwords")):
            self.assertEqual(
                preprocess_text("  Not safe! #Fire &amp; smoke 123\n", mode="minimal", lowercase=False),
                "Not safe! #Fire & smoke 123",
            )

    def test_neural_settings_validation(self):
        self.assertEqual(set(self.config["models"]), {"bge_logistic_regression", "distilbert"})
        for params in [{"max_length": 0}, {"training": {"seed": 10}}, {"device": "typo"}]:
            config = deepcopy(self.config)
            config["models"] = {"distilbert": params}
            path = self.folder / "invalid.yaml"
            path.write_text(yaml.safe_dump(config), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_config(path)

    def test_bge_encoder_and_classifier_are_saved_and_reloaded_locally(self):
        FakeEncoder.calls = []
        with patch.dict("sys.modules", {"sentence_transformers": SimpleNamespace(SentenceTransformer=FakeEncoder)}), \
                patch("text_classification.experiments.build_tfidf_features", side_effect=AssertionError("TF-IDF used")):
            run = run_experiments(self.train, self.config, ["raw"], ["bge_logistic_regression"])
            self.assertEqual(run.vectorizers, {})
            self.assertEqual(run.tfidf_dataset, {})
            self.assertNotIn("predictions_test", run.results_df)
            model = run.trained_models["raw"]["BGE Logistic Regression"]
            expected = model.predict(self.test["text"])
            manifest = save_training_run(run)
            with patch("sklearn.linear_model.LogisticRegression.fit", side_effect=AssertionError("Refit")):
                report = evaluate_saved_models(manifest, self.test)
        np.testing.assert_array_equal(report.iloc[0]["predictions_test"], expected)
        self.assertTrue((self.folder / "models/raw/bge_logistic_regression/model/encoder/encoder.json").is_file())
        self.assertTrue((self.folder / "models/raw/bge_logistic_regression/model/classifier.joblib").is_file())
        self.assertTrue(FakeEncoder.calls[-1][1]["local_files_only"])

    def test_transformer_dispatch_and_saved_backend(self):
        fake = FakeTransformer()
        with patch("text_classification.neural.train_transformer_classifier", return_value=fake) as fit, \
                patch("text_classification.experiments.build_tfidf_features", side_effect=AssertionError("TF-IDF used")):
            run = run_experiments(self.train, self.config, ["raw"], ["distilbert"])
        self.assertEqual(fit.call_count, 1)
        args, kwargs = fit.call_args
        self.assertEqual(len(args[0]), 12)
        self.assertEqual(len(args[2]), 4)
        self.assertEqual(kwargs["training"]["num_train_epochs"], 3)
        self.assertEqual(kwargs["seed"], 42)
        manifest = save_training_run(run)
        with patch("text_classification.neural.load_neural_model", return_value=fake) as load:
            report = evaluate_saved_models(manifest, self.test)
        self.assertEqual(load.call_args.args[0], "transformer")
        np.testing.assert_array_equal(report.iloc[0]["predictions_test"], [1, 0])

    def test_classical_and_bge_can_share_a_run(self):
        self.config["models"]["logistic_regression"] = {"max_iter": 100}
        self.config["tfidf"] = {"min_df": 1, "max_df": 1.0}
        with patch.dict("sys.modules", {"sentence_transformers": SimpleNamespace(SentenceTransformer=FakeEncoder)}):
            run = run_experiments(self.train, self.config, ["raw"],
                                  ["logistic_regression", "bge_logistic_regression"])
            manifest = save_training_run(run)
            reports = evaluate_saved_models(manifest, self.test)
        self.assertEqual(len(reports), 2)
        self.assertEqual(set(run.vectorizers), {"raw"})


@unittest.skipUnless(NEURAL_INSTALLED, "Requires the neural Poetry dependencies")
class TinyNeuralIntegrationTests(NeuralFixture):
    """Exercise real libraries on tiny locally generated models, without downloads."""
    def make_local_transformer(self, classification):
        from transformers import (
            DistilBertConfig, DistilBertModel, DistilBertForSequenceClassification,
            DistilBertTokenizerFast,
        )

        path = self.folder / ("tiny-classifier" if classification else "tiny-encoder")
        path.mkdir()
        vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "fire", "sunny", "picnic", "not", "safe", "!", "."]
        (path / "vocab.txt").write_text("\n".join(vocab) + "\n", encoding="utf-8")
        tokenizer = DistilBertTokenizerFast(vocab_file=str(path / "vocab.txt"), do_lower_case=True)
        tokenizer.save_pretrained(str(path))
        config = DistilBertConfig(vocab_size=len(vocab), dim=16, hidden_dim=32,
                                 n_heads=2, n_layers=1, max_position_embeddings=64,
                                 dropout=0.0, attention_dropout=0.0, num_labels=2)
        model = DistilBertForSequenceClassification(config) if classification else DistilBertModel(config)
        model.save_pretrained(str(path))
        return path

    def test_real_distilbert_training_save_and_predict(self):
        source = self.make_local_transformer(classification=True)
        self.config["models"] = {"distilbert": {
            "model_name": str(source), "max_length": 32, "device": "cpu",
            "training": {"max_steps": 1, "num_train_epochs": 1, "warmup_steps": 0,
                         "per_device_train_batch_size": 2, "gradient_accumulation_steps": 1,
                         "disable_tqdm": True},
        }}
        with patch.dict("os.environ", {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}):
            run = run_experiments(self.train, self.config, ["raw"], ["distilbert"])
            expected = run.trained_models["raw"]["DistilBERT"].predict(self.test["text"])
            manifest = save_training_run(run)
            report = evaluate_saved_models(manifest, self.test)
        np.testing.assert_array_equal(report.iloc[0]["predictions_test"], expected)

    def test_real_bge_backend_roundtrip(self):
        from sentence_transformers import SentenceTransformer
        from sentence_transformers.sentence_transformer import modules

        source = self.make_local_transformer(classification=False)
        transformer = modules.Transformer(str(source), max_seq_length=32)
        pooling = modules.Pooling(transformer.get_embedding_dimension())
        encoder = SentenceTransformer(modules=[transformer, pooling], device="cpu")
        encoder_dir = self.folder / "sentence-encoder"
        encoder.save(str(encoder_dir))
        self.config["models"] = {"bge_logistic_regression": {
            "encoder_name": str(encoder_dir), "batch_size": 2, "device": "cpu",
        }}
        with patch.dict("os.environ", {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}):
            run = run_experiments(self.train, self.config, ["raw"], ["bge_logistic_regression"])
            expected = run.trained_models["raw"]["BGE Logistic Regression"].predict(self.test["text"])
            manifest = save_training_run(run)
            report = evaluate_saved_models(manifest, self.test)
        np.testing.assert_array_equal(report.iloc[0]["predictions_test"], expected)

    def test_neural_cli_train_and_evaluate(self):
        from sentence_transformers import SentenceTransformer
        from sentence_transformers.sentence_transformer import modules

        encoder_source = self.make_local_transformer(classification=False)
        classifier_source = self.make_local_transformer(classification=True)
        transformer = modules.Transformer(str(encoder_source), max_seq_length=32)
        encoder = SentenceTransformer(
            modules=[transformer, modules.Pooling(transformer.get_embedding_dimension())], device="cpu"
        )
        encoder_dir = self.folder / "sentence-encoder"
        encoder.save(str(encoder_dir))
        data = self.folder / "data"
        data.mkdir()
        self.train.to_csv(data / "train.csv", index=False)
        self.test.to_csv(data / "test.csv", index=False)
        self.config["data_dir"] = str(data)
        self.config["models"] = {
            "bge_logistic_regression": {"encoder_name": str(encoder_dir), "batch_size": 2, "device": "cpu"},
            "distilbert": {"model_name": str(classifier_source), "max_length": 32, "device": "cpu",
                           "training": {"max_steps": 1, "warmup_steps": 0,
                                        "per_device_train_batch_size": 2,
                                        "gradient_accumulation_steps": 1, "disable_tqdm": True}},
        }
        path = self.folder / "neural.yaml"
        path.write_text(yaml.safe_dump(self.config), encoding="utf-8")
        env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                   PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
        trained = subprocess.run(
            [sys.executable, "-B", str(ROOT / "scripts/train.py"), "--config", str(path), "--versions", "raw"],
            cwd=self.folder, env=env, capture_output=True, text=True, encoding="utf-8", timeout=120,
        )
        self.assertEqual(trained.returncode, 0, trained.stdout + trained.stderr)
        metrics = pd.read_csv(self.folder / "reports/validation/tfidf-validation-metrics.csv")
        self.assertEqual(set(metrics["model"]), {"BGE Logistic Regression", "DistilBERT"})
        (data / "train.csv").unlink()
        path.unlink()
        evaluated = subprocess.run(
            [sys.executable, "-B", str(ROOT / "scripts/evaluate.py"), "--run-dir", str(self.folder / "models")],
            cwd=self.folder, env=env, capture_output=True, text=True, encoding="utf-8", timeout=120,
        )
        self.assertEqual(evaluated.returncode, 0, evaluated.stdout + evaluated.stderr)
        for label in ["BGE Logistic Regression", "DistilBERT"]:
            report = pd.read_csv(self.folder / "reports/test" / f"raw_{label}.csv")
            self.assertEqual(report["id"].tolist(), self.test["id"].tolist())
            self.assertTrue(report["target"].isin([0, 1]).all())
