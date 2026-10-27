"""Neural backends; heavy dependencies are imported only when selected."""

import json
import logging
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from text_classification.metrics import calculate_metrics


logger = logging.getLogger(__name__)


def _write_settings(path, settings):
    path.mkdir(parents=True, exist_ok=True)
    (path / "backend.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")


class BGEClassifier:
    def __init__(self, encoder, classifier, batch_size=32, normalize_embeddings=True, device="auto"):
        self.encoder = encoder
        self.classifier = classifier
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.device = device

    def encode(self, texts):
        texts = list(texts)
        logger.info("Computing BGE embeddings for %s texts on %s", len(texts),
                    getattr(self.encoder, "device", self.device))
        return self.encoder.encode(
            texts, batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings, show_progress_bar=True,
        )

    def predict(self, texts):
        return self.classifier.predict(self.encode(texts))

    def offload(self):
        self.encoder.to("cpu")

    def save(self, path):
        path = Path(path)
        _write_settings(path, {
            "kind": "bge", "batch_size": self.batch_size,
            "normalize_embeddings": self.normalize_embeddings, "device": self.device,
        })
        self.encoder.save(str(path / "encoder"))
        joblib.dump(self.classifier, path / "classifier.joblib", compress=3)

    @classmethod
    def load(cls, path):
        from sentence_transformers import SentenceTransformer

        path = Path(path)
        settings = json.loads((path / "backend.json").read_text(encoding="utf-8"))
        settings.pop("kind")
        device = settings["device"]
        encoder = SentenceTransformer(
            str(path / "encoder"), device=None if device == "auto" else device,
            local_files_only=True,
        )
        return cls(encoder, joblib.load(path / "classifier.joblib"), **settings)


def train_bge_classifier(train_texts, labels, seed, **params):
    from sentence_transformers import SentenceTransformer

    params = dict(params)
    encoder_name = params.pop("encoder_name", "BAAI/bge-base-en-v1.5")
    classifier_params = {"max_iter": 1000, "random_state": seed}
    classifier_params.update(params.pop("classifier", {}))
    device = params.get("device", "auto")
    encoder = SentenceTransformer(encoder_name, device=None if device == "auto" else device)
    classifier = LogisticRegression(**classifier_params)
    model = BGEClassifier(encoder, classifier, **params)
    embeddings = model.encode(train_texts)
    logger.info("Training logistic regression on BGE embeddings")
    classifier.fit(embeddings, labels)
    logger.info("BGE classifier training complete")
    return model


class TransformerClassifier:
    def __init__(self, model, tokenizer, max_length=128, batch_size=16, device="auto"):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device

    def predict(self, texts):
        import torch

        texts = list(texts)
        self.model.eval()
        device = next(self.model.parameters()).device
        predictions = []
        with torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                batch = self.tokenizer(
                    texts[start:start + self.batch_size], padding=True,
                    truncation=True, max_length=self.max_length, return_tensors="pt",
                ).to(device)
                logits = self.model(**batch).logits
                predictions.extend(logits.argmax(dim=-1).cpu().tolist())
        return np.asarray(predictions, dtype=np.int64)

    def offload(self):
        self.model.to("cpu")

    def save(self, path):
        path = Path(path)
        _write_settings(path, {"kind": "transformer", "max_length": self.max_length,
                               "batch_size": self.batch_size, "device": self.device})
        self.model.save_pretrained(str(path))
        self.tokenizer.save_pretrained(str(path))

    @classmethod
    def load(cls, path):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        path = Path(path)
        settings = json.loads((path / "backend.json").read_text(encoding="utf-8"))
        settings.pop("kind")
        device = settings["device"]
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForSequenceClassification.from_pretrained(str(path), local_files_only=True)
        model.to(device)
        tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
        return cls(model, tokenizer, **settings)


def _compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    return calculate_metrics(labels, np.argmax(logits, axis=-1))


def train_transformer_classifier(train_texts, labels, val_texts, val_labels, seed, work_dir, **params):
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
        Trainer, TrainingArguments, set_seed,
    )

    params = dict(params)
    model_name = params.pop("model_name", "distilbert/distilbert-base-uncased")
    max_length = params.pop("max_length", 128)
    batch_size = params.pop("inference_batch_size", 16)
    device = params.pop("device", "auto")
    options = {
        "learning_rate": 2e-5, "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 16, "gradient_accumulation_steps": 2,
        "num_train_epochs": 3, "weight_decay": 0.01, "warmup_steps": 100,
        "eval_strategy": "epoch", "save_strategy": "epoch",
        "load_best_model_at_end": True, "metric_for_best_model": "f1",
        "greater_is_better": True, "save_total_limit": 2, "logging_steps": 50,
        "optim": "adamw_torch", "dataloader_num_workers": 0,
    }
    options.update(params.pop("training", {}))
    if params:
        raise ValueError(f"Unknown transformer parameters: {', '.join(params)}.")
    if device == "cpu":
        options["use_cpu"] = True
    elif device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable.")
    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2,
        id2label={0: "NOT_DISASTER", 1: "DISASTER"},
        label2id={"NOT_DISASTER": 0, "DISASTER": 1},
    )

    def dataset(texts, targets):
        frame = Dataset.from_dict({"text": list(texts), "labels": list(targets)})
        return frame.map(
            lambda batch: tokenizer(batch["text"], truncation=True, max_length=max_length),
            batched=True, remove_columns=["text"],
        )

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".distilbert-", dir=work_dir) as checkpoint_dir:
        trainer = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir=checkpoint_dir, seed=seed, data_seed=seed,
                report_to="none", push_to_hub=False, **options,
            ),
            train_dataset=dataset(train_texts, labels),
            eval_dataset=dataset(val_texts, val_labels), processing_class=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
            compute_metrics=_compute_metrics,
        )
        trainer.train()
        return TransformerClassifier(trainer.model, tokenizer, max_length, batch_size, device)


def load_neural_model(kind, path):
    if kind == "bge":
        return BGEClassifier.load(path)
    if kind == "transformer":
        return TransformerClassifier.load(path)
    raise ValueError(f"Unknown neural model kind: {kind}.")
