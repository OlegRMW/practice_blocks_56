# Repository Guidelines

## Project Structure & Module Organization

This repository contains Python notebooks for binary text classification.
- `notebooks/main.ipynb`: text preprocessing variants, TF-IDF, logistic regression, gradient boosting, and BGE sentence embeddings.
- `notebooks/eda.ipynb`: initial exploration of training labels.
- `practice.ipynb`: separate PyTorch tensor exercises.
- `data/`: original training/test CSVs and derived preprocessing datasets.
- `data/reports/`: generated prediction CSVs containing `id` and `target`.

There is no packaged application, dedicated source directory, or automated test suite.

## Build, Test, and Development Commands

Use Python 3.11, matching the notebook metadata. No dependency manifest is currently provided. From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install jupyterlab ipykernel pandas numpy nltk scikit-learn sentence-transformers torch
.\.venv\Scripts\python.exe -m jupyterlab
```

Select the virtual environment's kernel. Run notebooks in `notebooks/` with that directory as the working directory; their CSV paths use `../data/`. Execute cells in order from a fresh kernel. There is no build step. Initial execution downloads NLTK resources and the `BAAI/bge-base-en-v1.5` model, requiring network access.

## Coding Style & Naming Conventions

Use four-space indentation and `snake_case` for functions and variables. Keep preprocessing, feature extraction, training, and evaluation in clearly labeled notebook sections. Follow existing variant names such as `v1_basic` and `v4_lemma_no_stopwords`; report filenames combine the variant and model. No formatter or linter is configured.

## Testing Guidelines

Validate changes by restarting the kernel and running affected notebooks from top to bottom. Preserve the stratified 80/20 validation split and `random_state=42` for comparable experiments. Fit vectorizers and dimensionality reduction only on training rows. Compare accuracy, precision, recall, and F1. Check that exported predictions preserve test-row order, contain one row per test example, and use the `id,target` schema. No automated coverage requirement exists.

## Commit & Pull Request Guidelines

This checkout has no Git metadata, so an established commit convention cannot be verified. Use concise, imperative messages such as `Add stemming comparison`. In pull requests, describe the experiment, affected notebooks/data, validation commands, and metric changes; link relevant issues when available.

## Data & Generated Outputs

Preserve original `data/train.csv` and `data/test.csv`. Review report changes before committing: running `main.ipynb` writes into `data/reports/` and can overwrite existing predictions. Avoid unrelated notebook output churn and never commit credentials or local environment files.
