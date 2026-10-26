# Text classification

Сравнение TF-IDF + Logistic Regression / Gradient Boosting и эксперименты с BGE и DistilBERT.
Python 3.12, зависимости фиксируются Poetry.

## Установка

```powershell
poetry install
poetry run python -m nltk.downloader stopwords wordnet omw-1.4
```

## Обучение выбранных вариантов

Из корня проекта:

```powershell
poetry run python scripts/train.py --config configs/classic.yaml --versions v1_basic v5_stem
```

Будут подготовлены только `v1_basic` и `v5_stem`, затем на каждом из них обучатся
классификаторы, объявленные в YAML. Аргумент `--versions` обязателен.
`train` читает только `train.csv`: обучает на train-части, считает метрики на validation
и сохраняет модели с отчётами. Файл `test.csv` на этом этапе не требуется.
Чтобы выбрать и модели:

```powershell
poetry run python scripts/train.py --config configs/classic.yaml --versions v3_lemma --models logistic_regression
```

Версии из стандартного конфига: `v1_basic`, `v2_no_stopwords`, `v3_lemma`,
`v4_lemma_no_stopwords`, `v5_stem`, `v6_original_case`.
Модели: `logistic_regression`, `gradient_boosting`.
Повторяющиеся имена запускаются один раз; неизвестные имена вызывают ошибку до обработки данных.

## Предсказания на test по сохранённым моделям

```powershell
poetry run python scripts/evaluate.py --run-dir models/classic
```

`evaluate` загружает модели из последнего сохранённого запуска и создаёт финальные CSV
для `test.csv`. Используются сохранённые параметры предобработки и TF-IDF; повторного
обучения нет. Исходный YAML и `train.csv` для этого шага не требуются.
Можно выбрать часть сохранённых моделей и указать другой CSV:

```powershell
poetry run python scripts/evaluate.py --run-dir models/classic --versions v1_basic --models logistic_regression --test data/test.csv
```

Без `--test` берётся `data_dir/test.csv` из сохранённого запуска.
`--output-dir` позволяет изменить папку финальных отчётов.

## YAML и результаты

В `configs/classic.yaml` находятся параметры предобработки для каждой версии,
настройки TF-IDF, моделей, seed, доля validation и пути `data_dir`, `models_dir`, `reports_dir`.
Пути разрешаются относительно YAML-файла. TF-IDF и SVD обучаются только на train.

Результаты обучения:

- `models_dir/<версия>/<модель>.joblib` — параметры предобработки, обученный TF-IDF
  и классификатор. Для Gradient Boosting сохраняется также обученный SVD.
- `models_dir/classic-run.yaml` — фактические настройки, версии библиотек и список моделей.
- `reports_dir/validation/*_validation.csv` — `id,target,prediction`: правильный ответ и прогноз.
- `reports_dir/validation/tfidf-validation-metrics.csv` — accuracy, precision, recall и F1.

Результаты `evaluate`: `reports_dir/test/<версия>_<модель>.csv` со столбцами `id,target`.
В test нет правильных ответов, поэтому метрики качества на нём не считаются.

Повторное обучение перезаписывает одноимённые артефакты и отчёты. Манифест описывает
последний запуск; оставшиеся файлы других комбинаций автоматически не подключаются.
Для хранения нескольких экспериментов указывай отдельные `models_dir` и `reports_dir`.
Загружай только собственные доверенные `.joblib`-файлы и используй то же Poetry-окружение.

## Ноутбуки

Открывай ноутбуки с рабочей директорией `notebooks/` и ядром Poetry `.venv`.
В `main.ipynb` и `modern.ipynb` меняй `selected_versions` и `selected_models`;
настройки читаются из того же YAML, запуск использует общий `run_classic_experiments`.
После обучения `save_training_run` сохраняет модели и validation-отчёты.
Следующая отдельная ячейка использует `evaluate_saved_models` для финальных test-отчётов.
Эксперименты BGE и fine-tuning DistilBERT находятся в отдельных разделах `main.ipynb`.

## Проверки

```powershell
poetry run python -m unittest discover -s tests -v
```
