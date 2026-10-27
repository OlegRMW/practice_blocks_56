# Text classification

Сравнение TF-IDF + Logistic Regression / Gradient Boosting и эксперименты с BGE и DistilBERT.
Python 3.12, зависимости фиксируются Poetry.

## Установка

```powershell
poetry install
poetry run python -m nltk.downloader stopwords wordnet omw-1.4
```

`pyproject.toml` и `poetry.lock` покрывают TF-IDF, BGE + Logistic Regression,
fine-tuning DistilBERT, скрипты `train`/`evaluate`, тесты и Jupyter.
Установи Python 3.12 и выполни `poetry install` — версии прямых и транзитивных
зависимостей берутся из lock-файла. Сохраняй оба файла в Git.
Ресурсы NLTK и предобученные веса Hugging Face скачиваются отдельно от Python-пакетов.

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
Классические модели: `logistic_regression`, `gradient_boosting`.
Повторяющиеся имена запускаются один раз; неизвестные имена вызывают ошибку до обработки данных.

## BGE и fine-tuning DistilBERT

Общие команды используют отдельный конфиг `configs/neural.yaml`:

```powershell
poetry run python scripts/train.py --config configs/neural.yaml --versions raw --models bge_logistic_regression
poetry run python scripts/train.py --config configs/neural.yaml --versions raw --models distilbert
```

Чтобы обучить оба подхода в одном запуске, передай
`--models bge_logistic_regression distilbert`. Вариант `raw` исправляет HTML-сущности
и пробелы, сохраняя регистр, отрицания, пунктуацию и словоформы. Предобработка,
имена моделей, batch size, параметры классификатора и `Trainer` задаются в YAML.
Другие версии также можно объявить в этом конфиге и выбрать через `--versions`.

BGE остаётся предобученным encoder: обучается Logistic Regression на его эмбеддингах.
Для DistilBERT дообучаются все веса трансформера и слой классификации через `Trainer`.
Разбиение train/validation и метрики одинаковы для всех подходов.
Первый запуск скачивает веса с Hugging Face; для fine-tuning желательно использовать GPU.
`device: cpu` принудительно выбирает CPU; `auto` использует доступное устройство.

```powershell
poetry run python scripts/evaluate.py --run-dir models/neural
```

После обучения encoder BGE и классификатор, либо DistilBERT и токенизатор,
сохраняются локально. `evaluate` загружает эти файлы с `local_files_only=True`
и не скачивает модель заново. Манифест описывает последний запуск в этой папке:
для нескольких экспериментов используй разные `models_dir`/`reports_dir`.

## Предсказания на test по сохранённым моделям

```powershell
poetry run python scripts/evaluate.py --run-dir models/classic
```

`evaluate` загружает модели из последнего сохранённого запуска и создаёт финальные CSV
для `test.csv`. Используются сохранённые параметры предобработки и соответствующий backend; повторного
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
- `models_dir/<версия>/<нейронная модель>/metadata.joblib` и `model/` — метаданные,
  сохранённый encoder и классификатор BGE либо веса DistilBERT и токенизатор.
  Имя манифеста `classic-run.yaml` сохранено для совместимости и используется для всех backend.
- `reports_dir/validation/*_validation.csv` — `id,target,prediction`: правильный ответ и прогноз.
- `reports_dir/validation/tfidf-validation-metrics.csv` — accuracy, precision, recall и F1.

Результаты `evaluate`: `reports_dir/test/<версия>_<модель>.csv` со столбцами `id,target`.
В test нет правильных ответов, поэтому метрики качества на нём не считаются.

Повторное обучение перезаписывает одноимённые артефакты и отчёты. Манифест описывает
последний запуск; оставшиеся файлы других комбинаций автоматически не подключаются.
Для хранения нескольких экспериментов указывай отдельные `models_dir` и `reports_dir`.
Загружай только собственные доверенные `.joblib`-файлы и используй то же Poetry-окружение.

