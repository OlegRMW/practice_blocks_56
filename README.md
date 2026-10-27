# Text classification of Disaster Tweets

## 1. Описание задачи 

> Задача бинарной классификации твитов. 
> Требуется понять, есть ли в твите упоминания и природных катаклизмах или нет. 

### Используемые подходы: 
- TF-IDF + Classic ML классификация 
- Предобученная BERT-подобная модель BGE-M3 + Classic ML классификация 
- Fine Tuning BERT-подобной модели DistilBERT

### Обоснование выбора DistilBERT для Fine-Tuning:
Для решения задачи бинарной классификации сообщений выбрана предобученная модель DistilBERT. Она обучена на англоязычных текстах, что соответствует языку нашего датасета, и предназначена для последующего дообучения, в том числе для классификации последовательностей. По сравнению с BERT модель компактнее и требует меньше вычислительных ресурсов. Карточка модели (https://huggingface.co/distilbert/distilbert-base-uncased).

При обучении обновляются веса трансформера и классификационной головы. Для предобработки используется минимальная очистка и штатный токенизатор модели: без стемминга, лемматизации и удаления стоп-слов, чтобы сохранить контекст и отрицания. Эффективность подхода проверяется сравнением F1 с TF-IDF + логистической регрессией на одинаковом validation-разбиении.

### Важные моменты по Classic ML моделям: 
В работе в pipe line не входит подбор гиперпараметров для log-регрессии и градиентного бустинга, так как эти модели рассматривались в роли экспериментальных. 

### Архитектура проекта: 
Проект оформлен в виде отдельных модулей и скриптов. Есть два отдельных скрипта для обучения и инференса модели (train.py/evaluate.py).

Конфигурация моделей и датасетов хранится в папке configs в YAML формате. 

| Модуль | За что отвечает |
|---|---|
| `config.py` | Читает YAML, проверяет параметры, выбирает запрошенные модели и версии предобработки. |
| `data.py` | Загружает CSV и делает воспроизводимое стратифицированное разбиение на train/validation. |
| `preprocessing.py` | Очищает тексты: базовая обработка, стоп-слова, стемминг, лемматизация либо минимальная очистка для трансформеров. |
| `tfidf.py` | Обучает TF-IDF на train и преобразует остальные выборки тем же векторизатором. |
| `models.py` | Обучает классические модели: логистическую регрессию и бустинг с SVD. Также содержит общий запуск предсказаний сохранённых моделей. |
| `neural.py` | Реализует BGE + логистическую регрессию и fine-tuning DistilBERT, включая предсказания и сохранение этих моделей. |
| `metrics.py` | Считает accuracy, precision, recall и F1. |
| `experiments.py` | Собирает эксперимент: разбиение → предобработка → признаки → обучение → оценка на validation. |
| `artifacts.py` | Сохраняет и загружает модели, векторизаторы и описание запуска; проверяет соответствие артефактов запуску. |
| `reports.py` | Записывает CSV с метриками, validation-предсказаниями и финальными test-предсказаниями. |

### Наглядный блокнот 
Блокнот с более наглядным начальным решением находится в notebooks/main.ipynb

## 2. Развертывание и запуск 

### Установка

```powershell
poetry install
poetry run python -m nltk.downloader stopwords wordnet omw-1.4
```

### Создание папки с датасетами 

В корневой директории требуется создать папку data и поместить туда train.csv и test.csv файлы с Kaggle. 

### Обучение выбранных вариантов

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

### Сводный отчёт по validation

Каждый новый запуск `train` дописывает метрики в `validation/metrics-history.csv`
внутри своего `reports_dir`, сохраняя время и идентификатор запуска. История начинается
с запусков после добавления этой возможности; прежние перезаписанные результаты не восстанавливаются.

```powershell
poetry run python scripts/report.py
```

Скрипт собирает историю из `data/reports/` и вложенных папок, включая neural,
оставляет последний результат каждой связки `version + model` и сортирует по F1
по убыванию. Итог сохраняется в `data/reports/metrics-ranking.csv`; история сохраняется полностью.
Другая метрика выбирается через `--metric accuracy`, `precision` или `recall`.
Для отдельной папки используй `--reports-dir data/reports/neural`,
для другого файла результата — `--output data/reports/comparison.csv`.
Для сопоставимости результатов сохраняй одинаковые данные, seed и validation-разбиение.

### BGE и fine-tuning DistilBERT

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

### Предсказания на test по сохранённым моделям

```powershell
poetry run python scripts/evaluate.py --run-dir models/classic
```

`evaluate` загружает модели из последнего сохранённого запуска train.py и создаёт финальные CSV
для `test.csv`. Используются сохранённые параметры предобработки и соответствующий backend; повторного
обучения нет. Исходный YAML и `train.csv` для этого шага не требуются.
Можно выбрать часть сохранённых моделей и указать другой CSV:

```powershell
poetry run python scripts/evaluate.py --run-dir models/classic --versions v1_basic --models logistic_regression --test data/test.csv
```

Без `--test` берётся `data_dir/test.csv` из сохранённого запуска.
`--output-dir` позволяет изменить папку финальных отчётов.

### Формирование отчёта по метрикам

```powershell
poetry run python scripts/report.py
```

Скрипт собирает накопленные validation-метрики classic и neural, оставляет последний
результат каждой связки «предобработка + модель» и сортирует по F1 по убыванию.
Отчёт сохраняется в `data/reports/metrics-ranking.csv`.
Для другой метрики добавляем метки `--metric accuracy`, `--metric precision` или `--metric recall`.
