# DSBA_EcommerceRecommendations — thesis scaffold edition

Этот репозиторий собран как **каркас ВКР по теме ML & DL for E-commerce Product Recommendations**.
Он показывает не только ноутбуки и модели, но и **понятную структуру работы**: data audit, метрики, baselines,
deep/sequential models, hybrid-срезы и Streamlit-dashboard для демонстрации результата.

## Что здесь появилось поверх исходной базы

- `dashboard/` — multipage Streamlit-интерфейс под защиту и обсуждение с научруком
- `docs/` — каркас работы, storyline, план показа результатов
- `starter_kit/` — таблицы с workstreams, research questions, backlog экспериментов и outline ноутбука
- `starter_kit/two_week_roadmap.csv` — зафиксированный ближайший план без распыления в лишние deep-ветки
- `draft_branches/` — черновые ветки / воркстримы, по которым видно, как проект разворачивается
- `src/` — стартовые python-модули для contracts, локальных путей, экспорта и валидации артефактов
- `config/local_paths.example.toml` — шаблон локальной конфигурации путей к raw-данным и dashboard artifacts

## Быстрый маршрут по проекту

1. `starter_kit/workstreams.csv` — посмотреть, какие потоки работ уже есть.
2. `docs/PROJECT_CARCASS.md` — открыть общую структуру исследования.
3. `dashboard/app.py` — запустить понятный интерфейс для научрука.
4. `starter_kit/notebook_outline.csv` — сопоставить текущий notebook и будущую главу результатов.
5. `docs/DEFENSE_SUMMARY.md` — взять готовую краткую сводку для главы и защиты.

## Как запускать dashboard

```bash
python -m pip install -r dashboard/requirements.txt
python dashboard/generate_demo_artifacts.py
streamlit run dashboard/app.py
```

Если реальные таблицы уже выгружены из ноутбука, положи их в `dashboard/artifacts/`.
Если нет, приложение стартует на `sample_artifacts/` и все равно показывает каркас.

## Notebook-first workflow

1. Создай `config/local_paths.toml` из `config/local_paths.example.toml`.
2. Настрой локальные пути к raw-датасетам и, при необходимости, к `dashboard/artifacts/`.
3. В notebook используй `src/local_data_config.py` для разрешения путей и `src/export_dashboard_artifacts.py` для финальной выгрузки.
4. После выгрузки прогоняй валидацию:

```bash
python -m src.validate_dashboard_artifacts --artifacts-dir dashboard/artifacts
```

## Real-data benchmark

Чтобы быстро наполнить pipeline реальными данными и перегенерировать `dashboard/artifacts/`, используй:

```bash
python -m src.run_real_benchmark --refresh
python -m src.validate_dashboard_artifacts --artifacts-dir dashboard/artifacts
python -m src.export_defense_summary
streamlit run dashboard/app.py
```

Скрипт сам подтянет официальные источники и соберет reproducible research subsets для:

- `MovieLens 20M`
- `Google Local South Carolina`
- `Goodreads Fantasy 10K`

В baseline stack уже входят:

- `Popularity`
- `TruncatedSVD`
- `EASE`
- `TFIDF_Content`
- `SequentialTransition`
- `HybridFeatureRerank`

### Что делает `HybridFeatureRerank`

Это не декоративный ансамбль, а реальный feature-engineered rerank поверх candidate set из `EASE`, `TruncatedSVD` и `TFIDF_Content`.
В score входят:

- collaborative signals: `ease`, `svd`
- content signal: `content`
- catalog controls: `popularity`, `novelty`, `quality`, `freshness`
- profile alignment: `tag_overlap`, `category_match`, `year_alignment`, `price_match`, `richness_alignment`

За счёт этого dashboard умеет показывать не только top-N, но и объяснение, **почему** конкретный item поднялся в выдаче.

### Зафиксированный benchmark по умолчанию

- datasets: `MovieLens 20M`, `Google Local South Carolina`, `Goodreads Fantasy 10K`
- temporal split: `75/25`
- main leaderboard: `exc`, `topk=10`, `NDCG@10`
- supporting metrics: `Recall@10`, `MAP@10`, `Precision@10`, `Coverage`, `Novelty`
- additional slices: `inc`, `topk=5/20`

### Snapshot последнего реального прогона

- `MovieLens 20M`: лучший baseline пока `EASE` с `NDCG@10 = 0.2013`, что логично для более плотного collaborative-сценария.
- `Google Local South Carolina`: лучший результат у `HybridFeatureRerank` с `NDCG@10 = 0.1218`, то есть metadata и профильные признаки здесь реально помогают.
- `Goodreads Fantasy 10K`: `HybridFeatureRerank` выходит на первое место с `NDCG@10 = 0.1690`, а `TFIDF_Content` сразу следом, что хорошо поддерживает content/hybrid narrative.
- `SequentialTransition` уже встроен как честный lightweight order-aware baseline: он закрывает sequential scope без тяжёлого deep-стека, но не становится главным драйвером качества.

Это уже даёт внятную историю для защиты:

- на плотных данных побеждает сильный shallow CF;
- на более текстово-богатых и разреженных доменах выигрывает hybrid/content-aware подход;
- sequential baseline посчитан и показал, что order-aware сигнал полезен, но пока не перевешивает сильные hybrid/shallow решения;
- feature engineering даёт не только метрику, но и объяснимую recommendation sandbox-выдачу.

### Significance and confidence

Для главных сравнений теперь автоматически считаются:

- `metric_intervals.csv` с bootstrap confidence intervals;
- `significance_tests.csv` с paired Wilcoxon tests, confidence intervals на delta и BH-corrected `p_value_adj`.

Это позволяет не только показать разницу в метриках, но и честно ответить, где она статистически подтверждена, а где нет.

## Что стоит сделать следующим шагом

- перенести выводы из `docs/DEFENSE_SUMMARY.md` в экспериментальную главу;
- финально упаковать 5–7 тезисов для научрука и слайдов;
- решить, расширяем ли Goodreads subset без потери воспроизводимости;

## Историческая база


