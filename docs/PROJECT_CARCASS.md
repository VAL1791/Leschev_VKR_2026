# PROJECT_CARCASS

## Идея

Каркас построен так, чтобы научрук или комиссия могли открыть репозиторий и быстро понять:

1. какие данные используются;
2. как устроена offline evaluation;
3. какие baseline и advanced модели сравниваются;
4. где находится новизна;
5. как результаты переводятся в business-интерпретацию.

## Логика работы

### Блок 1. Data foundation
- единый schema для трех датасетов;
- локальный конфиг путей к raw-данным;
- quality gates;
- temporal split;
- EDA и long-tail diagnostics.

### Блок 2. Metrics and baselines
- фиксированный primary benchmark: `exc`, `topk=10`, `NDCG@10`;
- ranking-метрики как основной критерий;
- exc. и inc. режимы;
- сильные baseline-модели как точка отсчета: `Popularity`, `TruncatedSVD`, `EASE`, `TFIDF_Content`.

### Блок 3. Hybrid as main contribution
- `HybridFeatureRerank` как основной proposed layer;
- feature engineering поверх collaborative и content candidate generators;
- explainability для sandbox;
- задача блока — показать не просто рост accuracy, а осмысленный вклад engineered features.

### Блок 4. Robustness and business view
- RFM-сегменты;
- cold-start и long-tail slices;
- cannibalization;
- business proxy метрики.

### Блок 5. Visualization and defense
- Streamlit-dashboard;
- recommendation sandbox;
- concise supervisor brief;
- итоговая storyline для защиты.

### Блок 6. Selective phase 2
- только один sequential reference (`SASRec` или `GRU4Rec`);
- `NCF` и `RBM` остаются вторичными ветками, а не центром проекта;
- задача блока — не размывать основную историю ради количества моделей.

## Главный принцип

Это не просто витрина "много моделей".
Это исследование вида:

**data quality -> consistent protocol -> strong baselines -> hybrid contribution -> robustness -> interpretation**

## Приоритетная траектория

Самая сильная версия работы сейчас выглядит так:

1. честный benchmark на сильных shallow/content baselines;
2. feature-engineered hybrid rerank как главный вклад;
3. сегментный и business-анализ как усиление практической значимости;
4. один selective sequential experiment как phase-2 extension, если не страдает основной storyline.
