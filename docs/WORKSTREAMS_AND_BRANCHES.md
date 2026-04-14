# WORKSTREAMS_AND_BRANCHES

Ниже — удобная схема черновых веток. Она нужна не ради Git как такового, а чтобы проект читался как взрослая исследовательская работа.

## Предлагаемые ветки

### `research/data-foundation`
Сюда входят:
- унификация схемы данных;
- EDA;
- quality checks;
- temporal split;
- экспорт dataset-level diagnostics.

### `research/metrics-baselines`
Сюда входят:
- выбор главных метрик;
- реализация leaderboard;
- Popularity / ALS / простые baseline;
- exc./inc. сравнение.

### `research/deep-sequential`
Сюда входят:
- RBM;
- NCF;
- SASRec;
- новые модели EASE и GRU4Rec.

### `research/hybrid-business`
Сюда входят:
- LightGBM_Hybrid;
- RFM;
- long-tail / cold-start slices;
- feature importance;
- business proxy интерпретация.

### `product/dashboard-defense`
Сюда входят:
- Streamlit pages;
- финальная упаковка результатов;
- supervisor brief;
- demo flow для показа.

## Как этим пользоваться

Если нужно показать каркас работы, не надо открывать все ноутбуки подряд.
Достаточно открыть:
1. `starter_kit/workstreams.csv`
2. `draft_branches/`
3. `dashboard/app.py`

По этим трем точкам уже видно, где база, где новизна и куда двигается проект.
