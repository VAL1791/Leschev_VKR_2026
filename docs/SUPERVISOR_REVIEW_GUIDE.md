# SUPERVISOR_REVIEW_GUIDE

## Что показать научруку за 5 минут

### 1) Project map
Открыть `dashboard/app.py` и страницу overview.
Цель: показать, что есть не хаотичный набор ноутбуков, а карта проекта.

### 2) Data audit
Открыть страницу Data Audit.
Цель: доказать, что данные очищены, сопоставимы и подходят для offline evaluation.

### 3) Leaderboard
Открыть страницу Model Benchmark.
Цель: показать главный ranking comparison.

### 4) Robustness / business
Открыть страницу Robustness.
Цель: показать, что работа не ограничивается NDCG в одной таблице.

### 5) Recommendation sandbox
Открыть sandbox.
Цель: дать понятный человеческий пример того, как рекомендации выглядят для конкретного пользователя.

## Что особенно хорошо воспринимается на защите

- единый экспериментальный протокол;
- explainability у hybrid ranker;
- comparison across slices;
- качество + стоимость обучения;
- отдельный разговор про long-tail и coverage.

## Что не стоит делать

- не перегружать главную таблицу регрессионными метриками;
- не смешивать content-only подходы и full recommender stack без пояснения роли baseline;
- не показывать слишком много сырых notebook outputs без storyline.
