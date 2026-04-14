# Repo Quickstart

## Что запускать

1. Пересобрать реальные benchmark-артефакты:

```powershell
python -m src.run_real_benchmark --refresh
```

2. Поднять dashboard:

```powershell
streamlit run dashboard/app.py
```

3. Для короткой демонстрации пройти страницы в таком порядке:

- `Home` — карта проекта, workstreams, модельный scope и текущие выводы.
- `Data Audit` — структура данных, sparsity, активность и long-tail.
- `Model Benchmark` — leaderboard, top-K динамика, интервалы и significance.
- `Robustness and Business` — ablation, сегменты, feature importance и business-срезы.
- `Recommendation Sandbox` — пример истории пользователя и объяснимой выдачи.
- `Supervisor Brief` — готовая последовательность показа научруку.

## Где лежат ключевые результаты

- `dashboard/artifacts/` — реальные CSV, которыми питается интерфейс.
- `docs/RESULT_PRESENTATION_BRIEF.docx` — краткое пояснение результата в презентационном формате.
- `docs/DEFENSE_SUMMARY.md` — собранные тезисы по benchmark, ablation и significance.
- `docs/media/dashboard_tour.mp4` — короткий видео-тур по интерфейсу.

## Что важно проговорить при показе

- На `MovieLens 20M` сильнее всего работает плотный collaborative baseline `EASE`.
- На `Google Local South Carolina` лучший результат даёт `HybridFeatureRerank`.
- На `Goodreads Fantasy 10K` доминирует content-aware сигнал, что видно и по ablation, и по significance.
- Dashboard показывает не только среднюю метрику, но и то, почему результату можно доверять и где именно он полезен.
