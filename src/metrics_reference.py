from __future__ import annotations

METRIC_REFERENCE = {
    "ndcg": {
        "why_use": "Главная ranking-метрика для ВКР: учитывает и релевантность, и порядок позиций.",
        "when_report": "Основная итоговая таблица leaderboard и сравнение top-K.",
        "short_comment": "Чем выше NDCG, тем лучше модель расставляет релевантные товары наверх.",
    },
    "recall": {
        "why_use": "Показывает полноту: сколько релевантных товаров модель вообще нашла.",
        "when_report": "Вместе с NDCG, особенно для e-commerce catalog retrieval.",
        "short_comment": "Recall полезен, если важно не потерять релевантные товары в длинном каталоге.",
    },
    "precision": {
        "why_use": "Показывает чистоту выдачи в top-K.",
        "when_report": "Как дополнительная метрика рядом с NDCG и Recall.",
        "short_comment": "Precision помогает понять, насколько top-K не засорен нерелевантными товарами.",
    },
    "map": {
        "why_use": "Суммирует quality по позициям, где встретились релевантные объекты.",
        "when_report": "Как дополнительный ranking signal для финальной таблицы.",
        "short_comment": "MAP удобен, когда нужно показать стабильность ранжирования внутри top-K.",
    },
    "coverage": {
        "why_use": "Показывает, насколько широко модель использует каталог, а не только head items.",
        "when_report": "В блоке beyond-accuracy и long-tail analysis.",
        "short_comment": "Coverage нужен, чтобы работа не выглядела как погоня только за popularity bias.",
    },
    "novelty": {
        "why_use": "Позволяет показать, что модель выдает не только самые популярные позиции.",
        "when_report": "В блоке business / robustness, рядом с coverage.",
        "short_comment": "Novelty помогает аргументировать практическую полезность рекомендаций.",
    },
}
