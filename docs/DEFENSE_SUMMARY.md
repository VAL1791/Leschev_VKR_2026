# Defense Summary

## Core benchmark
- Goodreads Fantasy 10K: лучший результат даёт HybridFeatureRerank (0.1685 NDCG@10).
- Google Local South Carolina: лучший результат даёт HybridFeatureRerank (0.1217 NDCG@10).
- MovieLens 20M: лучший результат даёт EASE (0.2013 NDCG@10).
- Goodreads Fantasy 10K: hybrid уверенно обходит чистый CF, значит доменные признаки и rerank реально добавляют сигнал.
- Goodreads Fantasy 10K: content-aware retrieval заметно сильнее popularity, текстовые и metadata-признаки здесь оправданы.
- Goodreads Fantasy 10K: в hybrid-слое главные драйверы качества сейчас content, ease.
- Google Local South Carolina: в hybrid-слое главные драйверы качества сейчас ease, svd.
- MovieLens 20M: в hybrid-слое главные драйверы качества сейчас ease, svd.

## Ablation reading
- Goodreads Fantasy 10K: content-heavy ablation неожиданно сильнее полного hybrid, это стоит трактовать как сигнал к перенастройке rerank weights.
- Goodreads Fantasy 10K: content/profile features дают заметный вклад в итоговый ranking.
- Google Local South Carolina: collaborative signals критичны; без них hybrid резко деградирует.
- Google Local South Carolina: content/profile signals почти не меняют итоговый ranking, домен ближе к pure collaborative case.
- MovieLens 20M: collaborative signals критичны; без них hybrid резко деградирует.
- MovieLens 20M: content/profile signals почти не меняют итоговый ranking, домен ближе к pure collaborative case.

## Significance reading
- Goodreads Fantasy 10K: HybridFeatureRerank статистически выше EASE по NDCG@10 (delta +0.0500, 95% CI [+0.0256, +0.0781], p_adj=0.0027).
- Goodreads Fantasy 10K: разница между HybridFeatureRerank и SequentialTransition по NDCG@10 не подтверждена статистически (delta +0.0473, 95% CI [-0.0001, +0.0891], p_adj=0.0360).
- Goodreads Fantasy 10K: HybridFeatureRerank статистически выше Hybrid_NoContentAblation по NDCG@10 (delta +0.0614, 95% CI [+0.0331, +0.0925], p_adj=0.0010).
- Goodreads Fantasy 10K: разница между HybridFeatureRerank и Hybrid_NoCollaborativeAblation по NDCG@10 не подтверждена статистически (delta -0.0088, 95% CI [-0.0383, +0.0201], p_adj=0.7416).
- Google Local South Carolina: HybridFeatureRerank статистически выше EASE по NDCG@10 (delta +0.0064, 95% CI [+0.0026, +0.0099], p_adj=0.0003).
- Google Local South Carolina: HybridFeatureRerank статистически выше SequentialTransition по NDCG@10 (delta +0.0599, 95% CI [+0.0534, +0.0670], p_adj=0.0000).
- Google Local South Carolina: разница между HybridFeatureRerank и Hybrid_NoContentAblation по NDCG@10 не подтверждена статистически (delta +0.0013, 95% CI [-0.0022, +0.0045], p_adj=0.3690).
- Google Local South Carolina: HybridFeatureRerank статистически выше Hybrid_NoCollaborativeAblation по NDCG@10 (delta +0.1129, 95% CI [+0.1065, +0.1200], p_adj=0.0000).
- MovieLens 20M: разница между HybridFeatureRerank и EASE по NDCG@10 не подтверждена статистически (delta -0.0036, 95% CI [-0.0079, +0.0003], p_adj=0.4866).
- MovieLens 20M: HybridFeatureRerank статистически выше SequentialTransition по NDCG@10 (delta +0.0382, 95% CI [+0.0287, +0.0476], p_adj=0.0000).
- MovieLens 20M: разница между HybridFeatureRerank и Hybrid_NoContentAblation по NDCG@10 не подтверждена статистически (delta -0.0023, 95% CI [-0.0053, +0.0004], p_adj=0.3304).
- MovieLens 20M: HybridFeatureRerank статистически выше Hybrid_NoCollaborativeAblation по NDCG@10 (delta +0.1402, 95% CI [+0.1305, +0.1498], p_adj=0.0000).

## Ablation table (NDCG@10, exc)

| Dataset | Hybrid | No Content | No Collaborative | Delta vs No Content | Delta vs No Collaborative |
|---|---:|---:|---:|---:|---:|
| Goodreads Fantasy 10K | 0.1685 | 0.1071 | 0.1773 | +0.0614 | -0.0088 |
| Google Local South Carolina | 0.1217 | 0.1204 | 0.0088 | +0.0013 | +0.1129 |
| MovieLens 20M | 0.1977 | 0.2000 | 0.0575 | -0.0023 | +0.1402 |

## Final framing
- Main line of the thesis: strong baselines -> hybrid contribution -> robustness/business interpretation.
- MovieLens should be framed as a dense collaborative benchmark where EASE is already very hard to beat.
- Google Local should be framed as the most convincing applied hybrid case: collaborative backbone plus profile-alignment features.
- Goodreads should be framed as a content-dominant domain where hybrid helps, but the ablation shows that content signals currently carry the main gain.
- Sequential models should stay phase-2 scope: one selective reference model is enough for completeness.
