# Defense Summary

## Core benchmark
- Goodreads Fantasy 10K: лучший результат даёт HybridFeatureRerank (0.1671 NDCG@10).
- Google Local South Carolina: лучший результат даёт HybridFeatureRerank (0.1220 NDCG@10).
- MovieLens 20M: лучший результат даёт EASE (0.2013 NDCG@10).
- Goodreads Fantasy 10K: hybrid уверенно обходит чистый CF, значит доменные признаки и rerank реально добавляют сигнал.
- Goodreads Fantasy 10K: content-aware retrieval заметно сильнее popularity, текстовые и metadata-признаки здесь оправданы.
- Goodreads Fantasy 10K: в hybrid-слое главные драйверы качества сейчас content, ease.
- Google Local South Carolina: в hybrid-слое главные драйверы качества сейчас ease, svd.
- MovieLens 20M: в hybrid-слое главные драйверы качества сейчас ease, svd.

## Neural and sequential references
- Goodreads Fantasy 10K: SASRec даёт NDCG@10 = 0.0867 и занимает 6-е место из 8 моделей в основном `exc` leaderboard.
- Goodreads Fantasy 10K: NCF даёт NDCG@10 = 0.0142 и занимает 8-е место из 8 моделей в основном `exc` leaderboard.
- Google Local South Carolina: SASRec даёт NDCG@10 = 0.0248 и занимает 5-е место из 8 моделей в основном `exc` leaderboard.
- Google Local South Carolina: NCF даёт NDCG@10 = 0.0045 и занимает 8-е место из 8 моделей в основном `exc` leaderboard.
- MovieLens 20M: SASRec даёт NDCG@10 = 0.1328 и занимает 6-е место из 8 моделей в основном `exc` leaderboard.
- MovieLens 20M: NCF даёт NDCG@10 = 0.0209 и занимает 8-е место из 8 моделей в основном `exc` leaderboard.

## Ablation reading
- Goodreads Fantasy 10K: content-heavy ablation неожиданно сильнее полного hybrid, это стоит трактовать как сигнал к перенастройке rerank weights.
- Goodreads Fantasy 10K: content/profile features дают заметный вклад в итоговый ranking.
- Google Local South Carolina: collaborative signals критичны; без них hybrid резко деградирует.
- Google Local South Carolina: content/profile signals почти не меняют итоговый ranking, домен ближе к pure collaborative case.
- MovieLens 20M: collaborative signals критичны; без них hybrid резко деградирует.
- MovieLens 20M: content/profile signals почти не меняют итоговый ranking, домен ближе к pure collaborative case.

## Significance reading
- Goodreads Fantasy 10K: HybridFeatureRerank статистически выше EASE по NDCG@10 (delta +0.0485, 95% CI [+0.0231, +0.0774], p_adj=0.0030).
- Goodreads Fantasy 10K: разница между HybridFeatureRerank и SequentialTransition по NDCG@10 не подтверждена статистически (delta +0.0458, 95% CI [-0.0012, +0.0879], p_adj=0.0380).
- Goodreads Fantasy 10K: HybridFeatureRerank статистически выше Hybrid_NoContentAblation по NDCG@10 (delta +0.0624, 95% CI [+0.0343, +0.0938], p_adj=0.0005).
- Goodreads Fantasy 10K: разница между HybridFeatureRerank и Hybrid_NoCollaborativeAblation по NDCG@10 не подтверждена статистически (delta -0.0065, 95% CI [-0.0351, +0.0221], p_adj=0.7416).
- Google Local South Carolina: HybridFeatureRerank статистически выше EASE по NDCG@10 (delta +0.0067, 95% CI [+0.0029, +0.0105], p_adj=0.0001).
- Google Local South Carolina: HybridFeatureRerank статистически выше SequentialTransition по NDCG@10 (delta +0.0602, 95% CI [+0.0534, +0.0671], p_adj=0.0000).
- Google Local South Carolina: разница между HybridFeatureRerank и Hybrid_NoContentAblation по NDCG@10 не подтверждена статистически (delta +0.0023, 95% CI [-0.0007, +0.0054], p_adj=0.1597).
- Google Local South Carolina: HybridFeatureRerank статистически выше Hybrid_NoCollaborativeAblation по NDCG@10 (delta +0.1127, 95% CI [+0.1062, +0.1192], p_adj=0.0000).
- MovieLens 20M: разница между HybridFeatureRerank и EASE по NDCG@10 не подтверждена статистически (delta -0.0042, 95% CI [-0.0081, -0.0001], p_adj=0.3419).
- MovieLens 20M: HybridFeatureRerank статистически выше SequentialTransition по NDCG@10 (delta +0.0375, 95% CI [+0.0279, +0.0472], p_adj=0.0000).
- MovieLens 20M: разница между HybridFeatureRerank и Hybrid_NoContentAblation по NDCG@10 не подтверждена статистически (delta -0.0027, 95% CI [-0.0055, +0.0001], p_adj=0.1176).
- MovieLens 20M: HybridFeatureRerank статистически выше Hybrid_NoCollaborativeAblation по NDCG@10 (delta +0.1398, 95% CI [+0.1311, +0.1489], p_adj=0.0000).

## Ablation table (NDCG@10, exc)

| Dataset | Hybrid | No Content | No Collaborative | Delta vs No Content | Delta vs No Collaborative |
|---|---:|---:|---:|---:|---:|
| Goodreads Fantasy 10K | 0.1671 | 0.1046 | 0.1736 | +0.0625 | -0.0065 |
| Google Local South Carolina | 0.1220 | 0.1197 | 0.0093 | +0.0023 | +0.1127 |
| MovieLens 20M | 0.1971 | 0.1998 | 0.0573 | -0.0027 | +0.1398 |

## Final framing
- Main line of the thesis: strong baselines -> hybrid contribution -> robustness/business interpretation.
- MovieLens should be framed as a dense collaborative benchmark where EASE is already very hard to beat.
- Google Local should be framed as the most convincing applied hybrid case: collaborative backbone plus profile-alignment features.
- Goodreads should be framed as a content-dominant domain where hybrid helps, but the ablation shows that content signals currently carry the main gain.
- SASRec now covers the self-attentive sequential reference scope, but it does not displace the strongest shallow or hybrid baselines.
- NCF should be reported as a neural reference model that did not prove competitive under the current implicit-feedback protocol.
