from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from dashboard_core import ensure_directory, export_artifacts


RNG = np.random.default_rng(42)

DATASETS = ['MovieLens 20M', 'Instacart', 'Online Retail']
ALGORITHMS = ['Popularity', 'ALS', 'RBM', 'NCF', 'SASRec', 'EASE', 'GRU4Rec', 'LightGBM_Hybrid']
TOPK = [5, 10, 20]
MODES = ['inc', 'exc']


def make_datasets_summary() -> pd.DataFrame:
    return pd.DataFrame([
        {
            'dataset': 'MovieLens 20M',
            'users': 138493,
            'items': 26744,
            'events': 20000263,
            'sparsity_pct': 99.46,
            'mean_events_per_user': 144.4,
            'mean_events_per_item': 747.9,
            'time_start': '1995-01',
            'time_end': '2015-03',
        },
        {
            'dataset': 'Instacart',
            'users': 206209,
            'items': 49688,
            'events': 33819106,
            'sparsity_pct': 99.67,
            'mean_events_per_user': 164.0,
            'mean_events_per_item': 680.6,
            'time_start': '2017-01',
            'time_end': '2017-08',
        },
        {
            'dataset': 'Online Retail',
            'users': 4339,
            'items': 3684,
            'events': 401604,
            'sparsity_pct': 97.49,
            'mean_events_per_user': 92.6,
            'mean_events_per_item': 109.0,
            'time_start': '2010-12',
            'time_end': '2011-12',
        },
    ])


def make_results() -> pd.DataFrame:
    base_by_dataset = {
        'MovieLens 20M': 0.11,
        'Instacart': 0.13,
        'Online Retail': 0.15,
    }
    lift_by_algo = {
        'Popularity': 0.00,
        'ALS': 0.018,
        'RBM': 0.012,
        'NCF': 0.024,
        'SASRec': 0.030,
        'EASE': 0.027,
        'GRU4Rec': 0.028,
        'LightGBM_Hybrid': 0.034,
    }
    rows = []
    for dataset in DATASETS:
        for algo in ALGORITHMS:
            for topk in TOPK:
                for mode in MODES:
                    mode_bonus = 0.012 if mode == 'inc' else 0.0
                    topk_adj = {5: 0.006, 10: 0.0, 20: -0.004}[topk]
                    noise = float(RNG.normal(0, 0.002))
                    ndcg = base_by_dataset[dataset] + lift_by_algo[algo] + mode_bonus + topk_adj + noise
                    recall = ndcg * 2.15 + 0.01 + float(RNG.normal(0, 0.004))
                    precision = ndcg * 0.95 + float(RNG.normal(0, 0.003))
                    map_k = ndcg * 0.82 + float(RNG.normal(0, 0.002))
                    coverage = {
                        'Popularity': 0.06,
                        'ALS': 0.17,
                        'RBM': 0.15,
                        'NCF': 0.19,
                        'SASRec': 0.22,
                        'EASE': 0.21,
                        'GRU4Rec': 0.23,
                        'LightGBM_Hybrid': 0.25,
                    }[algo] + {5: 0.0, 10: 0.015, 20: 0.04}[topk]
                    novelty = {
                        'Popularity': 0.20,
                        'ALS': 0.28,
                        'RBM': 0.27,
                        'NCF': 0.30,
                        'SASRec': 0.33,
                        'EASE': 0.32,
                        'GRU4Rec': 0.34,
                        'LightGBM_Hybrid': 0.36,
                    }[algo] + {5: 0.0, 10: 0.01, 20: 0.02}[topk]
                    train_seconds = {
                        'Popularity': 5.6,
                        'ALS': 252.0,
                        'RBM': 728.0,
                        'NCF': 860.0,
                        'SASRec': 1190.0,
                        'EASE': 40.0,
                        'GRU4Rec': 940.0,
                        'LightGBM_Hybrid': 125.0,
                    }[algo]
                    inference_ms = {
                        'Popularity': 0.08,
                        'ALS': 1.10,
                        'RBM': 2.75,
                        'NCF': 2.30,
                        'SASRec': 3.80,
                        'EASE': 0.42,
                        'GRU4Rec': 2.95,
                        'LightGBM_Hybrid': 1.60,
                    }[algo]
                    rows.append({
                        'dataset': dataset,
                        'algorithm': algo,
                        'topk': topk,
                        'mode': mode,
                        'ndcg': round(ndcg, 4),
                        'recall': round(recall, 4),
                        'precision': round(precision, 4),
                        'map': round(map_k, 4),
                        'coverage': round(min(coverage, 0.99), 4),
                        'novelty': round(min(novelty, 0.99), 4),
                        'train_seconds': train_seconds,
                        'inference_ms_per_user': inference_ms,
                    })
    return pd.DataFrame(rows)


def make_cannibalization(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, algorithm, topk), grp in results.groupby(['dataset', 'algorithm', 'topk']):
        try:
            inc = grp[grp['mode'] == 'inc'].iloc[0]
            exc = grp[grp['mode'] == 'exc'].iloc[0]
        except IndexError:
            continue
        rows.append({
            'dataset': dataset,
            'algorithm': algorithm,
            'topk': topk,
            'map_exc': exc['map'],
            'map_inc': inc['map'],
            'ndcg_exc': exc['ndcg'],
            'ndcg_inc': inc['ndcg'],
            'precision_exc': exc['precision'],
            'precision_inc': inc['precision'],
            'recall_exc': exc['recall'],
            'recall_inc': inc['recall'],
            'ndcg_gap': round(inc['ndcg'] - exc['ndcg'], 4),
            'recall_gap': round(inc['recall'] - exc['recall'], 4),
            'precision_gap': round(inc['precision'] - exc['precision'], 4),
            'map_gap': round(inc['map'] - exc['map'], 4),
        })
    return pd.DataFrame(rows)


def make_rfm_metrics() -> pd.DataFrame:
    segments = ['Champions', 'Loyal', 'Potential', 'At Risk']
    rows = []
    base_by_algo = {
        'Popularity': 0.09, 'ALS': 0.11, 'RBM': 0.106, 'NCF': 0.122,
        'SASRec': 0.128, 'EASE': 0.124, 'GRU4Rec': 0.126, 'LightGBM_Hybrid': 0.136
    }
    for algo in ALGORITHMS:
        for i, seg in enumerate(segments):
            ndcg = base_by_algo[algo] + (0.012 - i * 0.008) + float(RNG.normal(0, 0.002))
            recall = ndcg * 1.95 + float(RNG.normal(0, 0.004))
            rows.append({
                'algorithm': algo,
                'rfm_segment': seg,
                'ndcg': round(ndcg, 4),
                'recall': round(recall, 4),
                'n_users': int([402, 734, 513, 261][i] * (1 + RNG.uniform(-0.05, 0.05))),
            })
    return pd.DataFrame(rows)


def make_revenue_mix() -> pd.DataFrame:
    rows = []
    for rfm in [3, 6, 9, 12]:
        for cat in ['low_margin', 'mid_margin', 'high_margin', 'bundle_friendly']:
            share = float(RNG.uniform(8, 28))
            rows.append({
                'RFM_score': rfm,
                'proxy_category': cat,
                'share': round(share, 2),
                'revenue': round(share * RNG.uniform(80, 130), 2),
            })
    return pd.DataFrame(rows)


def make_slice_metrics() -> pd.DataFrame:
    rows = []
    slices = ['Cold users', 'Frequent users', 'Head items', 'Long-tail items']
    lift = {
        'Popularity': [0.00, 0.00, 0.01, -0.02],
        'ALS': [0.01, 0.015, 0.012, 0.006],
        'RBM': [0.005, 0.010, 0.008, 0.004],
        'NCF': [0.012, 0.016, 0.010, 0.007],
        'SASRec': [0.015, 0.018, 0.009, 0.010],
        'EASE': [0.013, 0.016, 0.010, 0.009],
        'GRU4Rec': [0.014, 0.017, 0.009, 0.011],
        'LightGBM_Hybrid': [0.018, 0.020, 0.012, 0.014],
    }
    for dataset in DATASETS:
        for algo in ALGORITHMS:
            for idx, slice_name in enumerate(slices):
                base = {'MovieLens 20M': 0.10, 'Instacart': 0.12, 'Online Retail': 0.14}[dataset]
                ndcg = base + lift[algo][idx] + float(RNG.normal(0, 0.002))
                recall = ndcg * 2.0 + float(RNG.normal(0, 0.004))
                rows.append({
                    'dataset': dataset,
                    'algorithm': algo,
                    'slice_name': slice_name,
                    'topk': 10,
                    'mode': 'exc',
                    'ndcg': round(ndcg, 4),
                    'recall': round(recall, 4),
                })
    return pd.DataFrame(rows)


def make_feature_importance() -> pd.DataFrame:
    return pd.DataFrame([
        {'algorithm': 'LightGBM_Hybrid', 'feature': 'recency_days', 'importance': 186},
        {'algorithm': 'LightGBM_Hybrid', 'feature': 'frequency_30d', 'importance': 174},
        {'algorithm': 'LightGBM_Hybrid', 'feature': 'monetary_total', 'importance': 167},
        {'algorithm': 'LightGBM_Hybrid', 'feature': 'als_score', 'importance': 161},
        {'algorithm': 'LightGBM_Hybrid', 'feature': 'ease_score', 'importance': 152},
        {'algorithm': 'LightGBM_Hybrid', 'feature': 'popularity_rank', 'importance': 130},
        {'algorithm': 'LightGBM_Hybrid', 'feature': 'hour_of_day', 'importance': 86},
        {'algorithm': 'LightGBM_Hybrid', 'feature': 'weekday', 'importance': 73},
    ])


def make_efficiency() -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        scale = {'MovieLens 20M': 1.0, 'Instacart': 1.12, 'Online Retail': 0.35}[dataset]
        for algo in ALGORITHMS:
            rows.append({
                'dataset': dataset,
                'algorithm': algo,
                'train_seconds': round({
                    'Popularity': 5.6,
                    'ALS': 252.0,
                    'RBM': 728.0,
                    'NCF': 860.0,
                    'SASRec': 1190.0,
                    'EASE': 40.0,
                    'GRU4Rec': 940.0,
                    'LightGBM_Hybrid': 125.0,
                }[algo] * scale, 2),
                'inference_ms_per_user': round({
                    'Popularity': 0.082,
                    'ALS': 1.10,
                    'RBM': 2.756,
                    'NCF': 2.30,
                    'SASRec': 3.80,
                    'EASE': 0.42,
                    'GRU4Rec': 2.95,
                    'LightGBM_Hybrid': 1.60,
                }[algo], 3),
                'param_millions': {
                    'Popularity': 0.0,
                    'ALS': 3.5,
                    'RBM': 5.4,
                    'NCF': 7.2,
                    'SASRec': 11.8,
                    'EASE': 0.4,
                    'GRU4Rec': 8.9,
                    'LightGBM_Hybrid': 0.7,
                }[algo],
                'memory_mb': {
                    'Popularity': 18,
                    'ALS': 350,
                    'RBM': 520,
                    'NCF': 610,
                    'SASRec': 910,
                    'EASE': 95,
                    'GRU4Rec': 680,
                    'LightGBM_Hybrid': 120,
                }[algo],
            })
    return pd.DataFrame(rows)


def make_monthly_interactions() -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        months = pd.period_range('2020-01', '2020-12', freq='M').astype(str).tolist()
        base = {'MovieLens 20M': 1_650_000, 'Instacart': 2_100_000, 'Online Retail': 30_000}[dataset]
        for i, month in enumerate(months):
            seasonal = 1.0 + 0.08 * np.sin(i / 2.0)
            interactions = int(base * seasonal * (1 + RNG.uniform(-0.03, 0.03)))
            active_users = int((base / 13) * seasonal * (1 + RNG.uniform(-0.02, 0.02)))
            active_items = int((base / 210) * seasonal * (1 + RNG.uniform(-0.02, 0.02)))
            rows.append({
                'dataset': dataset,
                'month': month,
                'interactions': interactions,
                'active_users': active_users,
                'active_items': active_items,
            })
    return pd.DataFrame(rows)


def make_recommendations() -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        for user_num in range(1001, 1007):
            user_id = f'U{user_num}'
            for algo in ['ALS', 'NCF', 'SASRec', 'EASE', 'LightGBM_Hybrid']:
                for rank in range(1, 11):
                    rows.append({
                        'dataset': dataset,
                        'user_id': user_id,
                        'algorithm': algo,
                        'rank': rank,
                        'item_id': f'{dataset[:3].upper()}_{rank + 1000}',
                        'item_name': f'Recommended {dataset} item {rank}',
                        'score': round(float(RNG.uniform(0.15, 0.95)), 4),
                        'seen_before': False,
                        'reason': {
                            'ALS': 'latent collaborative similarity',
                            'NCF': 'nonlinear user-item interaction',
                            'SASRec': 'sequence-aware next-item signal',
                            'EASE': 'strong linear co-occurrence pattern',
                            'LightGBM_Hybrid': 'hybrid ranker with recency and CF features',
                        }[algo],
                    })
    return pd.DataFrame(rows)


def make_user_history() -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        for user_num in range(1001, 1007):
            user_id = f'U{user_num}'
            for rank in range(1, 8):
                rows.append({
                    'dataset': dataset,
                    'user_id': user_id,
                    'interaction_rank': rank,
                    'item_id': f'{dataset[:3].upper()}_{100 + rank}',
                    'item_name': f'{dataset} history item {rank}',
                    'interaction_type': 'history',
                })
    return pd.DataFrame(rows)


def make_activity_distribution() -> pd.DataFrame:
    buckets = ['1', '2-3', '4-5', '6-10', '11-20', '21-50', '51+']
    rows = []
    for dataset in DATASETS:
        base_user = np.array([12000, 18000, 16000, 14000, 8000, 4500, 2200], dtype=float)
        base_item = np.array([9000, 14000, 12000, 9500, 5200, 2600, 950], dtype=float)
        scale = {'MovieLens 20M': 1.0, 'Instacart': 1.3, 'Online Retail': 0.18}[dataset]
        for entity_type, base in [('user', base_user), ('item', base_item)]:
            for bucket, count in zip(buckets, base * scale):
                rows.append({
                    'dataset': dataset,
                    'entity_type': entity_type,
                    'bucket': bucket,
                    'count': int(count * (1 + RNG.uniform(-0.08, 0.08))),
                })
    return pd.DataFrame(rows)


def make_rating_distribution() -> pd.DataFrame:
    rows = []
    for rating, count in zip([1, 2, 3, 4, 5], [620000, 950000, 2100000, 5600000, 10700000]):
        rows.append({'dataset': 'MovieLens 20M', 'rating': rating, 'count': count})
    for rating, count in zip([1, 2, 3, 4, 5], [3000, 8200, 15700, 52200, 128000]):
        rows.append({'dataset': 'Online Retail', 'rating': rating, 'count': count})
    return pd.DataFrame(rows)


def make_long_tail_curve() -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        alpha = {'MovieLens 20M': 0.33, 'Instacart': 0.28, 'Online Retail': 0.40}[dataset]
        for pct in [1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
            interaction_share = 100 * ((pct / 100) ** alpha)
            rows.append({
                'dataset': dataset,
                'item_share_pct': pct,
                'interaction_share_pct': round(interaction_share, 2),
            })
    return pd.DataFrame(rows)


def main() -> None:
    out_dir = ensure_directory(Path(__file__).resolve().parent / 'sample_artifacts')

    results = make_results()
    export_artifacts(
        out_dir,
        datasets_summary=make_datasets_summary(),
        results=results,
        cannibalization=make_cannibalization(results),
        rfm_metrics=make_rfm_metrics(),
        revenue_mix=make_revenue_mix(),
        slice_metrics=make_slice_metrics(),
        feature_importance=make_feature_importance(),
        efficiency=make_efficiency(),
        monthly_interactions=make_monthly_interactions(),
        recommendations=make_recommendations(),
        user_history=make_user_history(),
        activity_distribution=make_activity_distribution(),
        rating_distribution=make_rating_distribution(),
        long_tail_curve=make_long_tail_curve(),
    )
    print(f'Demo artifacts written to {out_dir}')


if __name__ == '__main__':
    main()
