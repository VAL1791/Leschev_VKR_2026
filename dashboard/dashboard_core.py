from __future__ import annotations

from pathlib import Path
from typing import Dict
import sys

import numpy as np
import pandas as pd
try:
    import streamlit as st
except ModuleNotFoundError:  # pragma: no cover - allows artifact generation without Streamlit
    st = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data_contract import (
    ARTIFACT_FILENAMES,
    COMPLETED_RUN_STATUSES,
    MINIMUM_REQUIRED_ARTIFACTS,
    OPTIONAL_ARTIFACTS,
    PRIMARY_BENCHMARK,
    filter_completed_results,
)


ARTIFACT_FILES = ARTIFACT_FILENAMES

STARTER_FILES = {
    'workstreams': 'workstreams.csv',
    'research_questions': 'research_questions.csv',
    'model_registry': 'model_registry.csv',
    'experiment_backlog': 'experiment_backlog.csv',
    'notebook_outline': 'notebook_outline.csv',
    'two_week_roadmap': 'two_week_roadmap.csv',
}

METRIC_HELP = {
    'ndcg': 'NDCG@K учитывает и релевантность, и порядок позиций. Это главная ranking-метрика для твоей ВКР.',
    'recall': 'Recall@K показывает, сколько релевантных товаров модель вообще смогла найти в top-K.',
    'precision': 'Precision@K показывает долю релевантных товаров внутри top-K.',
    'map': 'MAP@K усредняет precision по позициям, где встретились релевантные объекты.',
    'coverage': 'Coverage@K показывает, насколько широко модель покрывает каталог, а не только head-товары.',
    'novelty': 'Novelty@K помогает показать, что модель выдает не только самые популярные товары.',
}

DISPLAY_NAME = {
    'ndcg': 'NDCG',
    'recall': 'Recall',
    'precision': 'Precision',
    'map': 'MAP',
    'coverage': 'Coverage',
    'novelty': 'Novelty',
}


def artifact_root() -> Path:
    return Path(__file__).resolve().parent


def project_root() -> Path:
    return PROJECT_ROOT


def starter_kit_root() -> Path:
    return project_root() / 'starter_kit'


def _read_csv(path: Path) -> pd.DataFrame:
    if path.exists() and path.is_file() and path.stat().st_size > 0:
        return pd.read_csv(path)
    return pd.DataFrame()


def _existing_csv_count(folder: Path) -> int:
    if not folder.exists():
        return 0
    return len(list(folder.glob('*.csv')))


def load_artifacts(base_dir: Path) -> Dict[str, pd.DataFrame]:
    base_dir = Path(base_dir)
    payload = {name: _read_csv(base_dir / filename) for name, filename in ARTIFACT_FILES.items()}

    results_all = payload['results']
    if not results_all.empty:
        numeric_cols = [
            'topk', 'precision', 'recall', 'map', 'ndcg', 'coverage', 'novelty',
            'train_seconds', 'inference_ms_per_user'
        ]
        for col in numeric_cols:
            if col in results_all.columns:
                results_all[col] = pd.to_numeric(results_all[col], errors='coerce')
        payload['results_all'] = results_all
        # Keep the full table for robustness pages, but default benchmark views to final
        # rows only so planned/ablation entries do not distort the main leaderboard.
        payload['results'] = filter_completed_results(results_all)
    else:
        payload['results_all'] = results_all

    for logical_name in ['efficiency', 'datasets_summary', 'monthly_interactions', 'long_tail_curve', 'metric_intervals', 'significance_tests']:
        frame = payload.get(logical_name)
        if frame is not None and not frame.empty:
            for col in frame.columns:
                if col.endswith('_pct') or col in ['users', 'items', 'events', 'interactions', 'active_users', 'active_items', 'train_seconds', 'memory_mb', 'mean', 'ci_low', 'ci_high', 'delta_mean', 'p_value', 'p_value_adj', 'n_users', 'inference_ms_per_user']:
                    converted = pd.to_numeric(frame[col], errors='coerce')
                    if converted.notna().any():
                        frame[col] = converted
            payload[logical_name] = frame

    return payload


def load_starter_tables() -> Dict[str, pd.DataFrame]:
    base = starter_kit_root()
    return {name: _read_csv(base / filename) for name, filename in STARTER_FILES.items()}


def artifact_readiness(payload: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for logical_name in MINIMUM_REQUIRED_ARTIFACTS + OPTIONAL_ARTIFACTS:
        frame = payload.get(logical_name, pd.DataFrame())
        rows.append(
            {
                'artifact': logical_name,
                'required': logical_name in MINIMUM_REQUIRED_ARTIFACTS,
                'status': 'ready' if not frame.empty else 'missing',
                'rows': int(len(frame)) if not frame.empty else 0,
            }
        )
    return pd.DataFrame(rows)


def benchmark_summary() -> str:
    secondary = ", ".join(metric.upper() for metric in PRIMARY_BENCHMARK['secondary_metrics'])
    return (
        f"Primary benchmark: {PRIMARY_BENCHMARK['primary_metric'].upper()}@{PRIMARY_BENCHMARK['topk']} "
        f"in {PRIMARY_BENCHMARK['mode']} mode, temporal split {PRIMARY_BENCHMARK['temporal_split']}. "
        f"Supporting metrics: {secondary}."
    )


def sidebar_controls() -> tuple[dict[str, pd.DataFrame], Path]:
    if st is None:
        raise RuntimeError('Streamlit is required for sidebar controls.')
    root = artifact_root()
    real_dir = root / 'artifacts'
    demo_dir = root / 'sample_artifacts'

    if 'use_demo_artifacts' not in st.session_state:
        st.session_state.use_demo_artifacts = _existing_csv_count(real_dir) == 0

    st.sidebar.title('Artifact source')
    st.session_state.use_demo_artifacts = st.sidebar.toggle(
        'Use demo artifacts',
        value=st.session_state.use_demo_artifacts,
        help='Если реальные выгрузки из notebook пока не готовы, оставь demo-режим включенным.',
    )
    default_dir = demo_dir if st.session_state.use_demo_artifacts else real_dir

    current_dir = st.session_state.get('artifact_dir', str(default_dir))
    artifact_dir = Path(st.sidebar.text_input('Artifacts folder', current_dir))
    st.session_state.artifact_dir = str(artifact_dir)

    st.sidebar.caption(f'Project root: {project_root()}')
    payload = load_artifacts(artifact_dir)
    readiness = artifact_readiness(payload)
    required_ready = int((readiness['required'] & (readiness['status'] == 'ready')).sum())
    optional_ready = int((~readiness['required'] & (readiness['status'] == 'ready')).sum())
    st.sidebar.caption(
        f"Required artifacts: {required_ready}/{len(MINIMUM_REQUIRED_ARTIFACTS)} | "
        f"Optional artifacts: {optional_ready}/{len(OPTIONAL_ARTIFACTS)}"
    )
    st.sidebar.caption(benchmark_summary())
    results_all = payload.get('results_all', pd.DataFrame())
    if not results_all.empty and 'run_status' in results_all.columns:
        statuses = results_all['run_status'].fillna('done').astype(str).str.strip().str.lower()
        inactive = int((~statuses.isin(COMPLETED_RUN_STATUSES)).sum())
        if inactive:
            st.sidebar.caption(f"Hidden non-final result rows: {inactive}")
    return payload, artifact_dir


def available_metrics(results: pd.DataFrame) -> list[str]:
    base = ['ndcg', 'recall', 'precision', 'map', 'coverage', 'novelty']
    if results.empty:
        return []
    return [col for col in base if col in results.columns and results[col].notna().any()]


def build_leaderboard(results: pd.DataFrame, dataset: str, topk: int, mode: str, metric: str) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    frame = results.copy()
    frame = frame[(frame['dataset'] == dataset) & (frame['topk'] == topk) & (frame['mode'] == mode)]
    frame = frame[frame[metric].notna()].copy()
    cols = [c for c in ['algorithm', metric, 'coverage', 'novelty', 'train_seconds', 'inference_ms_per_user'] if c in frame.columns]
    frame = frame[cols].sort_values(metric, ascending=False).reset_index(drop=True)
    if metric in frame.columns:
        frame['rank'] = np.arange(1, len(frame) + 1)
        frame = frame[['rank'] + [c for c in frame.columns if c != 'rank']]
    return frame


def best_model_by_dataset(results: pd.DataFrame, metric: str, topk: int, mode: str) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    frame = results[(results['topk'] == topk) & (results['mode'] == mode)].copy()
    frame = frame[frame[metric].notna()].copy()
    if frame.empty:
        return frame
    idx = frame.groupby('dataset')[metric].idxmax()
    return frame.loc[idx, ['dataset', 'algorithm', metric]].sort_values('dataset').reset_index(drop=True)


def benchmark_takeaways(
    results: pd.DataFrame,
    feature_importance: pd.DataFrame | None = None,
    topk: int | None = None,
    mode: str | None = None,
    metric: str = 'ndcg',
) -> list[str]:
    if results.empty:
        return []
    topk = PRIMARY_BENCHMARK['topk'] if topk is None else topk
    mode = PRIMARY_BENCHMARK['mode'] if mode is None else mode
    frame = results[(results['topk'] == topk) & (results['mode'] == mode)].copy()
    frame = frame[frame[metric].notna()].copy()
    if frame.empty:
        return []

    lines: list[str] = []
    winners = best_model_by_dataset(frame, metric=metric, topk=topk, mode=mode)
    for row in winners.itertuples(index=False):
        lines.append(f"{row.dataset}: лучший результат даёт {row.algorithm} ({getattr(row, metric):.4f} {metric.upper()}@{topk}).")

    # These takeaways are intentionally written as defense-ready claims that summarize
    # what the metric tables mean for each domain, not just who won numerically.
    for dataset in sorted(frame['dataset'].unique().tolist()):
        dataset_rows = frame[frame['dataset'] == dataset].set_index('algorithm')
        if {'HybridFeatureRerank', 'EASE'}.issubset(dataset_rows.index):
            delta = float(dataset_rows.at['HybridFeatureRerank', metric] - dataset_rows.at['EASE', metric])
            if delta > 0.01:
                lines.append(f"{dataset}: hybrid уверенно обходит чистый CF, значит доменные признаки и rerank реально добавляют сигнал.")
            elif delta < -0.01:
                lines.append(f"{dataset}: EASE сильнее hybrid, что типично для более плотного collaborative-сценария.")
        if {'HybridFeatureRerank', 'Hybrid_NoContentAblation'}.issubset(dataset_rows.index):
            delta = float(dataset_rows.at['HybridFeatureRerank', metric] - dataset_rows.at['Hybrid_NoContentAblation', metric])
            if delta > 0.005:
                lines.append(f"{dataset}: удаление content/profile signals заметно просаживает hybrid, значит feature engineering реально работает.")
        if {'HybridFeatureRerank', 'Hybrid_NoCollaborativeAblation'}.issubset(dataset_rows.index):
            delta = float(dataset_rows.at['HybridFeatureRerank', metric] - dataset_rows.at['Hybrid_NoCollaborativeAblation', metric])
            if delta > 0.005:
                lines.append(f"{dataset}: без collaborative signals hybrid заметно теряет качество, поэтому rerank нельзя строить только на metadata.")
        if {'TFIDF_Content', 'Popularity'}.issubset(dataset_rows.index):
            delta = float(dataset_rows.at['TFIDF_Content', metric] - dataset_rows.at['Popularity', metric])
            if delta > 0.03:
                lines.append(f"{dataset}: content-aware retrieval заметно сильнее popularity, текстовые и metadata-признаки здесь оправданы.")

    if feature_importance is not None and not feature_importance.empty:
        feat_frame = feature_importance.copy()
        if 'algorithm' in feat_frame.columns:
            feat_frame = feat_frame[feat_frame['algorithm'] == 'HybridFeatureRerank'].copy()
        if 'dataset' in feat_frame.columns:
            for dataset in sorted(feat_frame['dataset'].dropna().astype(str).unique().tolist()):
                top_features = feat_frame[feat_frame['dataset'] == dataset].sort_values('importance', ascending=False)['feature'].head(2).tolist()
                if top_features:
                    labels = ", ".join(top_features)
                    lines.append(f"{dataset}: в hybrid-слое главные драйверы качества сейчас {labels}.")
        else:
            top_features = feat_frame.sort_values('importance', ascending=False)['feature'].head(3).tolist()
            if top_features:
                lines.append(f"Главные признаки hybrid-модели: {', '.join(top_features)}.")
    return lines


def ablation_summary(
    results: pd.DataFrame,
    topk: int | None = None,
    mode: str | None = None,
    metric: str = 'ndcg',
) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    topk = PRIMARY_BENCHMARK['topk'] if topk is None else topk
    mode = PRIMARY_BENCHMARK['mode'] if mode is None else mode
    frame = results[(results['topk'] == topk) & (results['mode'] == mode)].copy()
    frame = frame[frame['algorithm'].isin(['HybridFeatureRerank', 'Hybrid_NoContentAblation', 'Hybrid_NoCollaborativeAblation'])]
    frame = frame[frame[metric].notna()].copy()
    if frame.empty:
        return pd.DataFrame()
    pivot = frame.pivot(index='dataset', columns='algorithm', values=metric).reset_index()
    for column in ['HybridFeatureRerank', 'Hybrid_NoContentAblation', 'Hybrid_NoCollaborativeAblation']:
        if column not in pivot.columns:
            pivot[column] = np.nan
    pivot['delta_vs_no_content'] = pivot['HybridFeatureRerank'] - pivot['Hybrid_NoContentAblation']
    pivot['delta_vs_no_collab'] = pivot['HybridFeatureRerank'] - pivot['Hybrid_NoCollaborativeAblation']
    return pivot


def ablation_takeaways(
    results: pd.DataFrame,
    topk: int | None = None,
    mode: str | None = None,
    metric: str = 'ndcg',
) -> list[str]:
    summary = ablation_summary(results, topk=topk, mode=mode, metric=metric)
    if summary.empty:
        return []
    lines: list[str] = []
    for row in summary.itertuples(index=False):
        content_delta = getattr(row, 'delta_vs_no_content')
        collab_delta = getattr(row, 'delta_vs_no_collab')
        if pd.notna(collab_delta) and collab_delta > 0.02:
            lines.append(f"{row.dataset}: collaborative signals критичны; без них hybrid резко деградирует.")
        elif pd.notna(collab_delta) and collab_delta < -0.005:
            lines.append(f"{row.dataset}: content-heavy ablation неожиданно сильнее полного hybrid, это стоит трактовать как сигнал к перенастройке rerank weights.")
        if pd.notna(content_delta) and content_delta > 0.02:
            lines.append(f"{row.dataset}: content/profile features дают заметный вклад в итоговый ranking.")
        elif pd.notna(content_delta) and abs(content_delta) <= 0.005:
            lines.append(f"{row.dataset}: content/profile signals почти не меняют итоговый ranking, домен ближе к pure collaborative case.")
    return lines


def significance_takeaways(significance_tests: pd.DataFrame, metric: str = 'ndcg') -> list[str]:
    if significance_tests.empty:
        return []
    frame = significance_tests[significance_tests['metric'] == metric].copy()
    frame = frame.sort_values(['dataset', 'reference_algorithm', 'comparison_algorithm'])
    lines: list[str] = []
    for dataset in sorted(frame['dataset'].dropna().astype(str).unique().tolist()):
        dataset_slice = frame[frame['dataset'] == dataset]
        for reference_algorithm, comparison_algorithm in [
            ('HybridFeatureRerank', 'EASE'),
            ('HybridFeatureRerank', 'SequentialTransition'),
            ('HybridFeatureRerank', 'Hybrid_NoContentAblation'),
            ('HybridFeatureRerank', 'Hybrid_NoCollaborativeAblation'),
        ]:
            row = dataset_slice[
                (dataset_slice['reference_algorithm'] == reference_algorithm)
                & (dataset_slice['comparison_algorithm'] == comparison_algorithm)
            ]
            if row.empty:
                continue
            record = row.iloc[0]
            delta = float(record['delta_mean'])
            ci_low = float(record['ci_low'])
            ci_high = float(record['ci_high'])
            p_adj = float(record['p_value_adj']) if pd.notna(record['p_value_adj']) else float('nan')
            significant = bool(record['significant'])
            if significant:
                direction = 'выше' if delta > 0 else 'ниже'
                lines.append(
                    f"{dataset}: {reference_algorithm} статистически {direction} {comparison_algorithm} "
                    f"по {metric.upper()}@10 (delta {delta:+.4f}, 95% CI [{ci_low:+.4f}, {ci_high:+.4f}], p_adj={p_adj:.4f})."
                )
            else:
                lines.append(
                    f"{dataset}: разница между {reference_algorithm} и {comparison_algorithm} по {metric.upper()}@10 "
                    f"не подтверждена статистически (delta {delta:+.4f}, 95% CI [{ci_low:+.4f}, {ci_high:+.4f}], p_adj={p_adj:.4f})."
                )
    return lines


def pivot_for_heatmap(results: pd.DataFrame, topk: int, mode: str, metric: str) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    frame = results[(results['topk'] == topk) & (results['mode'] == mode)].copy()
    frame = frame[frame[metric].notna()].copy()
    if frame.empty:
        return pd.DataFrame()
    return frame.pivot(index='algorithm', columns='dataset', values=metric).sort_index()


def metric_card_value(results: pd.DataFrame, dataset: str, algorithm: str, metric: str, topk: int, mode: str) -> float:
    if results.empty:
        return float('nan')
    frame = results[
        (results['dataset'] == dataset)
        & (results['algorithm'] == algorithm)
        & (results['topk'] == topk)
        & (results['mode'] == mode)
    ]
    if frame.empty:
        return float('nan')
    return float(frame.iloc[0][metric])


def safely_round_df(frame: pd.DataFrame, digits: int = 4) -> pd.DataFrame:
    out = frame.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].round(digits)
    return out


def ensure_directory(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def export_artifacts(output_dir: Path, **frames: pd.DataFrame) -> None:
    output_dir = ensure_directory(output_dir)
    for logical_name, filename in ARTIFACT_FILES.items():
        frame = frames.get(logical_name)
        if frame is not None and not frame.empty:
            frame.to_csv(output_dir / filename, index=False)


def page_intro(title: str, description: str) -> None:
    if st is None:
        raise RuntimeError('Streamlit is required for page rendering.')
    st.title(title)
    st.caption(description)
    st.markdown('---')
