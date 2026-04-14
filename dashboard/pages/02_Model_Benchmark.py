from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard_core import (
    DISPLAY_NAME,
    METRIC_HELP,
    benchmark_summary,
    available_metrics,
    best_model_by_dataset,
    build_leaderboard,
    metric_card_value,
    page_intro,
    pivot_for_heatmap,
    safely_round_df,
    sidebar_controls,
)

payload, artifact_dir = sidebar_controls()
results = payload['results']
efficiency = payload['efficiency']
metric_intervals = payload['metric_intervals']
significance_tests = payload['significance_tests']

page_intro('Model Benchmark', 'Главная страница сравнения моделей: leaderboard, heatmap, top-K dynamics и quality vs cost.')
st.caption(benchmark_summary())

if results.empty:
    st.error('results.csv не найден.')
    st.stop()

all_datasets = sorted(results['dataset'].dropna().unique().tolist())
all_topk = sorted(int(x) for x in results['topk'].dropna().unique())
metric_options = available_metrics(results)
all_modes = sorted(results['mode'].dropna().unique().tolist())

col_a, col_b, col_c, col_d = st.columns(4)
dataset = col_a.selectbox('Dataset', all_datasets, index=0)
metric = col_b.selectbox('Main metric', metric_options, index=metric_options.index('ndcg') if 'ndcg' in metric_options else 0)
topk = col_c.select_slider('Top-K', options=all_topk, value=10 if 10 in all_topk else all_topk[0])
mode = col_d.selectbox('Mode', all_modes, index=all_modes.index('exc') if 'exc' in all_modes else 0)

st.info(METRIC_HELP.get(metric, 'Metric description is not available.'))

leaderboard = build_leaderboard(results, dataset=dataset, topk=topk, mode=mode, metric=metric)
if leaderboard.empty:
    st.warning('Нет строк под выбранный срез.')
    st.stop()

best_algorithm = leaderboard.iloc[0]['algorithm']
best_value = float(leaderboard.iloc[0][metric])
pop_value = metric_card_value(results, dataset, 'Popularity', metric, topk, mode)

m1, m2, m3, m4 = st.columns(4)
m1.metric(f'Best {DISPLAY_NAME.get(metric, metric).upper()}@{topk}', f'{best_value:.4f}')
m2.metric('Winner', best_algorithm)
m3.metric('Lift vs Popularity', f'{(best_value - pop_value):+.4f}' if pd.notna(pop_value) else 'n/a')
m4.metric('Models compared', f"{leaderboard['algorithm'].nunique()}")

left, right = st.columns([1.05, 0.95])

with left:
    st.markdown('**Leaderboard**')
    st.dataframe(safely_round_df(leaderboard), use_container_width=True, hide_index=True)

    trend_df = results[(results['dataset'] == dataset) & (results['mode'] == mode)].copy()
    fig = px.line(
        trend_df,
        x='topk',
        y=metric,
        color='algorithm',
        markers=True,
        title=f'{DISPLAY_NAME.get(metric, metric).upper()} across top-K',
    )
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)

with right:
    heatmap_df = pivot_for_heatmap(results, topk=topk, mode=mode, metric=metric)
    if not heatmap_df.empty:
        fig_heat = px.imshow(
            heatmap_df,
            text_auto='.3f',
            aspect='auto',
            title=f'{DISPLAY_NAME.get(metric, metric).upper()} heatmap',
            labels={'x': 'Dataset', 'y': 'Algorithm', 'color': DISPLAY_NAME.get(metric, metric).upper()},
        )
        fig_heat.update_layout(height=430)
        st.plotly_chart(fig_heat, use_container_width=True)

winners = best_model_by_dataset(results, metric=metric, topk=topk, mode=mode)
lower_left, lower_right = st.columns([1.0, 1.0])

with lower_left:
    st.markdown('**Best model per dataset**')
    st.dataframe(safely_round_df(winners), use_container_width=True, hide_index=True)

with lower_right:
    compare_df = results[(results['dataset'] == dataset) & (results['topk'] == topk) & (results['mode'] == mode)].copy()
    compare_df = compare_df[compare_df[metric].notna()].copy()
    metric_cols = [c for c in ['ndcg', 'recall', 'precision', 'map', 'coverage', 'novelty'] if c in compare_df.columns]
    radar_algorithms = st.multiselect(
        'Algorithms for multimetric comparison',
        options=compare_df['algorithm'].unique().tolist(),
        default=compare_df.sort_values(metric, ascending=False)['algorithm'].head(4).tolist(),
    )
    if radar_algorithms:
        radar_df = compare_df[compare_df['algorithm'].isin(radar_algorithms)]
        fig_radar = go.Figure()
        for algo in radar_algorithms:
            row = radar_df[radar_df['algorithm'] == algo].iloc[0]
            values = [row[m] for m in metric_cols]
            fig_radar.add_trace(go.Scatterpolar(r=values, theta=metric_cols, fill='toself', name=algo))
        fig_radar.update_layout(title='Multi-metric shape by algorithm', polar=dict(radialaxis=dict(visible=True)), height=480)
        st.plotly_chart(fig_radar, use_container_width=True)

if not efficiency.empty:
    st.markdown('**Quality vs training cost**')
    merged = leaderboard.merge(efficiency[efficiency['dataset'] == dataset], on='algorithm', how='left')
    if 'train_seconds' in merged.columns and metric in merged.columns:
        fig_eff = px.scatter(
            merged,
            x='train_seconds',
            y=metric,
            size='memory_mb' if 'memory_mb' in merged.columns else None,
            hover_name='algorithm',
            title='Training cost against quality',
            log_x=True,
        )
        fig_eff.update_layout(height=400)
        st.plotly_chart(fig_eff, use_container_width=True)

st.markdown('**Confidence intervals and significance**')
interval_view = metric_intervals[
    (metric_intervals['dataset'] == dataset) & (metric_intervals['metric'] == metric)
].copy() if not metric_intervals.empty else pd.DataFrame()
if not interval_view.empty:
    interval_view = interval_view.sort_values('mean', ascending=False)
    st.dataframe(safely_round_df(interval_view), use_container_width=True, hide_index=True)

sig_view = significance_tests[
    (significance_tests['dataset'] == dataset) & (significance_tests['metric'] == metric)
].copy() if not significance_tests.empty else pd.DataFrame()
if not sig_view.empty:
    sig_view = sig_view.sort_values(['significant', 'delta_mean'], ascending=[False, False])
    st.dataframe(safely_round_df(sig_view), use_container_width=True, hide_index=True)
elif interval_view.empty:
    st.info('significance_tests.csv пока не выгружен.')
