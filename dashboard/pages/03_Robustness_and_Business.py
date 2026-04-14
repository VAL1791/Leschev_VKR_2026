from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import plotly.express as px
import streamlit as st

from dashboard_core import DISPLAY_NAME, ablation_summary, ablation_takeaways, page_intro, safely_round_df, sidebar_controls, significance_takeaways

payload, artifact_dir = sidebar_controls()
results = payload.get('results_all', payload['results'])
cannibalization = payload['cannibalization']
rfm_metrics = payload['rfm_metrics']
revenue_mix = payload['revenue_mix']
slice_metrics = payload['slice_metrics']
feature_importance = payload['feature_importance']
significance_tests = payload['significance_tests']

page_intro('Robustness & Business', 'То самое мясо сверх обычного leaderboard: inc./exc., RFM, long-tail/cold-start и explainability.')

if results.empty:
    st.error('results.csv не найден.')
    st.stop()

datasets = sorted(results['dataset'].dropna().unique().tolist())
metric_options = [m for m in ['ndcg', 'recall', 'precision', 'map'] if m in results.columns]
topk_options = sorted(int(x) for x in results['topk'].dropna().unique())
mode_options = sorted(results['mode'].dropna().unique().tolist())

c1, c2, c3, c4 = st.columns(4)
dataset = c1.selectbox('Dataset', datasets, index=0)
metric = c2.selectbox('Metric', metric_options, index=0)
topk = c3.selectbox('Top-K', topk_options, index=topk_options.index(10) if 10 in topk_options else 0)
mode = c4.selectbox('Mode', mode_options, index=mode_options.index('exc') if 'exc' in mode_options else 0)

upper_left, upper_right = st.columns([1.0, 1.0])

with upper_left:
    st.markdown('**inc. vs exc. gap**')
    can_df = cannibalization[(cannibalization['dataset'] == dataset) & (cannibalization['topk'] == topk)].copy()
    metric_gap = f'{metric}_gap'
    if not can_df.empty and metric_gap in can_df.columns:
        fig_gap = px.bar(
            can_df.sort_values(metric_gap, ascending=False),
            x='algorithm',
            y=metric_gap,
            title=f'{DISPLAY_NAME.get(metric, metric).upper()} gap: inc - exc',
        )
        fig_gap.update_layout(height=420)
        st.plotly_chart(fig_gap, use_container_width=True)
    else:
        st.info('Для выбранного среза gap table не готова.')

with upper_right:
    st.markdown('**Cold-start / long-tail slices**')
    slice_df = slice_metrics[(slice_metrics['dataset'] == dataset) & (slice_metrics['topk'] == topk) & (slice_metrics['mode'] == mode)].copy()
    if not slice_df.empty:
        fig_slice = px.bar(
            slice_df,
            x='slice_name',
            y=metric if metric in slice_df.columns else 'ndcg',
            color='algorithm',
            barmode='group',
            title='Robustness across slices',
        )
        fig_slice.update_layout(height=420)
        st.plotly_chart(fig_slice, use_container_width=True)
    else:
        st.info('slice_metrics.csv отсутствует.')

lower_left, lower_right = st.columns([1.0, 1.0])

with lower_left:
    st.markdown('**RFM × quality**')
    if not rfm_metrics.empty:
        rfm_metric = st.selectbox('RFM metric', [c for c in ['ndcg', 'recall'] if c in rfm_metrics.columns], index=0)
        rfm_view = rfm_metrics[rfm_metrics['dataset'] == dataset].copy() if 'dataset' in rfm_metrics.columns else rfm_metrics.copy()
        rfm_pivot = rfm_view.pivot(index='algorithm', columns='rfm_segment', values=rfm_metric)
        fig_rfm = px.imshow(
            rfm_pivot,
            text_auto='.3f',
            aspect='auto',
            title=f'RFM segment heatmap ({rfm_metric})',
        )
        fig_rfm.update_layout(height=460)
        st.plotly_chart(fig_rfm, use_container_width=True)
    else:
        st.info('rfm_metrics.csv отсутствует.')

with lower_right:
    st.markdown('**Revenue mix**')
    if not revenue_mix.empty:
        mix_view = revenue_mix[revenue_mix['dataset'] == dataset].copy() if 'dataset' in revenue_mix.columns else revenue_mix.copy()
        mix_pivot = mix_view.pivot(index='proxy_category', columns='RFM_score', values='share')
        fig_mix = px.imshow(mix_pivot, text_auto='.1f', aspect='auto', title='Revenue share heatmap')
        fig_mix.update_layout(height=460)
        st.plotly_chart(fig_mix, use_container_width=True)
    else:
        st.info('revenue_mix.csv отсутствует.')

if not feature_importance.empty:
    st.markdown('**Hybrid model explainability**')
    feat_df = feature_importance[feature_importance['dataset'] == dataset].copy() if 'dataset' in feature_importance.columns else feature_importance.copy()
    algo_options = sorted(feat_df['algorithm'].dropna().unique().tolist()) if 'algorithm' in feat_df.columns else []
    algo_choice = st.selectbox(
        'Explainability algorithm',
        options=algo_options,
        index=algo_options.index('HybridFeatureRerank') if 'HybridFeatureRerank' in algo_options else 0,
    ) if algo_options else None
    if algo_choice:
        feat_df = feat_df[feat_df['algorithm'] == algo_choice].copy()
    feat_df = feat_df.sort_values('importance', ascending=True)
    fig_feat = px.bar(
        feat_df,
        x='importance',
        y='feature',
        orientation='h',
        title=f'{algo_choice or "Hybrid"} feature weights',
    )
    fig_feat.update_layout(height=420)
    st.plotly_chart(fig_feat, use_container_width=True)

st.markdown('**Hybrid ablation**')
ablation_df = ablation_summary(results, topk=topk, mode=mode, metric=metric)
if not ablation_df.empty:
    ablation_view = ablation_df[ablation_df['dataset'] == dataset].copy()
    if not ablation_view.empty:
        plot_df = ablation_view.melt(
            id_vars='dataset',
            value_vars=['HybridFeatureRerank', 'Hybrid_NoContentAblation', 'Hybrid_NoCollaborativeAblation'],
            var_name='algorithm',
            value_name=metric,
        )
        fig_ablation = px.bar(
            plot_df,
            x='algorithm',
            y=metric,
            title=f'{DISPLAY_NAME.get(metric, metric).upper()} for hybrid ablation variants',
            text_auto='.3f',
        )
        fig_ablation.update_layout(height=420)
        st.plotly_chart(fig_ablation, use_container_width=True)
        st.dataframe(safely_round_df(ablation_view), use_container_width=True, hide_index=True)
    insights = [line for line in ablation_takeaways(results, topk=topk, mode=mode, metric=metric) if line.startswith(dataset)]
    if insights:
        st.caption(' '.join(insights))
else:
    st.info('Ablation rows пока не выгружены.')

st.markdown('**Significance reading**')
if not significance_tests.empty:
    sig_view = significance_tests[
        (significance_tests['dataset'] == dataset) & (significance_tests['metric'] == metric)
    ].copy()
    if not sig_view.empty:
        st.dataframe(safely_round_df(sig_view), use_container_width=True, hide_index=True)
    insight_lines = [line for line in significance_takeaways(significance_tests, metric=metric) if line.startswith(dataset)]
    if insight_lines:
        st.caption(' '.join(insight_lines))
else:
    st.info('significance_tests.csv отсутствует.')
