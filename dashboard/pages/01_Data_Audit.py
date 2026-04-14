from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import plotly.express as px
import streamlit as st

from dashboard_core import page_intro, safely_round_df, sidebar_controls

payload, artifact_dir = sidebar_controls()

datasets_summary = payload['datasets_summary']
monthly_interactions = payload['monthly_interactions']
activity_distribution = payload['activity_distribution']
rating_distribution = payload['rating_distribution']
long_tail_curve = payload['long_tail_curve']

page_intro('Data Audit', 'Сильный старт для ВКР: показать, что данные очищены, сравнимы и имеют понятную структуру.')

if datasets_summary.empty:
    st.error('datasets_summary.csv не найден.')
    st.stop()

dataset = st.selectbox('Dataset', sorted(datasets_summary['dataset'].unique().tolist()), index=0)

current = datasets_summary[datasets_summary['dataset'] == dataset]
if not current.empty:
    row = current.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Users', f"{int(row['users']):,}")
    c2.metric('Items', f"{int(row['items']):,}")
    c3.metric('Events', f"{int(row['events']):,}")
    if 'sparsity_pct' in current.columns:
        c4.metric('Sparsity %', f"{float(row['sparsity_pct']):.2f}")

left, right = st.columns([1.05, 0.95])

with left:
    st.markdown('**Cross-dataset summary**')
    st.dataframe(safely_round_df(datasets_summary), use_container_width=True, hide_index=True)

    if not monthly_interactions.empty:
        month_df = monthly_interactions[monthly_interactions['dataset'] == dataset].copy()
        if not month_df.empty:
            fig = px.line(
                month_df,
                x='month',
                y=['interactions', 'active_users', 'active_items'],
                markers=True,
                title=f'{dataset}: volume and activity over time',
            )
            fig.update_layout(height=420)
            st.plotly_chart(fig, use_container_width=True)

with right:
    if not activity_distribution.empty:
        dist_df = activity_distribution[activity_distribution['dataset'] == dataset].copy()
        if not dist_df.empty:
            fig_user = px.bar(
                dist_df,
                x='bucket',
                y='count',
                color='entity_type',
                barmode='group',
                title='User / item activity distribution',
            )
            fig_user.update_layout(height=420)
            st.plotly_chart(fig_user, use_container_width=True)

if not long_tail_curve.empty:
    st.markdown('**Long-tail diagnostic**')
    lt_df = long_tail_curve[long_tail_curve['dataset'] == dataset].copy()
    if not lt_df.empty:
        fig_lt = px.line(
            lt_df,
            x='item_share_pct',
            y='interaction_share_pct',
            markers=True,
            title='How concentrated interactions are in the item head',
        )
        fig_lt.update_layout(height=420, xaxis_title='Top item share, %', yaxis_title='Captured interaction share, %')
        st.plotly_chart(fig_lt, use_container_width=True)

if not rating_distribution.empty:
    st.markdown('**Explicit target distribution**')
    rd_df = rating_distribution[rating_distribution['dataset'] == dataset].copy()
    if not rd_df.empty:
        fig_rd = px.bar(
            rd_df,
            x='rating',
            y='count',
            title='Rating / target distribution',
        )
        fig_rd.update_layout(height=360)
        st.plotly_chart(fig_rd, use_container_width=True)
    else:
        st.info('Для выбранного датасета explicit rating distribution не подготовлен.')
