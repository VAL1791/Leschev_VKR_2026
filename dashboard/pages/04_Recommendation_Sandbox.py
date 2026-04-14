from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import streamlit as st

from dashboard_core import page_intro, sidebar_controls

payload, artifact_dir = sidebar_controls()
recommendations = payload['recommendations']
user_history = payload['user_history']

page_intro('Recommendation Sandbox', 'Человеческий слой поверх таблиц: показать один пример истории пользователя и выдачи разных моделей.')

if recommendations.empty:
    st.info('recommendations.csv отсутствует — sandbox скрыт.')
    st.stop()

sb_left, sb_right = st.columns([0.33, 0.67])

with sb_left:
    sb_dataset = st.selectbox('Dataset', sorted(recommendations['dataset'].unique().tolist()), index=0)
    sb_user = st.selectbox(
        'User',
        sorted(recommendations[recommendations['dataset'] == sb_dataset]['user_id'].unique().tolist()),
    )
    sb_algorithms = st.multiselect(
        'Algorithms',
        options=sorted(recommendations[recommendations['dataset'] == sb_dataset]['algorithm'].unique().tolist()),
        default=[
            algo
            for algo in ['HybridFeatureRerank', 'EASE', 'TFIDF_Content']
            if algo in recommendations[recommendations['dataset'] == sb_dataset]['algorithm'].unique().tolist()
        ] or sorted(recommendations[recommendations['dataset'] == sb_dataset]['algorithm'].unique().tolist())[:3],
    )

    if not user_history.empty:
        st.markdown('**User history**')
        hist = user_history[(user_history['dataset'] == sb_dataset) & (user_history['user_id'] == sb_user)].copy()
        if not hist.empty:
            st.dataframe(hist[['interaction_rank', 'item_id', 'item_name']], use_container_width=True, hide_index=True)

with sb_right:
    view = recommendations[
        (recommendations['dataset'] == sb_dataset)
        & (recommendations['user_id'] == sb_user)
        & (recommendations['algorithm'].isin(sb_algorithms))
    ].copy()
    if view.empty:
        st.warning('Для выбранной комбинации нет рекомендаций.')
    else:
        display_cols = ['algorithm', 'rank', 'item_id', 'item_name', 'score', 'reason']
        for col in ['main_driver', 'matched_tags', 'proxy_category', 'explanation']:
            if col in view.columns:
                display_cols.append(col)
        st.dataframe(
            view[display_cols].sort_values(['algorithm', 'rank']),
            use_container_width=True,
            hide_index=True,
        )

st.caption('Этот блок нужен не для оценки качества, а чтобы объяснить рекомендацию человеческим языком.')
