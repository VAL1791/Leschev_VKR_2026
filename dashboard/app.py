from __future__ import annotations

import streamlit as st

import pandas as pd

from dashboard_core import artifact_readiness, benchmark_summary, benchmark_takeaways, load_starter_tables, page_intro, sidebar_controls


st.set_page_config(
    page_title='Recommender Thesis Dashboard',
    page_icon='🧠',
    layout='wide',
    initial_sidebar_state='expanded',
)

st.markdown(
    """
    <style>
        .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
        .stMetric {background: rgba(17,24,39,0.03); padding: 0.6rem; border-radius: 12px;}
    </style>
    """,
    unsafe_allow_html=True,
)

payload, artifact_dir = sidebar_controls()
starter = load_starter_tables()

workstreams = starter['workstreams']
research_questions = starter['research_questions']
model_registry = starter['model_registry']
experiment_backlog = starter['experiment_backlog']
notebook_outline = starter['notebook_outline']
two_week_roadmap = starter['two_week_roadmap']
results = payload['results']
results_all = payload.get('results_all', pd.DataFrame())
feature_importance = payload['feature_importance']
artifact_status = artifact_readiness(payload)

page_intro(
    'ML & DL for E-commerce Product Recommendations',
    'Каркас ВКР: сильный data analysis, понятный интерфейс и отдельные workstreams, по которым видно, как строится работа.',
)

top_left, top_mid, top_right, top_extra = st.columns(4)
top_left.metric('Starter workstreams', f"{len(workstreams)}")
top_mid.metric('Models in scope', f"{len(model_registry)}")
top_right.metric('Notebook blocks', f"{len(notebook_outline[notebook_outline['type'] == 'section']) if not notebook_outline.empty else 0}")
top_extra.metric('Artifact rows', f"{len(results):,}" if not results.empty else '0')

st.info(
    'Этот dashboard сделан не как интернет-магазин, а как research interface: '
    'сначала project map, потом data audit, затем benchmark, robustness и sandbox.'
)
st.caption(benchmark_summary())

required_ready = int((artifact_status['required'] & (artifact_status['status'] == 'ready')).sum())
optional_ready = int((~artifact_status['required'] & (artifact_status['status'] == 'ready')).sum())
st.caption(
    f"Artifact readiness: required {required_ready}/{int(artifact_status['required'].sum())}, "
    f"optional {optional_ready}/{int((~artifact_status['required']).sum())}."
)
if not results_all.empty and len(results_all) != len(results):
    st.caption(f"Only final benchmark rows are shown in benchmark views: {len(results)} of {len(results_all)} rows.")

left, right = st.columns([1.1, 0.9])

with left:
    st.subheader('1. Project map')
    st.markdown(
        """
        **Как читать проект**
        1. **Data foundation** — данные и quality gates  
        2. **Metrics & baselines** — ranking protocol  
        3. **Deep / sequential** — NCF, RBM, SASRec, новые модели  
        4. **Hybrid / business** — RFM, long-tail, cannibalization  
        5. **Defense layer** — понятная визуализация и финальная подача
        """
    )

    if not workstreams.empty:
        status_counts = workstreams['status'].value_counts().rename_axis('status').reset_index(name='count')
        st.markdown('**Workstreams**')
        st.dataframe(workstreams, use_container_width=True, hide_index=True)
        st.caption('Эта таблица нужна, чтобы научрук быстро увидел, из каких кусков состоит работа и что уже готово.')

with right:
    st.subheader('2. Research questions')
    if not research_questions.empty:
        st.dataframe(research_questions, use_container_width=True, hide_index=True)

    st.subheader('3. Models and novelty')
    if not model_registry.empty:
        st.dataframe(model_registry, use_container_width=True, hide_index=True)

st.markdown('---')
bottom_left, bottom_right = st.columns([1.0, 1.0])

with bottom_left:
    st.subheader('4. Notebook outline → thesis storyline')
    if not notebook_outline.empty:
        st.dataframe(notebook_outline.head(20), use_container_width=True, hide_index=True)
        st.caption('Outline взят из текущего notebook и превращен в понятную карту блоков.')

with bottom_right:
    st.subheader('5. Experiment backlog')
    if not experiment_backlog.empty:
        st.dataframe(experiment_backlog, use_container_width=True, hide_index=True)
        st.caption('Тут видно, где база уже есть, а где еще планируются новые эксперименты.')

st.markdown('---')
st.subheader('6. Priority Roadmap')
if not two_week_roadmap.empty:
    st.dataframe(two_week_roadmap, use_container_width=True, hide_index=True)
    st.caption('Этот roadmap удерживает фокус на hybrid-first линии и не дает проекту расползтись в лишние deep-ветки.')

st.markdown('---')
st.subheader('7. Current Takeaways')
takeaways = benchmark_takeaways(results, feature_importance=feature_importance)
if takeaways:
    st.markdown('\n'.join(f"- {line}" for line in takeaways))
else:
    st.info('Реальные benchmark takeaways появятся после экспорта results.csv.')

st.markdown('---')
st.subheader('8. Artifact status')
st.dataframe(artifact_status, use_container_width=True, hide_index=True)

st.markdown('---')
st.markdown(
    """
    **Как показывать это руководителю**
    - Home: project map и research questions  
    - Data Audit: качество и структура данных  
    - Model Benchmark: главная таблица сравнения  
    - Robustness & Business: long-tail, RFM, cannibalization  
    - Recommendation Sandbox: один человеческий пример выдачи
    """
)
