from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import streamlit as st

from dashboard_core import ablation_takeaways, artifact_readiness, benchmark_summary, benchmark_takeaways, load_starter_tables, page_intro, sidebar_controls, significance_takeaways

payload, artifact_dir = sidebar_controls()
starter = load_starter_tables()
artifact_status = artifact_readiness(payload)
results = payload['results']
feature_importance = payload['feature_importance']
significance_tests = payload['significance_tests']

workstreams = starter['workstreams']
research_questions = starter['research_questions']
experiment_backlog = starter['experiment_backlog']
notebook_outline = starter['notebook_outline']
two_week_roadmap = starter['two_week_roadmap']

page_intro('Supervisor Brief', 'Готовая страница для объяснения научруку: что уже есть, что остается и как это показывать на защите.')
st.caption(benchmark_summary())

st.subheader('Review sequence')
st.markdown(
    """
    1. **Project map** — показать, что работа собрана в понятную систему  
    2. **Data audit** — доказать корректность данных  
    3. **Leaderboard** — защитить метрики и базовое сравнение  
    4. **Robustness** — показать, что есть не только accuracy, но и бизнес-смысл  
    5. **Sandbox** — завершить понятным примером рекомендаций
    """
)

st.subheader('Auto-generated defense takeaways')
takeaways = benchmark_takeaways(results, feature_importance=feature_importance)
if takeaways:
    st.markdown('\n'.join(f"- {line}" for line in takeaways))
else:
    st.info('Takeaways появятся после загрузки реальных benchmark-артефактов.')

st.subheader('Ablation reading')
ablation_lines = ablation_takeaways(payload.get('results_all', results))
if ablation_lines:
    st.markdown('\n'.join(f"- {line}" for line in ablation_lines))
else:
    st.info('Ablation takeaways появятся после выгрузки ablation-строк.')

st.subheader('Significance reading')
significance_lines = significance_takeaways(significance_tests)
if significance_lines:
    st.markdown('\n'.join(f"- {line}" for line in significance_lines))
else:
    st.info('Significance takeaways появятся после выгрузки significance_tests.csv.')

col1, col2 = st.columns([1.0, 1.0])

with col1:
    st.subheader('Workstreams')
    st.dataframe(workstreams, use_container_width=True, hide_index=True)

    st.subheader('Research questions')
    st.dataframe(research_questions, use_container_width=True, hide_index=True)

with col2:
    st.subheader('Experiment backlog')
    st.dataframe(experiment_backlog, use_container_width=True, hide_index=True)

    st.subheader('Notebook map')
    st.dataframe(notebook_outline, use_container_width=True, hide_index=True)

st.subheader('Priority roadmap for the next two weeks')
if not two_week_roadmap.empty:
    st.dataframe(two_week_roadmap, use_container_width=True, hide_index=True)
    st.caption('Эта таблица специально держит фокус на hybrid-first истории: сначала вклад feature engineering, потом один selective sequential reference.')

st.markdown('---')
st.markdown(
    """
    **Что стоит заполнить перед ближайшим показом**
    - перенести выводы из ablation и defense summary в экспериментальную главу;
    - зафиксировать 5–7 финальных тезисов по различию shallow / hybrid / sequential;
    - выбрать нужен ли больший Goodreads subset или текущий достаточно убедителен;
    - добавить только один selective sequential reference, если на это остается вычислительный бюджет.
    """
)
st.subheader('Artifact readiness')
st.dataframe(artifact_status, use_container_width=True, hide_index=True)
