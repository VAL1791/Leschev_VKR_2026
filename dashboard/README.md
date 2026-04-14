# dashboard

Multipage Streamlit dashboard for the thesis.

## Pages

- **Home** — project map, research questions, model scope, notebook outline
- **Data Audit** — datasets, temporal volume, activity buckets, long-tail, explicit target distribution
- **Model Benchmark** — leaderboard, heatmap, top-K trajectories, quality vs cost
- **Robustness & Business** — inc./exc. gap, RFM, slices, revenue mix, explainability
- **Recommendation Sandbox** — user history and example recommendations
- **Supervisor Brief** — ready-made summary for quick review

## Run

```bash
cd dashboard
pip install -r requirements.txt
python generate_demo_artifacts.py
streamlit run app.py
```

## Expected real inputs

Place exported CSV tables into `dashboard/artifacts/`.
If they are absent, the UI falls back to `sample_artifacts/`.

## Why this dashboard exists

The point is not to imitate an online shop UI.
The point is to make the **research structure** readable:
data -> metrics -> benchmark -> robustness -> interpretation.
