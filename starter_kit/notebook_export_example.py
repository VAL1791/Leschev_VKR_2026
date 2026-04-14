# Paste this pattern into the notebook after you build final tables.

from pathlib import Path

from src.local_data_config import load_local_data_config
from src.export_dashboard_artifacts import export_dashboard_artifacts

cfg = load_local_data_config()
output_dir = Path(cfg.dashboard_artifacts_root)

frames = {
    'datasets_summary': datasets_summary_df,
    'results': results_df,
    'cannibalization': cannibalization_df,
    'rfm_metrics': rfm_metrics_df,
    'revenue_mix': revenue_mix_df,
    'slice_metrics': slice_metrics_df,
    'feature_importance': feature_importance_df,
    'efficiency': efficiency_df,
    'monthly_interactions': monthly_interactions_df,
    'recommendations': recommendations_df,
    'user_history': user_history_df,
    'activity_distribution': activity_distribution_df,
    'rating_distribution': rating_distribution_df,
    'long_tail_curve': long_tail_curve_df,
}

export_dashboard_artifacts(output_dir, frames)
print(f"Exported to {output_dir.resolve()}")
print("Validate next: python -m src.validate_dashboard_artifacts --artifacts-dir dashboard/artifacts")
