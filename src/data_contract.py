from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd

DATASETS_IN_SCOPE = (
    "MovieLens 20M",
    "Google Local South Carolina",
    "Goodreads Fantasy 10K",
)

PRIMARY_BENCHMARK = {
    "mode": "exc",
    "topk": 10,
    "primary_metric": "ndcg",
    "secondary_metrics": ("recall", "map", "precision", "coverage", "novelty"),
    "temporal_split": "75/25",
}

REQUIRED_INTERACTION_COLUMNS = (
    "dataset",
    "user_id",
    "item_id",
    "timestamp",
)

OPTIONAL_INTERACTION_COLUMNS = (
    "rating",
    "target",
    "order_id",
    "text",
    "category",
)

RESULTS_KEY_COLUMNS = (
    "dataset",
    "algorithm",
    "topk",
    "mode",
)

REQUIRED_RESULTS_COLUMNS = (
    *RESULTS_KEY_COLUMNS,
    "ndcg",
    "recall",
    "precision",
    "map",
)

OPTIONAL_RESULTS_COLUMNS = (
    "coverage",
    "novelty",
    "train_seconds",
    "inference_ms_per_user",
    "run_status",
    "note",
)

ARTIFACT_FILENAMES = {
    "datasets_summary": "datasets_summary.csv",
    "results": "results.csv",
    "cannibalization": "cannibalization.csv",
    "rfm_metrics": "rfm_metrics.csv",
    "revenue_mix": "revenue_mix.csv",
    "slice_metrics": "slice_metrics.csv",
    "feature_importance": "feature_importance.csv",
    "efficiency": "efficiency.csv",
    "metric_intervals": "metric_intervals.csv",
    "significance_tests": "significance_tests.csv",
    "monthly_interactions": "monthly_interactions.csv",
    "recommendations": "recommendations.csv",
    "user_history": "user_history.csv",
    "activity_distribution": "activity_distribution.csv",
    "rating_distribution": "rating_distribution.csv",
    "long_tail_curve": "long_tail_curve.csv",
}

ARTIFACT_SCHEMAS = {
    "datasets_summary": ("dataset", "users", "items", "events"),
    "results": REQUIRED_RESULTS_COLUMNS,
    "cannibalization": ("dataset", "algorithm", "topk"),
    "rfm_metrics": ("algorithm", "rfm_segment"),
    "revenue_mix": ("RFM_score", "proxy_category", "share"),
    "slice_metrics": ("dataset", "algorithm", "slice_name", "topk", "mode"),
    "feature_importance": ("algorithm", "feature", "importance"),
    "efficiency": ("dataset", "algorithm", "train_seconds", "inference_ms_per_user"),
    "metric_intervals": ("dataset", "algorithm", "metric", "mean", "ci_low", "ci_high", "n_users"),
    "significance_tests": (
        "dataset",
        "metric",
        "reference_algorithm",
        "comparison_algorithm",
        "delta_mean",
        "ci_low",
        "ci_high",
        "p_value",
        "p_value_adj",
        "significant",
        "n_users",
    ),
    "monthly_interactions": ("dataset", "month", "interactions"),
    "recommendations": ("dataset", "user_id", "algorithm", "rank", "item_id"),
    "user_history": ("dataset", "user_id", "interaction_rank", "item_id"),
    "activity_distribution": ("dataset", "entity_type", "bucket", "count"),
    "rating_distribution": ("dataset", "rating", "count"),
    "long_tail_curve": ("dataset", "item_share_pct", "interaction_share_pct"),
}

MINIMUM_REQUIRED_ARTIFACTS = (
    "datasets_summary",
    "results",
    "monthly_interactions",
    "recommendations",
    "user_history",
)

OPTIONAL_ARTIFACTS = tuple(
    logical_name for logical_name in ARTIFACT_FILENAMES if logical_name not in MINIMUM_REQUIRED_ARTIFACTS
)

VALID_RESULTS_MODES = ("exc", "inc")
SUPPORTED_TOPK = (5, 10, 20)

COMPLETED_RUN_STATUSES = {"done", "ready", "completed"}
INACTIVE_RUN_STATUSES = {"not_run", "not applicable", "not_applicable", "planned", "draft", "failed", "ablation"}


def missing_columns(frame: pd.DataFrame, required_columns: Sequence[str]) -> list[str]:
    return [column for column in required_columns if column not in frame.columns]


def filter_completed_results(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty or "run_status" not in results.columns:
        return results
    statuses = results["run_status"].fillna("done").astype(str).str.strip().str.lower()
    return results[statuses.isin(COMPLETED_RUN_STATUSES)].copy()


def validate_interaction_frame(frame: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    missing = missing_columns(frame, REQUIRED_INTERACTION_COLUMNS)
    if missing:
        errors.append(
            "Interaction frame is missing required columns: " + ", ".join(missing)
        )
    return errors


def validate_results_frame(
    results: pd.DataFrame,
    known_datasets: Iterable[str] = DATASETS_IN_SCOPE,
    known_algorithms: Iterable[str] | None = None,
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    missing = missing_columns(results, REQUIRED_RESULTS_COLUMNS)
    if missing:
        errors.append("results.csv is missing required columns: " + ", ".join(missing))
        return errors, warnings

    duplicate_mask = results.duplicated(list(RESULTS_KEY_COLUMNS), keep=False)
    if duplicate_mask.any():
        duplicate_rows = results.loc[duplicate_mask, list(RESULTS_KEY_COLUMNS)].astype(str)
        preview = duplicate_rows.head(5).to_dict(orient="records")
        errors.append(f"results.csv contains duplicate benchmark keys: {preview}")

    numeric_columns = [column for column in REQUIRED_RESULTS_COLUMNS if column not in RESULTS_KEY_COLUMNS]
    numeric_columns += [column for column in OPTIONAL_RESULTS_COLUMNS if column in results.columns]
    numeric_columns = [column for column in numeric_columns if column not in {"run_status", "note"}]
    for column in numeric_columns:
        coerced = pd.to_numeric(results[column], errors="coerce")
        bad_rows = int((coerced.isna() & results[column].notna()).sum())
        if bad_rows > 0:
            warnings.append(f"Column {column} has {bad_rows} non-numeric values.")

    bounded_metrics = [
        column
        for column in ("ndcg", "recall", "precision", "map", "coverage", "novelty")
        if column in results.columns
    ]
    for column in bounded_metrics:
        values = pd.to_numeric(results[column], errors="coerce")
        if ((values < 0) | (values > 1)).any():
            warnings.append(f"Column {column} contains values outside [0, 1].")

    modes = set(results["mode"].dropna().astype(str).unique())
    unknown_modes = sorted(modes.difference(VALID_RESULTS_MODES))
    if unknown_modes:
        warnings.append("results.csv has unknown modes: " + ", ".join(unknown_modes))

    topk = pd.to_numeric(results["topk"], errors="coerce")
    if topk.isna().any():
        warnings.append("results.csv contains non-numeric topk values.")
    else:
        unexpected_topk = sorted({int(value) for value in topk.unique() if int(value) not in SUPPORTED_TOPK})
        if unexpected_topk:
            warnings.append(
                "results.csv contains topk values outside the agreed benchmark grid: "
                + ", ".join(map(str, unexpected_topk))
            )

    known_dataset_set = set(known_datasets)
    unknown_datasets = sorted(set(results["dataset"].dropna().astype(str)).difference(known_dataset_set))
    if unknown_datasets:
        warnings.append("results.csv has datasets outside the scoped benchmark: " + ", ".join(unknown_datasets))

    if known_algorithms is not None:
        known_algorithm_set = set(known_algorithms)
        unknown_algorithms = sorted(
            set(results["algorithm"].dropna().astype(str)).difference(known_algorithm_set)
        )
        if unknown_algorithms:
            warnings.append(
                "results.csv has algorithms outside model_registry.csv: " + ", ".join(unknown_algorithms)
            )

    return errors, warnings
