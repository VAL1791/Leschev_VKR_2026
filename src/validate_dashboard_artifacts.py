from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data_contract import (
    ARTIFACT_FILENAMES,
    ARTIFACT_SCHEMAS,
    DATASETS_IN_SCOPE,
    MINIMUM_REQUIRED_ARTIFACTS,
    missing_columns,
    validate_results_frame,
)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def _known_algorithms(project_root: Path) -> list[str]:
    registry_path = project_root / "starter_kit" / "model_registry.csv"
    if not registry_path.exists():
        return []
    registry = pd.read_csv(registry_path)
    if "algorithm" not in registry.columns:
        return []
    return sorted(registry["algorithm"].dropna().astype(str).unique().tolist())


def validate_artifact_dir(base_dir: Path, project_root: Path) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    for logical_name in MINIMUM_REQUIRED_ARTIFACTS:
        filename = ARTIFACT_FILENAMES[logical_name]
        artifact_path = base_dir / filename
        if not artifact_path.exists():
            errors.append(f"Missing required artifact: {filename}")

    loaded_frames: dict[str, pd.DataFrame] = {}
    for logical_name, filename in ARTIFACT_FILENAMES.items():
        frame = _read_csv(base_dir / filename)
        if frame.empty:
            continue
        loaded_frames[logical_name] = frame
        required_columns = ARTIFACT_SCHEMAS.get(logical_name, ())
        missing = missing_columns(frame, required_columns)
        if missing:
            errors.append(f"{filename} is missing columns: {', '.join(missing)}")

    results = loaded_frames.get("results")
    if results is not None:
        result_errors, result_warnings = validate_results_frame(
            results,
            known_datasets=DATASETS_IN_SCOPE,
            known_algorithms=_known_algorithms(project_root),
        )
        errors.extend(result_errors)
        warnings.extend(result_warnings)

    recommendations = loaded_frames.get("recommendations")
    user_history = loaded_frames.get("user_history")
    if recommendations is not None and user_history is not None:
        rec_datasets = set(recommendations["dataset"].dropna().astype(str))
        hist_datasets = set(user_history["dataset"].dropna().astype(str))
        for dataset_name in DATASETS_IN_SCOPE:
            if dataset_name not in rec_datasets:
                warnings.append(f"recommendations.csv has no sandbox example for dataset '{dataset_name}'")
            if dataset_name not in hist_datasets:
                warnings.append(f"user_history.csv has no sandbox history for dataset '{dataset_name}'")

    return errors, warnings


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate dashboard artifact exports.")
    parser.add_argument(
        "--artifacts-dir",
        default="dashboard/artifacts",
        help="Directory with CSV exports produced by the notebook.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    artifacts_dir = Path(args.artifacts_dir)
    if not artifacts_dir.is_absolute():
        artifacts_dir = (project_root / artifacts_dir).resolve()

    errors, warnings = validate_artifact_dir(artifacts_dir, project_root)

    print(f"Artifact directory: {artifacts_dir}")
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    if errors:
        print("Errors:")
        for error in errors:
            print(f"  - {error}")
        return 1

    print("Artifact validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
