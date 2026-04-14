from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pandas as pd

from src.data_contract import ARTIFACT_FILENAMES, ARTIFACT_SCHEMAS, missing_columns


def _validate_export_frame(logical_name: str, frame: pd.DataFrame) -> None:
    required_columns = ARTIFACT_SCHEMAS.get(logical_name, ())
    missing = missing_columns(frame, required_columns)
    if missing:
        raise ValueError(
            f"{logical_name} is missing required columns for dashboard export: {', '.join(missing)}"
        )


def export_dashboard_artifacts(output_dir: str | Path, frames: Mapping[str, pd.DataFrame]) -> Path:
    """Export notebook outputs into dashboard/artifacts.

    The API stays intentionally small:
    - notebook code passes one mapping of logical artifact names to DataFrames;
    - every provided frame is schema-checked before writing;
    - partial exports are allowed, but each exported artifact must satisfy its contract.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for logical_name, filename in ARTIFACT_FILENAMES.items():
        frame = frames.get(logical_name)
        if frame is None:
            continue
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"{logical_name} must be a pandas DataFrame.")
        if frame.empty:
            continue
        _validate_export_frame(logical_name, frame)
        frame.to_csv(output_dir / filename, index=False)

    return output_dir


if __name__ == "__main__":
    print("Import this module inside the notebook and call export_dashboard_artifacts(...).")
