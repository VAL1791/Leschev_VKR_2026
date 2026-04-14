from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import tomllib


DEFAULT_CONFIG_PATH = Path("config/local_paths.toml")
EXAMPLE_CONFIG_PATH = Path("config/local_paths.example.toml")


@dataclass(frozen=True)
class LocalDataConfig:
    project_root: Path
    raw_data_root: Path
    dashboard_artifacts_root: Path
    notebook_root: Path
    dataset_paths: dict[str, Path]


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_path(base_dir: Path, raw_value: str) -> Path:
    path = Path(raw_value).expanduser()
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _read_config(config_path: Path) -> dict:
    if not config_path.exists():
        return {}
    return tomllib.loads(config_path.read_text(encoding="utf-8"))


def load_local_data_config(config_path: str | Path | None = None) -> LocalDataConfig:
    root = project_root()
    resolved_config = root / DEFAULT_CONFIG_PATH if config_path is None else Path(config_path)
    if not resolved_config.is_absolute():
        resolved_config = (root / resolved_config).resolve()

    config = _read_config(resolved_config)
    paths_cfg = config.get("paths", {})
    datasets_cfg = config.get("datasets", {})

    raw_root = _resolve_path(
        root,
        os.environ.get("THESIS_RAW_DATA_ROOT", paths_cfg.get("raw_data_root", "data/raw")),
    )
    artifacts_root = _resolve_path(
        root,
        os.environ.get("THESIS_ARTIFACTS_ROOT", paths_cfg.get("dashboard_artifacts_root", "dashboard/artifacts")),
    )
    notebook_root = _resolve_path(root, paths_cfg.get("notebook_root", "."))

    dataset_paths = {
        dataset_name: _resolve_path(root, raw_path)
        for dataset_name, raw_path in datasets_cfg.items()
    }

    return LocalDataConfig(
        project_root=root,
        raw_data_root=raw_root,
        dashboard_artifacts_root=artifacts_root,
        notebook_root=notebook_root,
        dataset_paths=dataset_paths,
    )


def get_dataset_path(dataset_name: str, config: LocalDataConfig | None = None, required: bool = False) -> Path | None:
    active_config = config or load_local_data_config()
    path = active_config.dataset_paths.get(dataset_name)
    if path is not None:
        return path
    fallback = active_config.raw_data_root / dataset_name.lower().replace(" ", "_").replace("-", "_")
    if fallback.exists():
        return fallback
    if required:
        raise FileNotFoundError(
            f"Dataset path for '{dataset_name}' is not configured. "
            f"Create {DEFAULT_CONFIG_PATH.as_posix()} from {EXAMPLE_CONFIG_PATH.as_posix()}."
        )
    return None


def describe_local_paths(config: LocalDataConfig | None = None) -> dict[str, str]:
    active_config = config or load_local_data_config()
    return {
        "project_root": str(active_config.project_root),
        "raw_data_root": str(active_config.raw_data_root),
        "dashboard_artifacts_root": str(active_config.dashboard_artifacts_root),
        "notebook_root": str(active_config.notebook_root),
        "configured_datasets": ", ".join(sorted(active_config.dataset_paths)) or "none",
    }
