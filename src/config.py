"""Project configuration loaded from the root ``config.yaml`` file."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .utils import ROOT_DIR


@dataclass(frozen=True, slots=True)
class ProjectConfig:
    similarity_threshold: float = 0.35
    output_directory: str = "outputs"
    font_scale: float = 0.55
    bounding_box_thickness: int = 2
    landmark_radius: int = 3
    logging_enabled: bool = True
    classifier_confidence_threshold: float = 0.70
    training_test_size: float = 0.20
    random_state: int = 42


def load_config(path: Path | None = None) -> ProjectConfig:
    """Load configuration while retaining safe defaults for missing keys."""
    config_path = path or ROOT_DIR / "config.yaml"
    if not config_path.exists():
        return ProjectConfig()
    with config_path.open("r", encoding="utf-8") as handle:
        values: dict[str, Any] = yaml.safe_load(handle) or {}
    allowed = {field: values[field] for field in ProjectConfig.__dataclass_fields__ if field in values}
    return ProjectConfig(**allowed)
