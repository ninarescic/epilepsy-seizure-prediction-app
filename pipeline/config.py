from dataclasses import dataclass
from pathlib import Path


@dataclass
class AppConfig:
    preprocessing_dir: Path = Path("outputs/preprocessing")
    feature_selection_dir: Path = Path("outputs/feature_selection")
    models_dir: Path = Path("outputs/models")
    explainability_dir: Path = Path("outputs/explainability")