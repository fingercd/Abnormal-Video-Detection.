"""特征抽取、训练与评测执行引擎。"""

from vadbench.engine.extract import FeatureExtractionEngine
from vadbench.engine.runner import (
    HeadOnlyTrainingConfig,
    TrainingRunResult,
    train_feature_head,
)

__all__ = [
    "FeatureExtractionEngine",
    "HeadOnlyTrainingConfig",
    "TrainingRunResult",
    "train_feature_head",
]
