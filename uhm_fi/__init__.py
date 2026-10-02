"""UHM-FI model package."""

from .builder import build_model, build_objective, resolve_config
from .config import (
    DownstreamConfig,
    InterleavingConfig,
    LossConfig,
    PretrainingConfig,
    TextConfig,
    UHMFIConfig,
    VisionConfig,
    WandbConfig,
)
from .interleaving import FeatureInterleaver
from .losses import (
    HierarchicalConditionConsistencyLoss,
    SymmetricContrastiveLoss,
    UHMFIObjective,
)
from .model import UHMFI
from .downstream import UHMFIClassifier, UHMFISegmenter, build_downstream_model
from .vision import HierarchicallyModulatedResNet50

__all__ = [
    "DownstreamConfig",
    "FeatureInterleaver",
    "HierarchicalConditionConsistencyLoss",
    "HierarchicallyModulatedResNet50",
    "InterleavingConfig",
    "LossConfig",
    "PretrainingConfig",
    "SymmetricContrastiveLoss",
    "TextConfig",
    "UHMFI",
    "UHMFIClassifier",
    "UHMFISegmenter",
    "UHMFIConfig",
    "UHMFIObjective",
    "VisionConfig",
    "WandbConfig",
    "build_model",
    "build_objective",
    "build_downstream_model",
    "resolve_config",
]
