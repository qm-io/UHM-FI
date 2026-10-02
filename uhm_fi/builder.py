"""Public construction helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Any

from .config import UHMFIConfig
from .losses import UHMFIObjective
from .model import UHMFI


def resolve_config(
    config: UHMFIConfig | Mapping[str, Any] | str | Path | None = None,
) -> UHMFIConfig:
    if config is None:
        resolved = UHMFIConfig()
    elif isinstance(config, UHMFIConfig):
        resolved = config
    elif isinstance(config, Mapping):
        resolved = UHMFIConfig.from_dict(config)
    else:
        resolved = UHMFIConfig.from_yaml(config)
    resolved.validate()
    return resolved


def build_model(
    config: UHMFIConfig | Mapping[str, Any] | str | Path | None = None,
) -> UHMFI:
    return UHMFI(resolve_config(config))


def build_objective(config: UHMFIConfig) -> UHMFIObjective:
    return UHMFIObjective(
        temperature=config.loss.temperature,
        condition_alpha=config.loss.condition_alpha,
        condition_beta=config.loss.condition_beta,
        condition_weight=config.loss.condition_weight,
        condition_distribution=config.loss.condition_distribution,
        condition_mse_weight=config.loss.condition_mse_weight,
        condition_norm_weight=config.loss.condition_norm_weight,
    )
