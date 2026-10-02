"""Checkpoint persistence and compatibility helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn


FORMAT_VERSION = 1


def unwrap_model(model: nn.Module) -> nn.Module:
    return getattr(model, "module", model)


def save_checkpoint(
    path: str | Path,
    *,
    kind: str,
    model: nn.Module,
    epoch: int,
    global_step: int,
    config: Mapping[str, Any],
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    scaler: Any = None,
    metrics: Mapping[str, float] | None = None,
) -> Path:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "kind": kind,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "config": dict(config),
        "model_state": unwrap_model(model).state_dict(),
        "metrics": dict(metrics or {}),
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler_state"] = scaler.state_dict()
    torch.save(payload, checkpoint_path)
    return checkpoint_path


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    checkpoint_path = Path(path).expanduser()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint object in {checkpoint_path}")
    return checkpoint


@dataclass
class LoadReport:
    source_format: str
    loaded_keys: int
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    skipped_shape_keys: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_format": self.source_format,
            "loaded_keys": self.loaded_keys,
            "missing_keys": list(self.missing_keys),
            "unexpected_keys": list(self.unexpected_keys),
            "skipped_shape_keys": list(self.skipped_shape_keys),
        }

    def __str__(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


def _legacy_image_key(key: str) -> str | None:
    prefix = "foundation_model.img_encoder."
    if not key.startswith(prefix):
        return None
    key = key[len(prefix) :]
    if key.startswith("pretrained_resnet."):
        return None
    replacements = (
        ("layer2.layers.", "layer2.blocks."),
        ("layer3.layers.", "layer3.blocks."),
        ("layer4.layers.", "layer4.blocks."),
        (".modulator.channel_fc.", ".modulator.channel_scale."),
        (".modulator.spatial_conv.", ".modulator.spatial_offset."),
        ("dynamic_codition_generator_fine.", "condition_synthesizers.fine."),
        ("dynamic_codition_generator_fused.", "condition_synthesizers.fused."),
        ("dynamic_codition_generator_coarse.", "condition_synthesizers.coarse."),
        ("hier_cond_encoder.fine_proj.0.", "label_embedder.fine_projection.0."),
        ("hier_cond_encoder.coarse_proj.0.", "label_embedder.coarse_projection.0."),
        ("hier_cond_encoder.hier_fusion.3.", "label_embedder.hierarchy_gate.0."),
        ("embedder_1.", "projection_head.0."),
        ("embedder_2.", "projection_head.2."),
        ("alpha_fine", "condition_mix_weights.fine"),
        ("alpha_fused", "condition_mix_weights.fused"),
        ("alpha_coarse", "condition_mix_weights.coarse"),
    )
    for source, target in replacements:
        key = key.replace(source, target)
    # The retained fused-label MLP has an additional layer and cannot be
    # migrated safely into the clarified implementation.
    if key.startswith("hier_cond_encoder.final_fusion") or key.startswith(
        "hier_cond_encoder.hier_fusion"
    ):
        return None
    return key


def _extract_encoder_state(checkpoint: Mapping[str, Any]) -> tuple[dict[str, torch.Tensor], str]:
    if "model_state" in checkpoint:
        state = checkpoint["model_state"]
        if not isinstance(state, Mapping):
            raise TypeError("model_state is not a mapping.")
        extracted = {
            key[len("image_encoder.") :]: value
            for key, value in state.items()
            if key.startswith("image_encoder.")
        }
        if not extracted:
            # A downstream checkpoint may store the encoder directly.
            extracted = {
                key[len("encoder.") :]: value
                for key, value in state.items()
                if key.startswith("encoder.")
            }
        if not extracted:
            extracted = dict(state)
        return extracted, "uhm-fi-v1"

    state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state, Mapping):
        raise TypeError("Checkpoint state is not a mapping.")
    legacy: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        mapped = _legacy_image_key(str(key))
        if mapped is not None and isinstance(value, torch.Tensor):
            legacy[mapped] = value
    if legacy:
        return legacy, "legacy-lightning"
    return dict(state), "raw-state-dict"


def load_image_encoder_weights(
    encoder: nn.Module,
    checkpoint_path: str | Path,
    *,
    strict: bool = False,
) -> LoadReport:
    checkpoint = load_checkpoint(checkpoint_path)
    source_state, source_format = _extract_encoder_state(checkpoint)
    target_state = encoder.state_dict()
    compatible: dict[str, torch.Tensor] = {}
    skipped_shapes: list[str] = []
    unexpected: list[str] = []
    for key, value in source_state.items():
        if key not in target_state:
            unexpected.append(key)
            continue
        if tuple(value.shape) != tuple(target_state[key].shape):
            skipped_shapes.append(key)
            continue
        compatible[key] = value
    incompatible = encoder.load_state_dict(compatible, strict=False)
    missing = tuple(incompatible.missing_keys)
    unexpected_all = tuple(sorted(set(unexpected).union(incompatible.unexpected_keys)))
    report = LoadReport(
        source_format=source_format,
        loaded_keys=len(compatible),
        missing_keys=missing,
        unexpected_keys=unexpected_all,
        skipped_shape_keys=tuple(skipped_shapes),
    )
    if strict and (missing or unexpected_all or skipped_shapes):
        raise RuntimeError(f"Strict encoder checkpoint loading failed:\n{report}")
    return report


def restore_training_state(
    checkpoint_path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    scaler: Any = None,
    strict: bool = True,
) -> tuple[int, int, dict[str, Any]]:
    checkpoint = load_checkpoint(checkpoint_path)
    state = checkpoint.get("model_state")
    if state is None:
        raise ValueError("Resume requires a new-format checkpoint with model_state.")
    unwrap_model(model).load_state_dict(state, strict=strict)
    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scheduler is not None and "scheduler_state" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    if scaler is not None and "scaler_state" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state"])
    return (
        int(checkpoint.get("epoch", -1)) + 1,
        int(checkpoint.get("global_step", 0)),
        dict(checkpoint.get("metrics", {})),
    )
