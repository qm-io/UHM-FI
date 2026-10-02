"""Label-derived and image-derived hierarchical conditions used by UHM-FI."""

from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor, nn


CONDITION_LEVELS = ("fine", "fused", "coarse")


class HierarchicalLabelEmbedder(nn.Module):
    """Embed fine/coarse multi-hot labels and synthesize their fused level."""

    def __init__(
        self,
        num_fine_labels: int,
        num_coarse_labels: int,
        condition_dim: int = 256,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_fine_labels = num_fine_labels
        self.num_coarse_labels = num_coarse_labels
        self.condition_dim = condition_dim

        self.fine_projection = nn.Sequential(
            nn.Linear(num_fine_labels, condition_dim),
            nn.LayerNorm(condition_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.coarse_projection = nn.Sequential(
            nn.Linear(num_coarse_labels, condition_dim),
            nn.LayerNorm(condition_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.hierarchy_gate = nn.Sequential(
            nn.Linear(condition_dim * 2, condition_dim),
            nn.Sigmoid(),
        )
        self.fused_projection = nn.Sequential(
            nn.Linear(condition_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, condition_dim),
            nn.LayerNorm(condition_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, fine_labels: Tensor, coarse_labels: Tensor) -> dict[str, Tensor]:
        if fine_labels.ndim != 2 or fine_labels.shape[1] != self.num_fine_labels:
            raise ValueError(
                f"fine_labels must have shape [B, {self.num_fine_labels}], "
                f"got {tuple(fine_labels.shape)}"
            )
        if coarse_labels.ndim != 2 or coarse_labels.shape[1] != self.num_coarse_labels:
            raise ValueError(
                f"coarse_labels must have shape [B, {self.num_coarse_labels}], "
                f"got {tuple(coarse_labels.shape)}"
            )

        fine_mask = fine_labels.abs().sum(dim=1, keepdim=True).gt(0)
        coarse_mask = coarse_labels.abs().sum(dim=1, keepdim=True).gt(0)
        fine = self.fine_projection(fine_labels.float()) * fine_mask
        coarse = self.coarse_projection(coarse_labels.float()) * coarse_mask
        gate = self.hierarchy_gate(torch.cat((coarse, fine), dim=1))
        fused = self.fused_projection(
            torch.cat((coarse * gate, fine * (1.0 - gate)), dim=1)
        )
        fused = fused * (fine_mask | coarse_mask)
        return {"fine": fine, "fused": fused, "coarse": coarse}


class ConditionSynthesizer(nn.Module):
    """Generate an anatomical condition directly from an image feature map."""

    def __init__(self, in_channels: int, condition_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, condition_dim),
            nn.Tanh(),
        )
        self.residual = nn.Parameter(torch.zeros(condition_dim))

    def forward(self, feature_map: Tensor) -> Tensor:
        pooled = self.pool(feature_map).flatten(1)
        return self.mlp(pooled) + self.residual


def merge_label_and_dynamic_conditions(
    dynamic: Tensor,
    label: Tensor | None,
    mode: str = "sum",
    label_weight: Tensor | float | None = None,
) -> Tensor:
    """Combine conditions during pre-training; use dynamic alone at inference.

    ``manual`` was the name used by the retained prototype.  The implementation
    uses ``label`` because these conditions are automatically derived from
    report labels rather than manually authored condition vectors.
    """

    if label is None:
        return dynamic
    if label.shape != dynamic.shape:
        raise ValueError("Label-derived and dynamic condition shapes must match.")
    if mode == "learned":
        if label_weight is None:
            raise ValueError("learned condition merge requires label_weight.")
        weight_logit = torch.as_tensor(
            label_weight,
            dtype=dynamic.dtype,
            device=dynamic.device,
        )
        weight = torch.sigmoid(weight_logit)
        return weight * label + (1.0 - weight) * dynamic
    if mode == "sum":
        return label + dynamic
    if mode == "mean":
        return 0.5 * (label + dynamic)
    if mode == "label":
        return label
    if mode == "dynamic":
        return dynamic
    raise ValueError(f"Unsupported condition merge mode: {mode}")


# Backward-compatible name used by the first executable reconstruction.
merge_manual_and_dynamic_conditions = merge_label_and_dynamic_conditions


def validate_condition_dict(conditions: Mapping[str, Tensor]) -> None:
    missing = [level for level in CONDITION_LEVELS if level not in conditions]
    if missing:
        raise KeyError(f"Missing condition levels: {missing}")
