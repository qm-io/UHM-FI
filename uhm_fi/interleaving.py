"""Cross-modal feature interleaving in the shared embedding space."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class FeatureInterleaver(nn.Module):
    """Exchange matched feature positions between image and text embeddings.

    The manuscript says that a predefined mutual-information strategy chooses
    the exchanged positions but does not provide an estimator.  This
    implementation ranks dimensions with the Gaussian mutual-information
    proxy derived from batch-wise Pearson correlation.
    """

    def __init__(
        self,
        ratio: float = 0.25,
        strategy: str = "mutual_information",
        detach_selection: bool = True,
    ) -> None:
        super().__init__()
        if not 0.0 <= ratio <= 1.0:
            raise ValueError("ratio must be in [0, 1].")
        if strategy not in {"mutual_information", "random", "fixed"}:
            raise ValueError(f"Unsupported interleaving strategy: {strategy}")
        self.ratio = ratio
        self.strategy = strategy
        self.detach_selection = detach_selection

    def _mutual_information_scores(self, image: Tensor, text: Tensor) -> Tensor:
        image_values = image.float()
        text_values = text.float()
        if self.detach_selection:
            image_values = image_values.detach()
            text_values = text_values.detach()
        if image_values.shape[0] < 2:
            return (image_values * text_values).abs().mean(dim=0)

        image_values = image_values - image_values.mean(dim=0, keepdim=True)
        text_values = text_values - text_values.mean(dim=0, keepdim=True)
        covariance = (image_values * text_values).mean(dim=0)
        image_variance = image_values.square().mean(dim=0)
        text_variance = text_values.square().mean(dim=0)
        correlation = covariance / (
            image_variance.sqrt() * text_variance.sqrt()
        ).clamp_min(1e-8)
        correlation_squared = correlation.square().clamp(max=1.0 - 1e-6)
        return -0.5 * torch.log1p(-correlation_squared)

    def _select_indices(self, image: Tensor, text: Tensor, count: int) -> Tensor:
        dimension = image.shape[1]
        if count == 0:
            return torch.empty(0, dtype=torch.long, device=image.device)
        if self.strategy == "mutual_information":
            scores = self._mutual_information_scores(image, text)
            return torch.topk(scores, k=count, largest=True, sorted=True).indices
        if self.strategy == "random":
            return torch.randperm(dimension, device=image.device)[:count]
        return torch.arange(count, device=image.device)

    def forward(
        self,
        image_embedding: Tensor,
        text_embedding: Tensor,
    ) -> tuple[Tensor, Tensor, dict[str, object]]:
        if image_embedding.ndim != 2 or text_embedding.ndim != 2:
            raise ValueError("Feature interleaving expects [B,D] embeddings.")
        if image_embedding.shape != text_embedding.shape:
            raise ValueError("Image and text embeddings must have identical shapes.")

        dimension = image_embedding.shape[1]
        count = min(dimension, int(math.floor(dimension * self.ratio + 0.5)))
        indices = self._select_indices(image_embedding, text_embedding, count)
        mask = torch.zeros(dimension, dtype=torch.bool, device=image_embedding.device)
        mask[indices] = True
        broadcast_mask = mask.unsqueeze(0)

        image_interleaved = torch.where(
            broadcast_mask, text_embedding, image_embedding
        )
        text_interleaved = torch.where(
            broadcast_mask, image_embedding, text_embedding
        )
        metadata = {
            "applied": count > 0,
            "ratio": self.ratio,
            "strategy": self.strategy,
            "detach_selection": self.detach_selection,
            "selected_count": count,
            "selected_indices": indices,
            "mask": mask,
        }
        return image_interleaved, text_interleaved, metadata
