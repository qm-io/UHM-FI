"""End-to-end UHM-FI image-text model."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn

from .config import UHMFIConfig
from .interleaving import FeatureInterleaver
from .text import build_text_encoder
from .vision import HierarchicallyModulatedResNet50


class UHMFI(nn.Module):
    """Universal Hierarchical Modulation with Feature Interleaving."""

    def __init__(self, config: UHMFIConfig | None = None) -> None:
        super().__init__()
        self.config = config or UHMFIConfig()
        self.config.validate()
        self.image_encoder = HierarchicallyModulatedResNet50(self.config.vision)
        self.text_encoder = build_text_encoder(
            self.config.text, self.config.vision.embedding_dim
        )
        self.feature_interleaver = FeatureInterleaver(
            ratio=self.config.interleaving.ratio,
            strategy=self.config.interleaving.strategy,
            detach_selection=self.config.interleaving.detach_selection,
        )

    @staticmethod
    def _unpack_batch(batch: Mapping[str, Tensor]) -> dict[str, Tensor | None]:
        return {
            "images": batch.get("images"),
            "input_ids": batch.get("input_ids"),
            "attention_mask": batch.get("attention_mask"),
            "token_type_ids": batch.get("token_type_ids"),
            "fine_labels": batch.get("fine_labels"),
            "coarse_labels": batch.get("coarse_labels"),
        }

    def encode_image(
        self,
        images: Tensor,
        fine_labels: Tensor | None = None,
        coarse_labels: Tensor | None = None,
    ) -> dict[str, object]:
        return self.image_encoder(images, fine_labels, coarse_labels)

    def encode_text(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        token_type_ids: Tensor | None = None,
    ) -> Tensor:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        return self.text_encoder(input_ids, attention_mask, token_type_ids)

    def forward(
        self,
        images: Tensor | Mapping[str, Tensor],
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        token_type_ids: Tensor | None = None,
        fine_labels: Tensor | None = None,
        coarse_labels: Tensor | None = None,
        apply_interleaving: bool | None = None,
    ) -> dict[str, object]:
        if isinstance(images, Mapping):
            unpacked = self._unpack_batch(images)
            images = unpacked["images"]
            input_ids = unpacked["input_ids"]
            attention_mask = unpacked["attention_mask"]
            token_type_ids = unpacked["token_type_ids"]
            fine_labels = unpacked["fine_labels"]
            coarse_labels = unpacked["coarse_labels"]
        if not isinstance(images, Tensor):
            raise TypeError("images must be a Tensor or a batch mapping.")

        image_output = self.encode_image(images, fine_labels, coarse_labels)
        raw_image_embedding = image_output["image_embedding"]
        assert isinstance(raw_image_embedding, Tensor)

        if input_ids is None:
            return {
                **image_output,
                "raw_image_embedding": raw_image_embedding,
                "raw_text_embedding": None,
                "text_embedding": None,
                "interleaving": {"applied": False, "reason": "image_only"},
            }

        raw_text_embedding = self.encode_text(
            input_ids, attention_mask, token_type_ids
        )
        if apply_interleaving is None:
            apply_interleaving = self.config.interleaving.enabled and (
                self.training or not self.config.interleaving.training_only
            )

        if apply_interleaving:
            image_embedding, text_embedding, metadata = self.feature_interleaver(
                raw_image_embedding, raw_text_embedding
            )
        else:
            image_embedding = raw_image_embedding
            text_embedding = raw_text_embedding
            metadata = {
                "applied": False,
                "ratio": self.config.interleaving.ratio,
                "strategy": self.config.interleaving.strategy,
            }

        return {
            **image_output,
            "raw_image_embedding": raw_image_embedding,
            "raw_text_embedding": raw_text_embedding,
            "image_embedding": image_embedding,
            "text_embedding": text_embedding,
            "interleaving": metadata,
        }
