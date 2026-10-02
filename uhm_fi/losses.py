"""Training objectives described in the UHM-FI manuscript."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .conditions import CONDITION_LEVELS, validate_condition_dict


class SymmetricContrastiveLoss(nn.Module):
    """Image-to-text plus text-to-image contrastive cross entropy."""

    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        self.temperature = temperature

    def forward(self, image_embedding: Tensor, text_embedding: Tensor) -> dict[str, Tensor]:
        if image_embedding.shape != text_embedding.shape:
            raise ValueError("Image and text embeddings must have identical shapes.")
        image_embedding = F.normalize(image_embedding, dim=1)
        text_embedding = F.normalize(text_embedding, dim=1)
        logits = image_embedding @ text_embedding.transpose(0, 1)
        logits = logits / self.temperature
        targets = torch.arange(logits.shape[0], device=logits.device)
        image_to_text = F.cross_entropy(logits, targets)
        text_to_image = F.cross_entropy(logits.transpose(0, 1), targets)
        return {
            "loss": image_to_text + text_to_image,
            "image_to_text": image_to_text,
            "text_to_image": text_to_image,
            "logits": logits,
        }


class HierarchicalConditionConsistencyLoss(nn.Module):
    """Cosine, KL and cross-level consistency from the manuscript equations."""

    def __init__(
        self,
        alpha: float = 0.7,
        beta: float = 0.3,
        distribution: str = "kl",
        mse_weight: float = 1.0,
        norm_weight: float = 0.0,
    ) -> None:
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1].")
        if beta < 0.0:
            raise ValueError("beta must be non-negative.")
        if distribution not in {"kl", "mse", "kl_mse"}:
            raise ValueError("distribution must be kl, mse, or kl_mse.")
        if mse_weight < 0.0 or norm_weight < 0.0:
            raise ValueError("mse_weight and norm_weight must be non-negative.")
        self.alpha = alpha
        self.beta = beta
        self.distribution = distribution
        self.mse_weight = mse_weight
        self.norm_weight = norm_weight

    def forward(
        self,
        dynamic_conditions: dict[str, Tensor],
        label_conditions: dict[str, Tensor],
        condition_masks: dict[str, Tensor] | None = None,
    ) -> dict[str, Tensor]:
        validate_condition_dict(dynamic_conditions)
        validate_condition_dict(label_conditions)
        if condition_masks is not None:
            validate_condition_dict(condition_masks)

        batch_size = dynamic_conditions["fine"].shape[0]
        masks: dict[str, Tensor] = {}
        for level in CONDITION_LEVELS:
            if condition_masks is None:
                mask = torch.ones(
                    batch_size,
                    dtype=torch.bool,
                    device=dynamic_conditions[level].device,
                )
            else:
                mask = condition_masks[level].to(
                    device=dynamic_conditions[level].device,
                    dtype=torch.bool,
                )
                if mask.ndim != 1 or mask.shape[0] != batch_size:
                    raise ValueError(
                        f"Condition mask {level!r} must have shape [B]."
                    )
            masks[level] = mask

        # Condition losses are numerically sensitive in mixed precision.  In
        # particular, evaluating cosine similarity on an all-zero, masked
        # label vector can produce a non-finite FP16 backward pass even when
        # that row is removed afterwards.  Likewise, filling excluded KL
        # logits with the minimum FP16 value leaves the backward pass exposed
        # to underflow.  Select valid rows first and carry out the loss math in
        # FP32; gradients still propagate through the casts to the model.
        dynamic_fp32 = {
            level: dynamic_conditions[level].float() for level in CONDITION_LEVELS
        }
        label_fp32 = {
            level: label_conditions[level].float() for level in CONDITION_LEVELS
        }

        zero = dynamic_fp32["fine"].sum() * 0.0
        cosine_terms: list[Tensor] = []
        for level in CONDITION_LEVELS:
            if masks[level].any():
                distance = 1.0 - F.cosine_similarity(
                    dynamic_fp32[level][masks[level]],
                    label_fp32[level][masks[level]],
                    dim=1,
                )
                cosine_terms.append(distance.mean())
        cosine = torch.stack(cosine_terms).mean() if cosine_terms else zero

        # Compute each observed fine/fused/coarse mask pattern independently.
        # This exactly excludes removed levels from the Softmax support without
        # introducing artificial -inf/min-float logits.  There are at most
        # seven non-empty patterns, so grouping adds negligible overhead.
        pattern_ids = torch.zeros(batch_size, dtype=torch.long, device=zero.device)
        for bit, level in enumerate(CONDITION_LEVELS):
            pattern_ids = pattern_ids | (masks[level].long() << bit)

        kl_terms: list[Tensor] = []
        norm_terms: list[Tensor] = []
        mse_sum = zero
        mse_elements = 0
        for pattern_id in range(1, 1 << len(CONDITION_LEVELS)):
            sample_mask = pattern_ids.eq(pattern_id)
            if not sample_mask.any():
                continue
            active_levels = [
                level
                for bit, level in enumerate(CONDITION_LEVELS)
                if pattern_id & (1 << bit)
            ]
            dynamic_valid = torch.cat(
                [dynamic_fp32[level][sample_mask] for level in active_levels],
                dim=1,
            )
            label_valid = torch.cat(
                [label_fp32[level][sample_mask] for level in active_levels],
                dim=1,
            )
            kl_terms.append(
                F.kl_div(
                    F.log_softmax(dynamic_valid, dim=1),
                    F.softmax(label_valid, dim=1),
                    reduction="none",
                ).sum(dim=1)
            )
            squared_error = (dynamic_valid - label_valid).square()
            mse_sum = mse_sum + squared_error.sum()
            mse_elements += squared_error.numel()
            norm_terms.append(
                (dynamic_valid.norm(dim=1) - label_valid.norm(dim=1)).square()
            )

        if kl_terms:
            kl = torch.cat(kl_terms).mean()
            mse = mse_sum / mse_elements
            norm = torch.cat(norm_terms).mean()
        else:
            kl = zero
            mse = zero
            norm = zero

        cross_mask = masks["coarse"] & masks["fine"]
        if cross_mask.any():
            dynamic_cross = (
                dynamic_fp32["coarse"][cross_mask]
                - dynamic_fp32["fine"][cross_mask]
            )
            label_cross = (
                label_fp32["coarse"][cross_mask]
                - label_fp32["fine"][cross_mask]
            )
            cross = (dynamic_cross - label_cross).square().mean(dim=1)
            cross = cross.mean()
        else:
            cross = zero
        if self.distribution == "kl":
            distribution_loss = kl
        elif self.distribution == "mse":
            distribution_loss = self.mse_weight * mse
        else:
            distribution_loss = kl + self.mse_weight * mse
        total = self.alpha * cosine
        total = total + (1.0 - self.alpha) * distribution_loss
        total = total + self.beta * cross + self.norm_weight * norm
        return {
            "loss": total,
            "cosine": cosine,
            "kl": kl,
            "mse": mse,
            "norm": norm,
            "cross": cross,
            "dynamic_norm": torch.cat(
                [dynamic_fp32[level] for level in CONDITION_LEVELS], dim=1
            ).norm(dim=1).mean(),
            "label_norm": torch.cat(
                [label_fp32[level] for level in CONDITION_LEVELS], dim=1
            ).norm(dim=1).mean(),
        }


class UHMFIObjective(nn.Module):
    """Combined cross-modal and condition-consistency objective."""

    def __init__(
        self,
        temperature: float = 0.1,
        condition_alpha: float = 0.7,
        condition_beta: float = 0.3,
        condition_weight: float = 1.0,
        condition_distribution: str = "kl",
        condition_mse_weight: float = 1.0,
        condition_norm_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.contrastive = SymmetricContrastiveLoss(temperature)
        self.condition = HierarchicalConditionConsistencyLoss(
            condition_alpha,
            condition_beta,
            distribution=condition_distribution,
            mse_weight=condition_mse_weight,
            norm_weight=condition_norm_weight,
        )
        self.condition_weight = condition_weight

    def forward(self, model_output: dict[str, object]) -> dict[str, Tensor]:
        image_embedding = model_output["image_embedding"]
        text_embedding = model_output["text_embedding"]
        if not isinstance(image_embedding, Tensor) or not isinstance(text_embedding, Tensor):
            raise ValueError("Both image and text embeddings are required for the objective.")

        contrastive = self.contrastive(image_embedding, text_embedding)
        label = model_output.get("label_conditions", model_output.get("manual_conditions"))
        dynamic = model_output["dynamic_conditions"]
        if label is None or self.condition_weight == 0.0:
            condition_zero = image_embedding.sum() * 0.0
            condition = {
                "loss": condition_zero,
                "cosine": condition_zero,
                "kl": condition_zero,
                "mse": condition_zero,
                "norm": condition_zero,
                "cross": condition_zero,
                "dynamic_norm": condition_zero,
                "label_norm": condition_zero,
            }
        else:
            if not isinstance(dynamic, dict) or not isinstance(label, dict):
                raise TypeError("Condition dictionaries have an invalid type.")
            masks = model_output.get("label_condition_mask")
            if masks is not None and not isinstance(masks, dict):
                raise TypeError("Condition masks have an invalid type.")
            condition = self.condition(dynamic, label, masks)

        total = contrastive["loss"] + self.condition_weight * condition["loss"]
        return {
            "loss": total,
            "contrastive": contrastive["loss"],
            "image_to_text": contrastive["image_to_text"],
            "text_to_image": contrastive["text_to_image"],
            "condition": condition["loss"],
            "condition_cosine": condition["cosine"],
            "condition_kl": condition["kl"],
            "condition_mse": condition["mse"],
            "condition_norm": condition["norm"],
            "condition_cross": condition["cross"],
            "dynamic_condition_norm": condition["dynamic_norm"],
            "label_condition_norm": condition["label_norm"],
        }
