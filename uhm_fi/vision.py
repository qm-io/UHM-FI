"""Hierarchically modulated ResNet-50 image encoder."""

from __future__ import annotations

import copy

import torch
from torch import Tensor, nn
from torchvision.models import ResNet50_Weights, resnet50

from .conditions import (
    CONDITION_LEVELS,
    ConditionSynthesizer,
    HierarchicalLabelEmbedder,
    merge_label_and_dynamic_conditions,
)
from .config import VisionConfig
from .modulation import ConditionalStage, UnconditionalStage


class HierarchicallyModulatedResNet50(nn.Module):
    """ResNet-50 with fine/fused/coarse modulation at stages 2/3/4."""

    stage_channels = {
        "layer1": 256,
        "layer2": 512,
        "layer3": 1024,
        "layer4": 2048,
    }

    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        if config.backbone != "resnet50":
            raise ValueError("UHM-FI currently reproduces the paper's ResNet-50 only.")
        self.config = config

        weights = ResNet50_Weights.IMAGENET1K_V2 if config.pretrained else None
        backbone = resnet50(weights=weights)

        # The stem and first residual stage remain unmodulated, as specified in
        # the manuscript to preserve generic ImageNet low-level features.
        self.conv1 = copy.deepcopy(backbone.conv1)
        self.bn1 = copy.deepcopy(backbone.bn1)
        self.relu = copy.deepcopy(backbone.relu)
        self.maxpool = copy.deepcopy(backbone.maxpool)
        self.layer1 = copy.deepcopy(backbone.layer1)

        stage_type = ConditionalStage if config.modulation_enabled else UnconditionalStage
        if config.modulation_enabled:
            self.layer2 = stage_type(
                backbone.layer2,
                config.condition_dim,
                config.modulation_variant,
                config.modulation_mode,
            )
            self.layer3 = stage_type(
                backbone.layer3,
                config.condition_dim,
                config.modulation_variant,
                config.modulation_mode,
            )
            self.layer4 = stage_type(
                backbone.layer4,
                config.condition_dim,
                config.modulation_variant,
                config.modulation_mode,
            )
        else:
            self.layer2 = stage_type(backbone.layer2)
            self.layer3 = stage_type(backbone.layer3)
            self.layer4 = stage_type(backbone.layer4)

        self.label_embedder = HierarchicalLabelEmbedder(
            num_fine_labels=config.num_fine_labels,
            num_coarse_labels=config.num_coarse_labels,
            condition_dim=config.condition_dim,
            hidden_dim=config.condition_hidden_dim,
            dropout=config.condition_dropout,
        )
        if config.dynamic_conditions_enabled:
            self.condition_synthesizers = nn.ModuleDict(
                {
                    "fine": ConditionSynthesizer(
                        self.stage_channels["layer1"],
                        config.condition_dim,
                        config.condition_hidden_dim,
                    ),
                    "fused": ConditionSynthesizer(
                        self.stage_channels["layer2"],
                        config.condition_dim,
                        config.condition_hidden_dim,
                    ),
                    "coarse": ConditionSynthesizer(
                        self.stage_channels["layer3"],
                        config.condition_dim,
                        config.condition_hidden_dim,
                    ),
                }
            )
        else:
            # A true HMCS ablation should not retain dormant trainable HMCS
            # parameters in its checkpoint or parameter-overhead accounting.
            self.condition_synthesizers = nn.ModuleDict()

        if config.condition_merge == "learned":
            initial_probability = min(max(config.condition_mix_init, 1e-6), 1.0 - 1e-6)
            initial_logit = torch.logit(torch.tensor(initial_probability))
            self.condition_mix_weights = nn.ParameterDict(
                {
                    level: nn.Parameter(initial_logit.clone())
                    for level in CONDITION_LEVELS
                }
            )
        else:
            self.condition_mix_weights = nn.ParameterDict()

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.projection_head = nn.Sequential(
            nn.Linear(self.stage_channels["layer4"], self.stage_channels["layer4"]),
            nn.ReLU(inplace=True),
            nn.Linear(self.stage_channels["layer4"], config.embedding_dim),
        )

    def _label_conditions(
        self,
        fine_labels: Tensor | None,
        coarse_labels: Tensor | None,
    ) -> tuple[dict[str, Tensor] | None, dict[str, Tensor] | None]:
        if fine_labels is None and coarse_labels is None:
            return None, None
        if fine_labels is None or coarse_labels is None:
            raise ValueError("fine_labels and coarse_labels must be provided together.")
        fine_available = fine_labels.abs().sum(dim=1).gt(0)
        coarse_available = coarse_labels.abs().sum(dim=1).gt(0)
        masks = {
            "fine": fine_available,
            "fused": fine_available | coarse_available,
            "coarse": coarse_available,
        }
        return self.label_embedder(fine_labels, coarse_labels), masks

    def _dynamic_condition(self, level: str, feature: Tensor) -> Tensor:
        if self.config.dynamic_conditions_enabled:
            return self.condition_synthesizers[level](feature)
        return feature.new_zeros((feature.shape[0], self.config.condition_dim))

    def _applied_condition(
        self,
        level: str,
        dynamic: Tensor,
        label_conditions: dict[str, Tensor] | None,
        label_condition_mask: dict[str, Tensor] | None,
    ) -> Tensor:
        label = None if label_conditions is None else label_conditions[level]
        weight = (
            self.condition_mix_weights[level]
            if self.config.condition_merge == "learned"
            else None
        )
        merged = merge_label_and_dynamic_conditions(
            dynamic,
            label,
            self.config.condition_merge,
            label_weight=weight,
        )
        if label is None or label_condition_mask is None:
            return merged
        mask = label_condition_mask[level].view(-1, 1)
        # A deliberately removed fine/coarse label should mean "dynamic only"
        # for that sample and level, not a learned projection of an all-zero
        # vector.  This gives label ablations an exact, auditable meaning.
        return torch.where(mask, merged, dynamic)

    def forward(
        self,
        images: Tensor,
        fine_labels: Tensor | None = None,
        coarse_labels: Tensor | None = None,
    ) -> dict[str, object]:
        label_conditions, label_condition_mask = self._label_conditions(
            fine_labels, coarse_labels
        )

        stem = self.conv1(images)
        stem = self.bn1(stem)
        stem = self.relu(stem)
        feature = self.maxpool(stem)

        layer1 = self.layer1(feature)
        dynamic_fine = self._dynamic_condition("fine", layer1)
        applied_fine = self._applied_condition(
            "fine", dynamic_fine, label_conditions, label_condition_mask
        )

        layer2 = self.layer2(layer1, applied_fine)
        dynamic_fused = self._dynamic_condition("fused", layer2)
        applied_fused = self._applied_condition(
            "fused", dynamic_fused, label_conditions, label_condition_mask
        )

        layer3 = self.layer3(layer2, applied_fused)
        dynamic_coarse = self._dynamic_condition("coarse", layer3)
        applied_coarse = self._applied_condition(
            "coarse", dynamic_coarse, label_conditions, label_condition_mask
        )

        layer4 = self.layer4(layer3, applied_coarse)
        global_feature = self.avgpool(layer4).flatten(1)
        image_embedding = self.projection_head(global_feature)

        dynamic = {
            "fine": dynamic_fine,
            "fused": dynamic_fused,
            "coarse": dynamic_coarse,
        }
        applied = {
            "fine": applied_fine,
            "fused": applied_fused,
            "coarse": applied_coarse,
        }
        assert tuple(dynamic) == CONDITION_LEVELS
        mix_weights = {
            level: torch.sigmoid(self.condition_mix_weights[level])
            for level in self.condition_mix_weights
        }

        return {
            "image_embedding": image_embedding,
            "global_feature": global_feature,
            "feature_maps": {
                "stem": stem,
                "layer1": layer1,
                "layer2": layer2,
                "layer3": layer3,
                "layer4": layer4,
            },
            "label_conditions": label_conditions,
            "label_condition_mask": label_condition_mask,
            # Compatibility alias for the retained prototype.
            "manual_conditions": label_conditions,
            "dynamic_conditions": dynamic,
            "applied_conditions": applied,
            "condition_mix_weights": mix_weights,
        }
