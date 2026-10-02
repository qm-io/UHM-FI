"""Channel-spatial adaptive modulation bottlenecks for the ResNet-50 encoder."""

from __future__ import annotations

import copy

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ChannelSpatialAdaptiveModulation(nn.Module):
    """Condition-dependent channel scaling followed by a spatial offset.

    The final layers are zero-initialized so every converted bottleneck starts
    as the original ResNet bottleneck: channel scale is one and spatial offset
    is zero.
    """

    def __init__(
        self,
        channels: int,
        condition_dim: int,
        variant: str = "stable",
        mode: str = "channel_spatial",
    ) -> None:
        super().__init__()
        if variant not in {"stable", "legacy"}:
            raise ValueError("variant must be 'stable' or 'legacy'.")
        if mode not in {"channel_spatial", "channel_only", "spatial_only"}:
            raise ValueError(
                "mode must be channel_spatial, channel_only, or spatial_only."
            )
        self.variant = variant
        self.mode = mode
        use_channel = mode in {"channel_spatial", "channel_only"}
        use_spatial = mode in {"channel_spatial", "spatial_only"}
        channel_hidden = max(channels // 8, 16)
        self.channel_scale: nn.Module | None = None
        self.spatial_condition: nn.Module | None = None
        self.spatial_offset: nn.Module | None = None
        if variant == "legacy":
            # Retained-prototype topology: sigmoid channel scaling followed by
            # spatially varying gamma and beta maps generated from the condition.
            spatial_hidden = max(channels // 2, 8)
            if use_channel:
                self.channel_scale = nn.Sequential(
                    nn.Linear(condition_dim, channel_hidden),
                    nn.ReLU(inplace=True),
                    nn.Linear(channel_hidden, channels),
                    nn.Sigmoid(),
                )
            if use_spatial:
                self.spatial_condition = nn.Identity()
                self.spatial_offset = nn.Sequential(
                    nn.Conv2d(
                        condition_dim, spatial_hidden, kernel_size=3, padding=1
                    ),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        spatial_hidden, channels * 2, kernel_size=3, padding=1
                    ),
                )
                nn.init.kaiming_normal_(self.spatial_offset[0].weight, mode="fan_out")
                nn.init.constant_(self.spatial_offset[-1].weight, 0.1)
        else:
            # Stable identity-initialized version used for new training runs.
            spatial_hidden = min(max(channels // 16, 8), 32)
            if use_channel:
                self.channel_scale = nn.Sequential(
                    nn.Linear(condition_dim, channel_hidden),
                    nn.GELU(),
                    nn.Linear(channel_hidden, channels),
                )
                nn.init.zeros_(self.channel_scale[-1].weight)
                nn.init.zeros_(self.channel_scale[-1].bias)
            if use_spatial:
                self.spatial_condition = nn.Linear(condition_dim, spatial_hidden)
                self.spatial_offset = nn.Sequential(
                    nn.Conv2d(
                        spatial_hidden + 2,
                        spatial_hidden,
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.GELU(),
                    nn.Conv2d(spatial_hidden, channels, kernel_size=1),
                )
                nn.init.zeros_(self.spatial_offset[-1].weight)
                nn.init.zeros_(self.spatial_offset[-1].bias)

    @staticmethod
    def _coordinate_grid(feature: Tensor) -> Tensor:
        batch, _, height, width = feature.shape
        y = torch.linspace(-1.0, 1.0, height, device=feature.device, dtype=feature.dtype)
        x = torch.linspace(-1.0, 1.0, width, device=feature.device, dtype=feature.dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        grid = torch.stack((xx, yy), dim=0).unsqueeze(0)
        return grid.expand(batch, -1, -1, -1)

    def forward(self, feature: Tensor, condition: Tensor) -> Tensor:
        if feature.ndim != 4 or condition.ndim != 2:
            raise ValueError("Expected feature [B,C,H,W] and condition [B,D].")
        if feature.shape[0] != condition.shape[0]:
            raise ValueError("Feature and condition batch sizes must match.")

        batch, channels, height, width = feature.shape
        output = feature
        if self.variant == "legacy":
            if self.channel_scale is not None:
                scale = self.channel_scale(condition).view(batch, channels, 1, 1)
                output = output * scale
            if self.spatial_offset is None:
                return output
            expanded = condition.view(batch, -1, 1, 1).expand(
                -1, -1, height, width
            )
            gamma, beta = torch.chunk(self.spatial_offset(expanded), 2, dim=1)
            return output * (1.0 + gamma) + beta

        if self.channel_scale is not None:
            scale = 1.0 + torch.tanh(self.channel_scale(condition))
            scale = scale.view(batch, channels, 1, 1)
            output = output * scale
        if self.spatial_condition is None or self.spatial_offset is None:
            return output
        spatial_condition = self.spatial_condition(condition)
        spatial_condition = spatial_condition.view(batch, -1, 1, 1)
        spatial_condition = spatial_condition.expand(-1, -1, height, width)
        offset_input = torch.cat(
            (spatial_condition, self._coordinate_grid(feature)), dim=1
        )
        offset = self.spatial_offset(offset_input)
        return output + offset


class ModulatedBottleneck(nn.Module):
    """A torchvision ResNet bottleneck with modulation after its 3x3 conv."""

    def __init__(
        self,
        source_block: nn.Module,
        condition_dim: int,
        modulation_variant: str = "stable",
        modulation_mode: str = "channel_spatial",
    ) -> None:
        super().__init__()
        self.conv1 = copy.deepcopy(source_block.conv1)
        self.bn1 = copy.deepcopy(source_block.bn1)
        self.conv2 = copy.deepcopy(source_block.conv2)
        self.bn2 = copy.deepcopy(source_block.bn2)
        self.conv3 = copy.deepcopy(source_block.conv3)
        self.bn3 = copy.deepcopy(source_block.bn3)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = copy.deepcopy(source_block.downsample)
        self.stride = source_block.stride
        self.modulator = ChannelSpatialAdaptiveModulation(
            channels=self.conv2.out_channels,
            condition_dim=condition_dim,
            variant=modulation_variant,
            mode=modulation_mode,
        )

    def forward(self, feature: Tensor, condition: Tensor) -> Tensor:
        identity = feature

        output = self.conv1(feature)
        output = self.bn1(output)
        output = self.relu(output)

        output = self.conv2(output)
        output = self.bn2(output)
        output = self.relu(output)
        output = self.modulator(output, condition)

        output = self.conv3(output)
        output = self.bn3(output)

        if self.downsample is not None:
            identity = self.downsample(feature)

        output += identity
        return self.relu(output)


class ConditionalStage(nn.Module):
    """Apply one hierarchical condition to every bottleneck in a ResNet stage."""

    def __init__(
        self,
        source_stage: nn.Sequential,
        condition_dim: int,
        modulation_variant: str = "stable",
        modulation_mode: str = "channel_spatial",
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            ModulatedBottleneck(
                block,
                condition_dim,
                modulation_variant,
                modulation_mode,
            )
            for block in source_stage
        )

    def forward(self, feature: Tensor, condition: Tensor) -> Tensor:
        for block in self.blocks:
            feature = block(feature, condition)
        return feature


class UnconditionalStage(nn.Module):
    """A copied torchvision ResNet stage with the conditional API."""

    def __init__(self, source_stage: nn.Sequential) -> None:
        super().__init__()
        self.stage = copy.deepcopy(source_stage)

    def forward(self, feature: Tensor, condition: Tensor) -> Tensor:
        del condition
        return self.stage(feature)
