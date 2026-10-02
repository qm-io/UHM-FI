"""UHM-FI downstream classification and segmentation models."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .checkpoint import LoadReport, load_checkpoint, load_image_encoder_weights
from .config import DownstreamConfig, UHMFIConfig
from .vision import HierarchicallyModulatedResNet50


def _fixed_label(
    values: tuple[float, ...],
    expected: int,
    batch_size: int,
    device: torch.device,
) -> Tensor | None:
    if not values:
        return None
    if len(values) != expected:
        raise ValueError(f"Fixed label requires {expected} values, got {len(values)}.")
    return torch.tensor(values, dtype=torch.float32, device=device).view(1, -1).expand(
        batch_size, -1
    )


class UHMFIClassifier(nn.Module):
    """Linear-probe or fine-tuning classifier over the image-only UHM-FI path."""

    def __init__(self, encoder: HierarchicallyModulatedResNet50, config: DownstreamConfig) -> None:
        super().__init__()
        self.encoder = encoder
        self.config = config
        self.encoder_frozen = config.task.protocol in {"linear_probe", "frozen_encoder"}
        feature_dim = (
            2048
            if config.task.feature_source == "global_feature"
            else config.model.vision.embedding_dim
        )
        output_dim = 1 if config.task.num_classes == 1 else config.task.num_classes
        self.head = nn.Sequential(
            nn.Dropout(config.task.classifier_dropout),
            nn.Linear(feature_dim, output_dim),
        )
        if self.encoder_frozen:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

    def train(self, mode: bool = True) -> "UHMFIClassifier":
        super().train(mode)
        if self.encoder_frozen:
            # Linear probing freezes both parameters and batch-normalization
            # statistics for every UHM-FI encoder component.
            self.encoder.eval()
        return self

    def forward(self, images: Tensor) -> Tensor:
        fine = _fixed_label(
            self.config.task.fixed_fine_labels,
            self.config.model.vision.num_fine_labels,
            images.shape[0],
            images.device,
        )
        coarse = _fixed_label(
            self.config.task.fixed_coarse_labels,
            self.config.model.vision.num_coarse_labels,
            images.shape[0],
            images.device,
        )
        context = torch.no_grad() if self.encoder_frozen else torch.enable_grad()
        with context:
            output = self.encoder(images, fine, coarse)
        feature = output[self.config.task.feature_source]
        if not isinstance(feature, Tensor):
            raise TypeError("The selected classifier feature is not a Tensor.")
        return self.head(feature)


def _normalization(channels: int) -> nn.Module:
    groups = min(32, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class DecoderBlock(nn.Module):
    def __init__(self, input_channels: int, skip_channels: int, output_channels: int) -> None:
        super().__init__()
        channels = input_channels + skip_channels
        self.block = nn.Sequential(
            nn.Conv2d(channels, output_channels, kernel_size=3, padding=1, bias=False),
            _normalization(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                output_channels, output_channels, kernel_size=3, padding=1, bias=False
            ),
            _normalization(output_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, feature: Tensor, skip: Tensor) -> Tensor:
        feature = F.interpolate(
            feature, size=skip.shape[-2:], mode="bilinear", align_corners=False
        )
        return self.block(torch.cat((feature, skip), dim=1))


class UHMFIUNetDecoder(nn.Module):
    """U-Net decoder consuming all four modulated ResNet feature stages."""

    def __init__(self, decoder_channels: int, num_classes: int) -> None:
        super().__init__()
        c3 = decoder_channels * 2
        c2 = decoder_channels
        c1 = max(decoder_channels // 2, 32)
        c0 = max(decoder_channels // 4, 32)
        self.layer3 = DecoderBlock(2048, 1024, c3)
        self.layer2 = DecoderBlock(c3, 512, c2)
        self.layer1 = DecoderBlock(c2, 256, c1)
        self.stem = DecoderBlock(c1, 64, c0)
        output_channels = 1 if num_classes == 1 else num_classes
        self.head = nn.Conv2d(c0, output_channels, kernel_size=1)

    def forward(
        self,
        feature_maps: dict[str, Tensor],
        output_size: tuple[int, int],
    ) -> Tensor:
        feature = self.layer3(feature_maps["layer4"], feature_maps["layer3"])
        feature = self.layer2(feature, feature_maps["layer2"])
        feature = self.layer1(feature, feature_maps["layer1"])
        feature = self.stem(feature, feature_maps["stem"])
        logits = self.head(feature)
        return F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)


class UHMFISegmenter(nn.Module):
    """Image-only UHM-FI encoder with an explicit U-Net decoder."""

    def __init__(self, encoder: HierarchicallyModulatedResNet50, config: DownstreamConfig) -> None:
        super().__init__()
        self.encoder = encoder
        self.config = config
        self.encoder_frozen = config.task.protocol in {"linear_probe", "frozen_encoder"}
        self.decoder = UHMFIUNetDecoder(
            config.task.decoder_channels, config.task.num_classes
        )
        if self.encoder_frozen:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

    def train(self, mode: bool = True) -> "UHMFISegmenter":
        super().train(mode)
        if self.encoder_frozen:
            self.encoder.eval()
        return self

    def forward(self, images: Tensor) -> Tensor:
        fine = _fixed_label(
            self.config.task.fixed_fine_labels,
            self.config.model.vision.num_fine_labels,
            images.shape[0],
            images.device,
        )
        coarse = _fixed_label(
            self.config.task.fixed_coarse_labels,
            self.config.model.vision.num_coarse_labels,
            images.shape[0],
            images.device,
        )
        context = torch.no_grad() if self.encoder_frozen else torch.enable_grad()
        with context:
            output = self.encoder(images, fine, coarse)
        feature_maps = output["feature_maps"]
        if not isinstance(feature_maps, dict):
            raise TypeError("Image encoder did not return feature maps.")
        return self.decoder(feature_maps, tuple(images.shape[-2:]))


def build_downstream_model(
    config: DownstreamConfig,
) -> tuple[nn.Module, LoadReport | None]:
    effective_config = config
    checkpoint_model_config: UHMFIConfig | None = None
    if config.pretrained_checkpoint and config.use_checkpoint_model_config:
        checkpoint = load_checkpoint(config.pretrained_checkpoint)
        saved_config = checkpoint.get("config")
        if isinstance(saved_config, dict):
            model_data = saved_config.get("model", saved_config)
            if isinstance(model_data, dict) and "vision" in model_data:
                checkpoint_model_config = UHMFIConfig.from_dict(model_data)
                effective_config = replace(config, model=checkpoint_model_config)

    vision_config = effective_config.model.vision
    if (
        vision_config.modulation_enabled
        and not vision_config.dynamic_conditions_enabled
        and not effective_config.task.fixed_fine_labels
    ):
        raise RuntimeError(
            "This ablation checkpoint has CSAM enabled but HMCS disabled. "
            "Downstream evaluation therefore requires an explicit fixed anatomy "
            "condition (for example --fixed-condition bone on MURA)."
        )
    if config.pretrained_checkpoint:
        # A UHM-FI checkpoint fully initializes the encoder; avoid an unrelated
        # ImageNet download before loading it.
        vision_config = replace(vision_config, pretrained=False)
    encoder = HierarchicallyModulatedResNet50(vision_config)
    report = None
    if config.pretrained_checkpoint:
        report = load_image_encoder_weights(
            encoder,
            config.pretrained_checkpoint,
            strict=False,
        )
        if (
            checkpoint_model_config is not None
            and report.source_format == "uhm-fi-v1"
            and (
                report.missing_keys
                or report.unexpected_keys
                or report.skipped_shape_keys
            )
        ):
            raise RuntimeError(
                "The downstream encoder did not exactly match the architecture "
                f"saved by the ablation checkpoint:\n{report}"
            )
        if (
            report.source_format == "legacy-lightning"
            and effective_config.model.vision.modulation_variant != "legacy"
        ):
            raise RuntimeError(
                "A retained legacy checkpoint requires vision.modulation_variant: "
                "legacy. The stable modulator has different parameter shapes and must "
                "not be silently mixed with historical weights."
            )
        if report.loaded_keys == 0:
            raise RuntimeError(
                "The supplied checkpoint did not contain any compatible image-encoder weights."
            )
    if config.task.type == "classification":
        return UHMFIClassifier(encoder, effective_config), report
    return UHMFISegmenter(encoder, effective_config), report


def classification_loss(logits: Tensor, labels: Tensor, num_classes: int) -> Tensor:
    if num_classes == 1:
        return F.binary_cross_entropy_with_logits(
            logits.flatten(), labels.float().flatten()
        )
    return F.cross_entropy(logits, labels.long())


def _binary_dice_loss(logits: Tensor, targets: Tensor, epsilon: float = 1e-6) -> Tensor:
    # Run the complete binary Dice calculation in float32.  Casting only after
    # sigmoid still leaves the sigmoid itself in the active AMP dtype, which
    # can overflow/saturate during long high-resolution fine-tuning runs.
    probabilities = torch.sigmoid(logits.float())
    targets = targets.float()
    intersection = (probabilities * targets).sum(dim=(1, 2, 3))
    denominator = probabilities.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    return 1.0 - ((2.0 * intersection + epsilon) / (denominator + epsilon)).mean()


def _multiclass_dice_loss(
    logits: Tensor, targets: Tensor, num_classes: int, epsilon: float = 1e-6
) -> Tensor:
    if targets.ndim == 4:
        targets = targets[:, 0]
    probabilities = torch.softmax(logits, dim=1).float()
    one_hot = F.one_hot(targets.long(), num_classes=num_classes).permute(0, 3, 1, 2)
    one_hot = one_hot.to(dtype=probabilities.dtype)
    intersection = (probabilities * one_hot).sum(dim=(0, 2, 3))
    denominator = probabilities.sum(dim=(0, 2, 3)) + one_hot.sum(dim=(0, 2, 3))
    # Exclude background for medical lesion segmentation when possible.
    dice = (2.0 * intersection + epsilon) / (denominator + epsilon)
    return 1.0 - dice[1:].mean()


def segmentation_loss(
    logits: Tensor,
    targets: Tensor,
    num_classes: int,
    loss_name: str,
) -> Tensor:
    if num_classes == 1:
        # BCE-with-logits is deliberately evaluated in float32.  This keeps the
        # objective finite when the surrounding model forward uses FP16/BF16
        # autocast and logits become large late in training.
        bce = F.binary_cross_entropy_with_logits(logits.float(), targets.float())
        dice = _binary_dice_loss(logits, targets)
        if loss_name == "bce":
            return bce
        if loss_name == "dice":
            return dice
        return bce + dice

    if targets.ndim == 4:
        targets = targets[:, 0]
    cross_entropy = F.cross_entropy(logits.float(), targets.long())
    dice = _multiclass_dice_loss(logits, targets, num_classes)
    if loss_name == "dice":
        return dice
    if loss_name == "ce_dice":
        return cross_entropy + dice
    raise ValueError("Multiclass segmentation requires dice or ce_dice loss.")


def model_parameter_summary(model: nn.Module) -> dict[str, Any]:
    return {
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }
