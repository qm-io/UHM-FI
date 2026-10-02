"""Dependency-light metrics used by downstream evaluation."""

from __future__ import annotations

import math

import torch
from torch import Tensor


class RunningMean:
    def __init__(self) -> None:
        self.total = 0.0
        self.weight = 0

    def update(self, value: float, weight: int = 1) -> None:
        self.total += float(value) * int(weight)
        self.weight += int(weight)

    @property
    def value(self) -> float:
        return self.total / max(self.weight, 1)


def binary_auroc(scores: Tensor, targets: Tensor) -> float:
    scores = scores.detach().float().flatten().cpu()
    targets = targets.detach().long().flatten().cpu()
    positive = targets.eq(1)
    count_positive = int(positive.sum())
    count_negative = int(targets.numel() - count_positive)
    if count_positive == 0 or count_negative == 0:
        return float("nan")

    order = torch.argsort(scores, stable=True)
    sorted_scores = scores[order]
    ranks = torch.arange(1, scores.numel() + 1, dtype=torch.float64)
    start = 0
    while start < sorted_scores.numel():
        end = start + 1
        while end < sorted_scores.numel() and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[start:end] = ranks[start:end].mean()
        start = end
    original_ranks = torch.empty_like(ranks)
    original_ranks[order] = ranks
    positive_rank_sum = original_ranks[positive].sum()
    auc = (
        positive_rank_sum - count_positive * (count_positive + 1) / 2.0
    ) / (count_positive * count_negative)
    return float(auc)


def classification_metrics(
    logits: Tensor,
    targets: Tensor,
    num_classes: int,
) -> dict[str, float]:
    targets = targets.long().flatten()
    if num_classes == 1:
        scores = torch.sigmoid(logits.flatten())
        predictions = scores.ge(0.5).long()
        return {
            "accuracy": float(predictions.eq(targets).float().mean()),
            "auroc": binary_auroc(scores, targets),
        }

    probabilities = torch.softmax(logits, dim=1)
    predictions = probabilities.argmax(dim=1)
    aucs = [
        binary_auroc(probabilities[:, index], targets.eq(index).long())
        for index in range(num_classes)
    ]
    finite = [value for value in aucs if math.isfinite(value)]
    return {
        "accuracy": float(predictions.eq(targets).float().mean()),
        "auroc": float(sum(finite) / len(finite)) if finite else float("nan"),
    }


class SegmentationMetricAccumulator:
    """Accumulate foreground segmentation metrics across an entire phase.

    Binary Dice/IoU are computed from dataset-level foreground intersections
    and unions.  Empty target/prediction pairs therefore do not receive an
    artificial perfect score, while false-positive pixels on empty images are
    still included in the denominator.  Positive-image macro metrics and empty
    mask accuracy are reported separately for diagnosis.
    """

    def __init__(self, num_classes: int, epsilon: float = 1e-6) -> None:
        if num_classes <= 0:
            raise ValueError("num_classes must be positive.")
        self.num_classes = num_classes
        self.epsilon = epsilon
        foreground_classes = 1 if num_classes == 1 else num_classes - 1
        self.intersection = torch.zeros(foreground_classes, dtype=torch.float64)
        self.prediction_size = torch.zeros(foreground_classes, dtype=torch.float64)
        self.target_size = torch.zeros(foreground_classes, dtype=torch.float64)
        self.positive_dice_total = 0.0
        self.positive_iou_total = 0.0
        self.positive_images = 0
        self.empty_images = 0
        self.correct_empty_images = 0

    def update(self, logits: Tensor, targets: Tensor) -> dict[str, float]:
        with torch.no_grad():
            if self.num_classes == 1:
                predictions = torch.sigmoid(logits).ge(0.5)
                target_mask = targets.gt(0.5)
                intersection = (predictions & target_mask).sum(dim=(1, 2, 3)).double()
                prediction_size = predictions.sum(dim=(1, 2, 3)).double()
                target_size = target_mask.sum(dim=(1, 2, 3)).double()

                self.intersection[0] += intersection.sum().cpu()
                self.prediction_size[0] += prediction_size.sum().cpu()
                self.target_size[0] += target_size.sum().cpu()

                positive = target_size.gt(0)
                if bool(positive.any()):
                    positive_intersection = intersection[positive]
                    positive_prediction = prediction_size[positive]
                    positive_target = target_size[positive]
                    positive_union = (
                        positive_prediction + positive_target - positive_intersection
                    )
                    self.positive_dice_total += float(
                        (
                            2.0
                            * positive_intersection
                            / (positive_prediction + positive_target)
                        ).sum()
                    )
                    self.positive_iou_total += float(
                        (positive_intersection / positive_union).sum()
                    )
                    self.positive_images += int(positive.sum())

                empty = target_size.eq(0)
                if bool(empty.any()):
                    self.correct_empty_images += int(prediction_size[empty].eq(0).sum())
                    self.empty_images += int(empty.sum())
            else:
                predictions = logits.argmax(dim=1)
                if targets.ndim == 4:
                    targets = targets[:, 0]
                targets = targets.long()
                for output_index, class_index in enumerate(range(1, self.num_classes)):
                    prediction_mask = predictions.eq(class_index)
                    target_mask = targets.eq(class_index)
                    self.intersection[output_index] += (
                        (prediction_mask & target_mask).sum().double().cpu()
                    )
                    self.prediction_size[output_index] += (
                        prediction_mask.sum().double().cpu()
                    )
                    self.target_size[output_index] += target_mask.sum().double().cpu()
        return self.compute()

    def compute(self) -> dict[str, float]:
        union = self.prediction_size + self.target_size - self.intersection
        dice_denominator = self.prediction_size + self.target_size
        valid_dice = dice_denominator.gt(0)
        valid_iou = union.gt(0)
        dice_values = torch.full_like(dice_denominator, float("nan"))
        iou_values = torch.full_like(union, float("nan"))
        dice_values[valid_dice] = (
            2.0 * self.intersection[valid_dice] / dice_denominator[valid_dice]
        )
        iou_values[valid_iou] = self.intersection[valid_iou] / union[valid_iou]
        finite_dice = dice_values[torch.isfinite(dice_values)]
        finite_iou = iou_values[torch.isfinite(iou_values)]
        result = {
            "dice": (
                float(finite_dice.mean()) if finite_dice.numel() else 0.0
            ),
            "iou": float(finite_iou.mean()) if finite_iou.numel() else 0.0,
        }
        if self.num_classes == 1:
            result.update(
                {
                    "positive_dice": (
                        self.positive_dice_total / self.positive_images
                        if self.positive_images
                        else float("nan")
                    ),
                    "positive_iou": (
                        self.positive_iou_total / self.positive_images
                        if self.positive_images
                        else float("nan")
                    ),
                    "empty_mask_accuracy": (
                        self.correct_empty_images / self.empty_images
                        if self.empty_images
                        else float("nan")
                    ),
                    "positive_images": float(self.positive_images),
                    "empty_images": float(self.empty_images),
                }
            )
        return result


def segmentation_metrics(
    logits: Tensor,
    targets: Tensor,
    num_classes: int,
    epsilon: float = 1e-6,
) -> dict[str, float]:
    accumulator = SegmentationMetricAccumulator(num_classes, epsilon=epsilon)
    return accumulator.update(logits, targets)
