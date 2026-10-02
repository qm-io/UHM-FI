"""Single-GPU mixed-precision training engines for UHM-FI.

The CLI wrappers require an explicit ``--execute`` flag before calling these
engines.  Importing this module or validating a configuration never starts a
training job.
"""

from __future__ import annotations

import json
import math
import random
import time
from collections.abc import Iterable, Mapping
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from .builder import build_model, build_objective
from .checkpoint import restore_training_state, save_checkpoint
from .config import (
    DownstreamConfig,
    OptimizerConfig,
    PretrainingConfig,
    SchedulerConfig,
    TrainerConfig,
)
from .data import build_downstream_loaders, build_pretraining_loaders
from .downstream import (
    build_downstream_model,
    classification_loss,
    model_parameter_summary,
    segmentation_loss,
)
from .metrics import (
    RunningMean,
    SegmentationMetricAccumulator,
    classification_metrics,
)


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable. Resolve the NVIDIA driver/NVML "
            "mismatch before starting GPU training."
        )
    return torch.device(requested)


def move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, Tensor):
        return value.to(device, non_blocking=device.type == "cuda")
    if isinstance(value, Mapping):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    return value


def build_optimizer(model: nn.Module, config: OptimizerConfig) -> torch.optim.Optimizer:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("The model has no trainable parameters.")
    name = config.name.lower()
    if name == "adam":
        return torch.optim.Adam(
            parameters,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(config.beta1, config.beta2),
        )
    if name == "adamw":
        return torch.optim.AdamW(
            parameters,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(config.beta1, config.beta2),
        )
    return torch.optim.SGD(
        parameters,
        lr=config.learning_rate,
        momentum=config.beta1,
        weight_decay=config.weight_decay,
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: SchedulerConfig,
    *,
    epochs: int,
    steps_per_epoch: int,
) -> Any:
    name = config.name.lower()
    if name == "none":
        return None
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config.plateau_factor,
            patience=config.plateau_patience,
            min_lr=config.minimum_learning_rate,
        )
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, epochs),
            eta_min=config.minimum_learning_rate,
        )
    if name == "warmup_cosine":
        total_steps = max(1, epochs * max(1, steps_per_epoch))
        warmup = max(0, config.warmup_steps)

        def scale(step: int) -> float:
            if warmup and step < warmup:
                return float(step + 1) / float(warmup)
            progress = (step - warmup) / max(1, total_steps - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)
    raise ValueError(f"Unsupported scheduler: {config.name}")


def _amp_dtype(config: TrainerConfig) -> torch.dtype:
    return torch.float16 if config.amp_dtype == "float16" else torch.bfloat16


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), sort_keys=True) + "\n")


def _float_losses(losses: Mapping[str, Tensor]) -> dict[str, float]:
    return {
        key: float(value.detach().float().cpu())
        for key, value in losses.items()
        if isinstance(value, Tensor) and value.ndim == 0
    }


def _accumulate_loss_sums(
    loss_sums: dict[str, Tensor],
    losses: Mapping[str, Tensor],
    batch_size: int,
) -> None:
    """Accumulate scalar losses on-device without synchronizing every batch."""

    for key, value in losses.items():
        if not isinstance(value, Tensor) or value.ndim != 0:
            continue
        detached = value.detach().float()
        total = loss_sums.get(key)
        if total is None:
            total = torch.zeros_like(detached)
            loss_sums[key] = total
        total.add_(detached * batch_size)


def _mean_loss_sums(
    loss_sums: Mapping[str, Tensor],
    processed_samples: int,
) -> dict[str, float]:
    """Synchronize accumulated scalar losses once at the end of a phase."""

    if processed_samples <= 0:
        return {}
    return {
        key: float((value / processed_samples).cpu())
        for key, value in loss_sums.items()
    }


def _duration_text(seconds: float | None) -> str | None:
    if seconds is None or not math.isfinite(seconds):
        return None
    value = max(int(round(seconds)), 0)
    hours, remainder = divmod(value, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _limited_loader_length(loader: Iterable[Any], maximum: int | None) -> int | None:
    length = len(loader) if hasattr(loader, "__len__") else None
    if maximum is None:
        return length
    return maximum if length is None else min(length, maximum)


def pretraining_parameter_summary(model: nn.Module) -> dict[str, int]:
    """Parameter accounting used in reviewer-facing ablation reports."""

    named = list(model.named_parameters())

    def count(fragment: str | None = None, *, trainable_only: bool = False) -> int:
        return sum(
            parameter.numel()
            for name, parameter in named
            if (fragment is None or fragment in name)
            and (not trainable_only or parameter.requires_grad)
        )

    return {
        "model_parameters": count(),
        "trainable_parameters": count(trainable_only=True),
        "image_encoder_parameters": count("image_encoder."),
        "text_encoder_parameters": count("text_encoder."),
        "csam_parameters": count(".modulator."),
        "hmcs_parameters": count("condition_synthesizers."),
        "label_condition_parameters": count("label_embedder."),
    }


def _initialise_wandb(
    config: PretrainingConfig | DownstreamConfig,
    output_dir: Path,
    *,
    job_type: str = "pretraining",
) -> Any | None:
    if not config.wandb.enabled or config.wandb.mode == "disabled":
        return None
    try:
        import wandb
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "W&B logging is enabled but wandb is not installed. "
            "Install the project requirements or run `python -m pip install wandb`."
        ) from exc
    return wandb.init(
        project=config.wandb.project,
        entity=config.wandb.entity,
        name=config.wandb.run_name,
        mode=config.wandb.mode,
        tags=list(config.wandb.tags),
        notes=config.wandb.notes,
        job_type=job_type,
        dir=str(output_dir),
        config=config.to_dict(),
    )


def _wandb_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {key.replace("/", "_"): value for key, value in payload.items()}


class PretrainingTrainer:
    def __init__(
        self,
        config: PretrainingConfig,
        *,
        model: nn.Module | None = None,
        objective: nn.Module | None = None,
        train_loader: Iterable[Any] | None = None,
        validation_loader: Iterable[Any] | None = None,
    ) -> None:
        self.config = config
        seed_everything(config.trainer.seed, config.trainer.deterministic)
        self.device = resolve_device(config.trainer.device)
        self.model = (model or build_model(config.model)).to(self.device)
        self.objective = (objective or build_objective(config.model)).to(self.device)
        self.parameter_summary = pretraining_parameter_summary(self.model)
        if train_loader is None or validation_loader is None:
            train_loader, validation_loader = build_pretraining_loaders(config)
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.train_batches_per_epoch = _limited_loader_length(
            self.train_loader, config.trainer.max_train_steps
        )
        self.validation_batches_per_epoch = _limited_loader_length(
            self.validation_loader, config.trainer.max_validation_steps
        )
        self._last_train_batch_rate: float | None = None
        self._last_validation_batch_rate: float | None = None
        self.sample_summary: dict[str, int] = {}
        for name, loader in (
            ("train_samples", self.train_loader),
            ("validation_samples", self.validation_loader),
        ):
            dataset = getattr(loader, "dataset", None)
            if dataset is not None and hasattr(dataset, "__len__"):
                self.sample_summary[name] = len(dataset)
        self.optimizer = build_optimizer(self.model, config.optimizer)
        self.scheduler = build_scheduler(
            self.optimizer,
            config.scheduler,
            epochs=config.trainer.epochs,
            steps_per_epoch=len(self.train_loader),  # type: ignore[arg-type]
        )
        self.amp_enabled = config.trainer.amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.amp_enabled,
            init_scale=config.trainer.amp_init_scale,
        )
        self.output_dir = Path(config.trainer.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.start_epoch = 0
        self.global_step = 0
        if config.trainer.resume:
            self.start_epoch, self.global_step, _ = restore_training_state(
                config.trainer.resume,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
            )
        self.wandb_run = _initialise_wandb(config, self.output_dir)

    def _validation_due(self, epoch: int) -> bool:
        return (epoch + 1) % self.config.trainer.validate_every_epochs == 0

    def _remaining_run_seconds(
        self,
        *,
        training: bool,
        epoch: int,
        processed_batches: int,
        current_batch_rate: float,
    ) -> float | None:
        if (
            self.train_batches_per_epoch is None
            or self.validation_batches_per_epoch is None
            or current_batch_rate <= 0
        ):
            return None
        train_rate = (
            current_batch_rate
            if training
            else self._last_train_batch_rate or current_batch_rate
        )
        validation_rate = (
            current_batch_rate
            if not training
            else self._last_validation_batch_rate or current_batch_rate
        )
        future_epochs = max(self.config.trainer.epochs - epoch - 1, 0)
        if training:
            remaining_train_batches = max(
                self.train_batches_per_epoch - processed_batches, 0
            )
            remaining_train_batches += future_epochs * self.train_batches_per_epoch
            remaining_validation_runs = int(self._validation_due(epoch))
        else:
            remaining_train_batches = future_epochs * self.train_batches_per_epoch
            remaining_validation_runs = 0
        remaining_validation_runs += sum(
            int(self._validation_due(future_epoch))
            for future_epoch in range(epoch + 1, self.config.trainer.epochs)
        )
        remaining_validation_batches = (
            remaining_validation_runs * self.validation_batches_per_epoch
        )
        if not training:
            remaining_validation_batches += max(
                self.validation_batches_per_epoch - processed_batches, 0
            )
        return (
            remaining_train_batches / train_rate
            + remaining_validation_batches / validation_rate
        )

    def _progress_fields(
        self,
        *,
        phase: str,
        epoch: int,
        processed_batches: int,
        total_batches: int | None,
        processed_samples: int,
        phase_started: float,
    ) -> dict[str, Any]:
        elapsed = max(time.monotonic() - phase_started, 1e-9)
        batch_rate = processed_batches / elapsed
        remaining_batches = (
            max(total_batches - processed_batches, 0)
            if total_batches is not None
            else None
        )
        phase_eta_seconds = (
            remaining_batches / batch_rate
            if remaining_batches is not None and batch_rate > 0
            else None
        )
        run_eta_seconds = self._remaining_run_seconds(
            training=phase == "train",
            epoch=epoch,
            processed_batches=processed_batches,
            current_batch_rate=batch_rate,
        )
        finish_time = (
            datetime.now().astimezone() + timedelta(seconds=run_eta_seconds)
            if run_eta_seconds is not None
            else None
        )
        return {
            "phase": phase,
            "phase_progress": (
                processed_batches / total_batches if total_batches else None
            ),
            "phase_batches_completed": processed_batches,
            "phase_batches_total": total_batches,
            "phase_batches_per_second": batch_rate,
            "phase_samples_per_second": processed_samples / elapsed,
            "phase_eta_seconds": phase_eta_seconds,
            "phase_eta": _duration_text(phase_eta_seconds),
            "run_eta_seconds": run_eta_seconds,
            "run_eta": _duration_text(run_eta_seconds),
            "estimated_finish_time": (
                finish_time.strftime("%Y-%m-%d %H:%M:%S %Z")
                if finish_time is not None
                else None
            ),
        }

    def _run_epoch(self, training: bool, epoch: int) -> dict[str, float]:
        loader = self.train_loader if training else self.validation_loader
        self.model.train(training)
        self.objective.train(training)
        loss_sums: dict[str, Tensor] = {}
        phase_started = time.monotonic()
        processed_batches = 0
        processed_samples = 0
        optimizer_steps = 0
        amp_overflows = 0
        accumulation = self.config.trainer.gradient_accumulation
        if training:
            self.optimizer.zero_grad(set_to_none=True)

        maximum = (
            self.config.trainer.max_train_steps
            if training
            else self.config.trainer.max_validation_steps
        )
        loader_length = len(loader) if hasattr(loader, "__len__") else None
        total_batches = _limited_loader_length(loader, maximum)
        for batch_index, batch in enumerate(loader):
            if maximum is not None and batch_index >= maximum:
                break
            batch = move_to_device(batch, self.device)
            grad_context = torch.enable_grad() if training else torch.no_grad()
            with grad_context:
                with torch.amp.autocast(
                    device_type=self.device.type,
                    dtype=_amp_dtype(self.config.trainer),
                    enabled=self.amp_enabled,
                ):
                    output = self.model(batch)
                    losses = self.objective(output)
                    loss = losses["loss"] / accumulation
                non_finite_losses = [
                    key
                    for key, value in losses.items()
                    if isinstance(value, Tensor)
                    and value.ndim == 0
                    and not torch.isfinite(value).all()
                ]
                if non_finite_losses:
                    raise FloatingPointError(
                        "Non-finite pretraining loss encountered at "
                        f"epoch={epoch}, batch={batch_index}, "
                        f"global_step={self.global_step}, "
                        f"components={non_finite_losses}."
                    )
                if training:
                    self.scaler.scale(loss).backward()

            batch_size = int(batch["images"].shape[0])
            processed_batches += 1
            processed_samples += batch_size
            _accumulate_loss_sums(loss_sums, losses, batch_size)

            should_step = training and (
                (batch_index + 1) % accumulation == 0
                or (maximum is not None and batch_index + 1 == maximum)
                or (loader_length is not None and batch_index + 1 == loader_length)
            )
            if should_step:
                self.scaler.unscale_(self.optimizer)
                parameters_with_grad = [
                    parameter
                    for parameter in self.model.parameters()
                    if parameter.grad is not None
                ]
                if not parameters_with_grad:
                    raise RuntimeError(
                        "Pretraining loss produced no parameter gradients at "
                        f"epoch={epoch}, batch={batch_index}, "
                        f"global_step={self.global_step}."
                    )
                # A single aggregate norm check avoids synchronizing once per
                # parameter tensor.  Any NaN/Inf gradient necessarily makes
                # this norm non-finite.  clip_grad_norm_ computes the norm
                # before applying the coefficient, so an invalid update is
                # still skipped below and all gradients are then cleared.
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    parameters_with_grad,
                    self.config.trainer.gradient_clip,
                    error_if_nonfinite=False,
                )
                gradient_norm_finite = bool(torch.isfinite(gradient_norm).item())

                if not gradient_norm_finite:
                    if not self.amp_enabled:
                        raise FloatingPointError(
                            "Non-finite pretraining gradient encountered at "
                            f"epoch={epoch}, batch={batch_index}, "
                            f"global_step={self.global_step}."
                        )
                    old_scale = self.scaler.get_scale()
                    # Skip explicitly and apply the normal AMP backoff.  This
                    # also covers the rare case where individually finite
                    # gradients overflow only while forming the global norm.
                    new_scale = old_scale * self.scaler.get_backoff_factor()
                    self.scaler.update(new_scale=new_scale)
                    self.optimizer.zero_grad(set_to_none=True)
                    amp_overflows += 1
                    overflow_event = {
                        "event": "amp_overflow",
                        "phase": "train",
                        "epoch": epoch,
                        "batch": batch_index,
                        "global_step": self.global_step,
                        "reason": "gradient_norm",
                        "amp_scale_before": old_scale,
                        "amp_scale_after": new_scale,
                    }
                    print(json.dumps(overflow_event, sort_keys=True), flush=True)
                    if self.wandb_run is not None:
                        self.wandb_run.log(_wandb_payload(overflow_event))
                    if not math.isfinite(new_scale) or new_scale <= 0.0:
                        raise FloatingPointError(
                            "AMP scale collapsed during pretraining at "
                            f"epoch={epoch}, batch={batch_index}, "
                            f"global_step={self.global_step}."
                        )
                else:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1
                    optimizer_steps += 1
                    if (
                        self.scheduler is not None
                        and self.config.scheduler.name == "warmup_cosine"
                    ):
                        self.scheduler.step()
                train_log_due = (
                    processed_batches <= accumulation
                    or self.global_step % self.config.trainer.log_every_steps == 0
                    or (
                        total_batches is not None
                        and processed_batches == total_batches
                    )
                )
                if train_log_due:
                    current_losses = _float_losses(losses)
                    progress = {
                        "event": "train_step",
                        "epoch": epoch,
                        "global_step": self.global_step,
                        "learning_rate": self.optimizer.param_groups[0]["lr"],
                        **self._progress_fields(
                            phase="train",
                            epoch=epoch,
                            processed_batches=processed_batches,
                            total_batches=total_batches,
                            processed_samples=processed_samples,
                            phase_started=phase_started,
                        ),
                        **{
                            f"train_step/{key}": value
                            for key, value in current_losses.items()
                        },
                    }
                    print(json.dumps(progress, sort_keys=True), flush=True)
                    if self.wandb_run is not None:
                        self.wandb_run.log(_wandb_payload(progress))

            validation_log_due = not training and (
                processed_batches == 1
                or processed_batches % self.config.trainer.log_every_steps == 0
                or (total_batches is not None and processed_batches == total_batches)
            )
            if validation_log_due:
                current_losses = _float_losses(losses)
                progress = {
                    "event": "validation_step",
                    "epoch": epoch,
                    "global_step": self.global_step,
                    **self._progress_fields(
                        phase="validation",
                        epoch=epoch,
                        processed_batches=processed_batches,
                        total_batches=total_batches,
                        processed_samples=processed_samples,
                        phase_started=phase_started,
                    ),
                    **{
                        f"validation_step/{key}": value
                        for key, value in current_losses.items()
                    },
                }
                print(json.dumps(progress, sort_keys=True), flush=True)
                if self.wandb_run is not None:
                    self.wandb_run.log(_wandb_payload(progress))

        elapsed = max(time.monotonic() - phase_started, 1e-9)
        batch_rate = processed_batches / elapsed
        if training:
            self._last_train_batch_rate = batch_rate
            if processed_batches > 0 and optimizer_steps == 0:
                raise FloatingPointError(
                    "Pretraining epoch completed without a successful optimizer "
                    f"update: epoch={epoch}, batches={processed_batches}, "
                    f"amp_overflows={amp_overflows}, "
                    f"amp_scale={self.scaler.get_scale()}."
                )
        else:
            self._last_validation_batch_rate = batch_rate
        return {
            **_mean_loss_sums(loss_sums, processed_samples),
            "duration_seconds": elapsed,
            "batches_per_second": batch_rate,
            "samples_per_second": processed_samples / elapsed,
            **(
                {
                    "optimizer_steps": float(optimizer_steps),
                    "amp_overflows": float(amp_overflows),
                }
                if training
                else {}
            ),
        }

    def fit(self) -> dict[str, Any]:
        try:
            result = self._fit()
            if self.wandb_run is not None:
                run_id = getattr(self.wandb_run, "id", None)
                run_url = getattr(self.wandb_run, "url", None)
                if run_id:
                    result["wandb_run_id"] = run_id
                if run_url:
                    result["wandb_run_url"] = run_url
                for key, value in result.items():
                    if isinstance(value, (int, float, str, bool)):
                        self.wandb_run.summary[key] = value
            with (self.output_dir / "run_summary.json").open(
                "w", encoding="utf-8"
            ) as handle:
                json.dump(result, handle, indent=2, sort_keys=True)
            return result
        finally:
            if self.wandb_run is not None:
                self.wandb_run.finish()

    def _fit(self) -> dict[str, Any]:
        best_validation = float("inf")
        epochs_without_improvement = 0
        history: list[dict[str, Any]] = []
        for epoch in range(self.start_epoch, self.config.trainer.epochs):
            train_metrics = self._run_epoch(training=True, epoch=epoch)
            validate = (epoch + 1) % self.config.trainer.validate_every_epochs == 0
            validation_metrics = (
                self._run_epoch(training=False, epoch=epoch) if validate else {}
            )
            validation_loss = validation_metrics.get("loss", train_metrics.get("loss", math.inf))

            if self.scheduler is not None:
                if self.config.scheduler.name == "plateau":
                    self.scheduler.step(validation_loss)
                elif self.config.scheduler.name == "cosine":
                    self.scheduler.step()

            metrics = {
                "epoch": epoch,
                "global_step": self.global_step,
                "learning_rate": self.optimizer.param_groups[0]["lr"],
                **{f"train/{key}": value for key, value in train_metrics.items()},
                **{f"validation/{key}": value for key, value in validation_metrics.items()},
            }
            history.append(metrics)
            _append_jsonl(self.output_dir / "metrics.jsonl", metrics)
            if self.wandb_run is not None:
                self.wandb_run.log(_wandb_payload(metrics))

            improved = validation_loss < best_validation
            if improved:
                best_validation = validation_loss
                epochs_without_improvement = 0
                save_checkpoint(
                    self.output_dir / "best.pt",
                    kind="pretraining",
                    model=self.model,
                    epoch=epoch,
                    global_step=self.global_step,
                    config=self.config.to_dict(),
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    scaler=self.scaler,
                    metrics=metrics,
                )
            else:
                epochs_without_improvement += 1

            if (epoch + 1) % self.config.trainer.save_every_epochs == 0:
                save_checkpoint(
                    self.output_dir / "last.pt",
                    kind="pretraining",
                    model=self.model,
                    epoch=epoch,
                    global_step=self.global_step,
                    config=self.config.to_dict(),
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    scaler=self.scaler,
                    metrics=metrics,
                )
            if epochs_without_improvement >= self.config.trainer.early_stopping_patience:
                break
        return {
            "status": "completed",
            "device": str(self.device),
            "best_validation_loss": best_validation,
            "epochs_ran": len(history),
            "global_step": self.global_step,
            "output_dir": str(self.output_dir),
            **self.parameter_summary,
            **self.sample_summary,
        }


class DownstreamTrainer:
    def __init__(
        self,
        config: DownstreamConfig,
        *,
        model: nn.Module | None = None,
        train_loader: Iterable[Any] | None = None,
        validation_loader: Iterable[Any] | None = None,
        test_loader: Iterable[Any] | None = None,
    ) -> None:
        self.config = config
        seed_everything(config.trainer.seed, config.trainer.deterministic)
        self.device = resolve_device(config.trainer.device)
        if model is None:
            model, self.load_report = build_downstream_model(config)
        else:
            self.load_report = None
        self.model = model.to(self.device)
        if train_loader is None or validation_loader is None or test_loader is None:
            train_loader, validation_loader, test_loader = build_downstream_loaders(config)
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.test_loader = test_loader
        self.train_batches_per_epoch = _limited_loader_length(
            self.train_loader, config.trainer.max_train_steps
        )
        self.validation_batches_per_epoch = _limited_loader_length(
            self.validation_loader, config.trainer.max_validation_steps
        )
        self.test_batches = _limited_loader_length(
            self.test_loader, config.trainer.max_validation_steps
        )
        self._last_train_batch_rate: float | None = None
        self._last_validation_batch_rate: float | None = None
        self._last_test_batch_rate: float | None = None
        self.optimizer = build_optimizer(self.model, config.optimizer)
        self.scheduler = build_scheduler(
            self.optimizer,
            config.scheduler,
            epochs=config.trainer.epochs,
            steps_per_epoch=len(self.train_loader),  # type: ignore[arg-type]
        )
        self.amp_enabled = config.trainer.amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.amp_enabled,
            init_scale=config.trainer.amp_init_scale,
        )
        self.output_dir = Path(config.trainer.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.start_epoch = 0
        self.global_step = 0
        self._resume_best_score: float | None = None
        if config.trainer.resume:
            self.start_epoch, self.global_step, resume_metrics = restore_training_state(
                config.trainer.resume,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
            )
            score_key = (
                "validation/auroc"
                if config.task.type == "classification"
                else "validation/dice"
            )
            historical_scores: list[float] = []
            candidate = resume_metrics.get(score_key)
            if isinstance(candidate, (int, float)) and math.isfinite(float(candidate)):
                historical_scores.append(float(candidate))
            history_path = self.output_dir / "metrics.jsonl"
            if history_path.exists():
                with history_path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        try:
                            record = json.loads(line)
                        except (json.JSONDecodeError, TypeError, ValueError):
                            continue
                        candidate = record.get(score_key)
                        if isinstance(candidate, (int, float)) and math.isfinite(float(candidate)):
                            historical_scores.append(float(candidate))
            if historical_scores:
                self._resume_best_score = max(historical_scores)
        self.wandb_run = _initialise_wandb(
            config,
            self.output_dir,
            job_type="downstream",
        )

    def _remaining_run_seconds(
        self,
        *,
        phase: str,
        epoch: int,
        processed_batches: int,
        current_batch_rate: float,
    ) -> float | None:
        """Estimate remaining train, validation and test wall-clock time."""

        if (
            self.train_batches_per_epoch is None
            or self.validation_batches_per_epoch is None
            or self.test_batches is None
            or current_batch_rate <= 0
        ):
            return None

        train_rate = self._last_train_batch_rate or current_batch_rate
        validation_rate = self._last_validation_batch_rate or current_batch_rate
        test_rate = self._last_test_batch_rate or validation_rate or current_batch_rate
        future_epochs = max(self.config.trainer.epochs - epoch - 1, 0)

        if phase == "train":
            remaining_train = max(
                self.train_batches_per_epoch - processed_batches, 0
            ) + future_epochs * self.train_batches_per_epoch
            remaining_validation = (future_epochs + 1) * self.validation_batches_per_epoch
            remaining_test = self.test_batches
        elif phase == "validation":
            remaining_train = future_epochs * self.train_batches_per_epoch
            remaining_validation = max(
                self.validation_batches_per_epoch - processed_batches, 0
            ) + future_epochs * self.validation_batches_per_epoch
            remaining_test = self.test_batches
        else:
            remaining_train = 0
            remaining_validation = 0
            remaining_test = max(self.test_batches - processed_batches, 0)

        return (
            remaining_train / train_rate
            + remaining_validation / validation_rate
            + remaining_test / test_rate
        )

    def _progress_fields(
        self,
        *,
        phase: str,
        epoch: int,
        processed_batches: int,
        total_batches: int | None,
        processed_samples: int,
        phase_started: float,
    ) -> dict[str, Any]:
        elapsed = max(time.monotonic() - phase_started, 1e-9)
        batch_rate = processed_batches / elapsed
        remaining_batches = (
            max(total_batches - processed_batches, 0)
            if total_batches is not None
            else None
        )
        phase_eta_seconds = (
            remaining_batches / batch_rate
            if remaining_batches is not None and batch_rate > 0
            else None
        )
        run_eta_seconds = self._remaining_run_seconds(
            phase=phase,
            epoch=epoch,
            processed_batches=processed_batches,
            current_batch_rate=batch_rate,
        )
        finish_time = (
            datetime.now().astimezone() + timedelta(seconds=run_eta_seconds)
            if run_eta_seconds is not None
            else None
        )
        return {
            "phase": phase,
            "epoch_progress": (
                (epoch + processed_batches / total_batches)
                / max(self.config.trainer.epochs, 1)
                if total_batches
                else None
            ),
            "phase_progress": (
                processed_batches / total_batches if total_batches else None
            ),
            "phase_batches_completed": processed_batches,
            "phase_batches_total": total_batches,
            "phase_batches_per_second": batch_rate,
            "phase_samples_per_second": processed_samples / elapsed,
            "phase_eta_seconds": phase_eta_seconds,
            "phase_eta": _duration_text(phase_eta_seconds),
            "run_eta_seconds": run_eta_seconds,
            "run_eta": _duration_text(run_eta_seconds),
            "estimated_finish_time": (
                finish_time.strftime("%Y-%m-%d %H:%M:%S %Z")
                if finish_time is not None
                else None
            ),
        }

    def _loss(self, logits: Tensor, batch: Mapping[str, Tensor]) -> Tensor:
        if self.config.task.type == "classification":
            return classification_loss(
                logits, batch["labels"], self.config.task.num_classes
            )
        return segmentation_loss(
            logits,
            batch["masks"],
            self.config.task.num_classes,
            self.config.task.segmentation_loss,
        )

    def _run(
        self,
        loader: Iterable[Any],
        training: bool,
        *,
        phase: str,
        epoch: int,
    ) -> dict[str, float]:
        self.model.train(training)
        loss_meter = RunningMean()
        logits_all: list[Tensor] = []
        targets_all: list[Tensor] = []
        segmentation_accumulator = (
            SegmentationMetricAccumulator(self.config.task.num_classes)
            if self.config.task.type == "segmentation"
            else None
        )
        phase_started = time.monotonic()
        processed_batches = 0
        processed_samples = 0
        accumulation = self.config.trainer.gradient_accumulation
        if training:
            self.optimizer.zero_grad(set_to_none=True)
        maximum = (
            self.config.trainer.max_train_steps
            if training
            else self.config.trainer.max_validation_steps
        )
        loader_length = len(loader) if hasattr(loader, "__len__") else None
        total_batches = _limited_loader_length(loader, maximum)
        for batch_index, batch in enumerate(loader):
            if maximum is not None and batch_index >= maximum:
                break
            batch = move_to_device(batch, self.device)
            grad_context = torch.enable_grad() if training else torch.no_grad()
            with grad_context:
                with torch.amp.autocast(
                    device_type=self.device.type,
                    dtype=_amp_dtype(self.config.trainer),
                    enabled=self.amp_enabled,
                ):
                    logits = self.model(batch["images"])
                    loss = self._loss(logits, batch)
                if not torch.isfinite(loss).all():
                    raise FloatingPointError(
                        "Non-finite downstream loss encountered at "
                        f"phase={phase}, epoch={epoch}, batch={batch_index}, "
                        f"global_step={self.global_step}."
                    )
                if training:
                    self.scaler.scale(loss / accumulation).backward()
            batch_size = int(batch["images"].shape[0])
            loss_value = float(loss.detach())
            processed_batches += 1
            processed_samples += batch_size
            loss_meter.update(loss_value, batch_size)
            target_key = "labels" if self.config.task.type == "classification" else "masks"
            batch_metrics: dict[str, float] = {}
            if self.config.task.type == "classification":
                logits_all.append(logits.detach().float().cpu())
                targets_all.append(batch[target_key].detach().cpu())
            else:
                assert segmentation_accumulator is not None
                # ``update`` returns the running phase metrics.  The final
                # Dice/IoU therefore aggregate foreground pixels across all
                # batches instead of averaging batch-level empty-mask scores.
                batch_metrics = segmentation_accumulator.update(
                    logits.detach(), batch[target_key]
                )

            should_step = training and (
                (batch_index + 1) % accumulation == 0
                or (maximum is not None and batch_index + 1 == maximum)
                or (loader_length is not None and batch_index + 1 == loader_length)
            )
            amp_overflow = False
            if should_step:
                self.scaler.unscale_(self.optimizer)
                gradients_finite = all(
                    torch.isfinite(parameter.grad).all()
                    for parameter in self.model.parameters()
                    if parameter.grad is not None
                )
                if not gradients_finite:
                    if not self.amp_enabled:
                        raise FloatingPointError(
                            "Non-finite downstream gradient encountered at "
                            f"phase={phase}, epoch={epoch}, batch={batch_index}, "
                            f"global_step={self.global_step}."
                        )
                    # This is the normal AMP overflow path: GradScaler has
                    # recorded the non-finite gradient during unscale_, so
                    # ``step`` skips the optimizer update and ``update`` backs
                    # off the scale.  Treating it as a fatal error prevents
                    # valid ablation runs from recovering from a rare FP16
                    # overflow late in training.
                    old_scale = self.scaler.get_scale()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    new_scale = self.scaler.get_scale()
                    self.optimizer.zero_grad(set_to_none=True)
                    amp_overflow = True
                    overflow_event = {
                        "event": "amp_overflow",
                        "phase": phase,
                        "epoch": epoch,
                        "batch": batch_index,
                        "global_step": self.global_step,
                        "amp_scale_before": old_scale,
                        "amp_scale_after": new_scale,
                    }
                    print(json.dumps(overflow_event, sort_keys=True), flush=True)
                    if self.wandb_run is not None:
                        self.wandb_run.log(_wandb_payload(overflow_event))
                else:
                    try:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.trainer.gradient_clip,
                            error_if_nonfinite=True,
                        )
                    except RuntimeError as error:
                        if not self.amp_enabled:
                            raise FloatingPointError(
                                "Non-finite downstream gradient encountered at "
                                f"phase={phase}, epoch={epoch}, batch={batch_index}, "
                                f"global_step={self.global_step}."
                            ) from error
                        # A finite element-wise gradient can still produce an
                        # overflowing FP32 norm.  Let GradScaler handle this
                        # the same way as an element-wise overflow.
                        old_scale = self.scaler.get_scale()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        new_scale = self.scaler.get_scale()
                        self.optimizer.zero_grad(set_to_none=True)
                        amp_overflow = True
                        overflow_event = {
                            "event": "amp_overflow",
                            "phase": phase,
                            "epoch": epoch,
                            "batch": batch_index,
                            "global_step": self.global_step,
                            "amp_scale_before": old_scale,
                            "amp_scale_after": new_scale,
                            "reason": "gradient_norm",
                        }
                        print(json.dumps(overflow_event, sort_keys=True), flush=True)
                        if self.wandb_run is not None:
                            self.wandb_run.log(_wandb_payload(overflow_event))
                    else:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad(set_to_none=True)
                        self.global_step += 1
                        if self.scheduler is not None and self.config.scheduler.name == "warmup_cosine":
                            self.scheduler.step()

            log_due = (
                training
                and should_step
                and (
                    processed_batches <= accumulation
                    or self.global_step % self.config.trainer.log_every_steps == 0
                    or (
                        total_batches is not None
                        and processed_batches == total_batches
                    )
                )
            ) or (
                not training
                and (
                    processed_batches == 1
                    or processed_batches % self.config.trainer.log_every_steps == 0
                    or (
                        total_batches is not None
                        and processed_batches == total_batches
                    )
                )
            )
            if log_due:
                progress = {
                    "event": f"{phase}_step",
                    "epoch": epoch,
                    "epochs_total": self.config.trainer.epochs,
                    "global_step": self.global_step,
                    "learning_rate": self.optimizer.param_groups[0]["lr"],
                    **self._progress_fields(
                        phase=phase,
                        epoch=epoch,
                        processed_batches=processed_batches,
                        total_batches=total_batches,
                        processed_samples=processed_samples,
                        phase_started=phase_started,
                    ),
                    f"{phase}_step/loss": loss_value,
                    **{
                        f"{phase}_step/{key}": value
                        for key, value in batch_metrics.items()
                    },
                    **({"amp_overflow": True} if amp_overflow else {}),
                }
                print(json.dumps(progress, sort_keys=True), flush=True)
                if self.wandb_run is not None:
                    self.wandb_run.log(_wandb_payload(progress))

        elapsed = max(time.monotonic() - phase_started, 1e-9)
        batch_rate = processed_batches / elapsed
        if phase == "train":
            self._last_train_batch_rate = batch_rate
        elif phase == "validation":
            self._last_validation_batch_rate = batch_rate
        else:
            self._last_test_batch_rate = batch_rate
        if self.config.task.type == "classification":
            logits_tensor = torch.cat(logits_all)
            targets_tensor = torch.cat(targets_all)
            metrics = classification_metrics(
                logits_tensor, targets_tensor, self.config.task.num_classes
            )
        else:
            assert segmentation_accumulator is not None
            metrics = segmentation_accumulator.compute()
        return {
            "loss": loss_meter.value,
            **metrics,
            "duration_seconds": elapsed,
            "batches_per_second": batch_rate,
            "samples_per_second": processed_samples / elapsed,
        }

    def fit(self) -> dict[str, Any]:
        try:
            result = self._fit()
            if self.wandb_run is not None:
                run_id = getattr(self.wandb_run, "id", None)
                run_url = getattr(self.wandb_run, "url", None)
                if run_id:
                    result["wandb_run_id"] = run_id
                if run_url:
                    result["wandb_run_url"] = run_url
                for key, value in result.items():
                    if isinstance(value, (int, float, str, bool)):
                        self.wandb_run.summary[key] = value
                result_path = self.output_dir / "test_metrics.json"
                if result_path.exists():
                    with result_path.open("w", encoding="utf-8") as handle:
                        json.dump(result, handle, indent=2, sort_keys=True)
            return result
        except KeyboardInterrupt:
            if self.wandb_run is not None:
                self.wandb_run.log(
                    {
                        "event": "interrupted",
                        "resume_from": str(self.output_dir / "last.pt"),
                    }
                )
            print(
                json.dumps(
                    {
                        "event": "interrupted",
                        "message": "Received SIGINT; the last completed epoch remains resumable.",
                        "resume_from": str(self.output_dir / "last.pt"),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            raise
        finally:
            if self.wandb_run is not None:
                self.wandb_run.finish()

    def _fit(self) -> dict[str, Any]:
        best_score = (
            self._resume_best_score
            if self._resume_best_score is not None
            else -float("inf")
        )
        epochs_without_improvement = 0
        for epoch in range(self.start_epoch, self.config.trainer.epochs):
            train_metrics = self._run(
                self.train_loader,
                training=True,
                phase="train",
                epoch=epoch,
            )
            validation_metrics = self._run(
                self.validation_loader,
                training=False,
                phase="validation",
                epoch=epoch,
            )
            if self.scheduler is not None:
                if self.config.scheduler.name == "plateau":
                    self.scheduler.step(validation_metrics["loss"])
                elif self.config.scheduler.name == "cosine":
                    self.scheduler.step()
            score_name = "auroc" if self.config.task.type == "classification" else "dice"
            score = validation_metrics[score_name]
            metrics = {
                "epoch": epoch,
                "global_step": self.global_step,
                "learning_rate": self.optimizer.param_groups[0]["lr"],
                **{f"train/{key}": value for key, value in train_metrics.items()},
                **{f"validation/{key}": value for key, value in validation_metrics.items()},
            }
            _append_jsonl(self.output_dir / "metrics.jsonl", metrics)
            if self.wandb_run is not None:
                self.wandb_run.log(_wandb_payload(metrics))
            if math.isfinite(score) and score > best_score:
                best_score = score
                epochs_without_improvement = 0
                save_checkpoint(
                    self.output_dir / "best.pt",
                    kind=self.config.task.type,
                    model=self.model,
                    epoch=epoch,
                    global_step=self.global_step,
                    config=self.config.to_dict(),
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    scaler=self.scaler,
                    metrics=metrics,
                )
            else:
                epochs_without_improvement += 1
            if (epoch + 1) % self.config.trainer.save_every_epochs == 0:
                save_checkpoint(
                    self.output_dir / "last.pt",
                    kind=self.config.task.type,
                    model=self.model,
                    epoch=epoch,
                    global_step=self.global_step,
                    config=self.config.to_dict(),
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    scaler=self.scaler,
                    metrics=metrics,
                )
            if epochs_without_improvement >= self.config.trainer.early_stopping_patience:
                break

        best_path = self.output_dir / "best.pt"
        if best_path.exists():
            checkpoint = torch.load(best_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint["model_state"])
        test_metrics = self._run(
            self.test_loader,
            training=False,
            phase="test",
            epoch=max(self.config.trainer.epochs - 1, 0),
        )
        result = {
            "status": "completed",
            "device": str(self.device),
            "task": self.config.task.type,
            "protocol": self.config.task.protocol,
            "best_validation_score": best_score,
            "test": test_metrics,
            "output_dir": str(self.output_dir),
            **model_parameter_summary(self.model),
        }
        if self.load_report is not None:
            result["checkpoint_load"] = self.load_report.to_dict()
        with (self.output_dir / "test_metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
        return result


def run_pretraining(config: PretrainingConfig) -> dict[str, Any]:
    return PretrainingTrainer(config).fit()


def run_downstream(config: DownstreamConfig) -> dict[str, Any]:
    return DownstreamTrainer(config).fit()
