"""Typed configuration for UHM-FI pre-training and downstream evaluation.

The manuscript specifies the high-level topology but omits a number of
implementation details.  Every such choice is represented explicitly here so
that experiments can be reproduced from a saved YAML file and checkpoint.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping


def _tuple(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)


def _load_yaml(path: str | Path) -> tuple[dict[str, Any], Path]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("PyYAML is required to load YAML configuration files.") from exc

    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Configuration root must be a mapping: {config_path}")
    return data, config_path


def _resolve_relative(path: str | None, config_path: Path) -> str | None:
    if path in {None, ""}:
        return path
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = config_path.parent / candidate
    return str(candidate.resolve())


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)  # type: ignore[arg-type]
        else:
            merged[key] = value
    return merged


@dataclass
class VisionConfig:
    backbone: str = "resnet50"
    pretrained: bool = True
    num_fine_labels: int = 10
    num_coarse_labels: int = 4
    condition_dim: int = 256
    condition_hidden_dim: int = 512
    condition_dropout: float = 0.1
    embedding_dim: int = 512
    # Goal-aligned default: both label-derived and dynamic conditions guide
    # pre-training, while their mean preserves the dynamic-only inference
    # scale when consistency training makes the two conditions agree.
    condition_merge: str = "mean"
    condition_mix_init: float = 0.5
    modulation_enabled: bool = True
    dynamic_conditions_enabled: bool = True
    modulation_variant: str = "stable"
    # CSAM branch ablations.  ``channel_spatial`` is the complete module from
    # the manuscript; the other values retain only one adaptive branch.
    modulation_mode: str = "channel_spatial"


@dataclass
class TextConfig:
    backend: str = "bioclinicalbert"
    model_name: str = "emilyalsentzer/Bio_ClinicalBERT"
    pretrained: bool = True
    local_files_only: bool = False
    freeze_layers: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
    vocab_size: int = 30_522
    max_length: int = 128
    hidden_dim: int = 256
    transformer_layers: int = 2
    attention_heads: int = 8
    dropout: float = 0.1


@dataclass
class InterleavingConfig:
    enabled: bool = True
    ratio: float = 0.25
    strategy: str = "mutual_information"
    training_only: bool = True
    detach_selection: bool = True


@dataclass
class LossConfig:
    temperature: float = 0.1
    condition_alpha: float = 0.7
    condition_beta: float = 0.3
    condition_weight: float = 1.0
    # kl reproduces the manuscript.  mse and kl_mse support the reviewer
    # requested continuous-feature comparison without changing trainer code.
    condition_distribution: str = "kl"
    condition_mse_weight: float = 1.0
    condition_norm_weight: float = 0.0


@dataclass
class UHMFIConfig:
    vision: VisionConfig = field(default_factory=VisionConfig)
    text: TextConfig = field(default_factory=TextConfig)
    interleaving: InterleavingConfig = field(default_factory=InterleavingConfig)
    loss: LossConfig = field(default_factory=LossConfig)

    def validate(self) -> None:
        if self.vision.backbone != "resnet50":
            raise ValueError("The manuscript architecture requires a ResNet-50 backbone.")
        if self.vision.condition_dim <= 0 or self.vision.embedding_dim <= 0:
            raise ValueError("Condition and embedding dimensions must be positive.")
        if self.vision.condition_merge not in {
            "learned",
            "sum",
            "mean",
            "label",
            "dynamic",
        }:
            raise ValueError("Unsupported condition_merge mode.")
        if not 0.0 <= self.vision.condition_mix_init <= 1.0:
            raise ValueError("condition_mix_init must be in [0, 1].")
        if self.vision.modulation_variant not in {"stable", "legacy"}:
            raise ValueError("modulation_variant must be 'stable' or 'legacy'.")
        if self.vision.modulation_mode not in {
            "channel_spatial",
            "channel_only",
            "spatial_only",
        }:
            raise ValueError(
                "modulation_mode must be channel_spatial, channel_only, or "
                "spatial_only. Use modulation_enabled: false for no CSAM."
            )
        if not 0.0 <= self.vision.condition_dropout < 1.0:
            raise ValueError("condition_dropout must be in [0, 1).")
        if not 0.0 <= self.interleaving.ratio <= 1.0:
            raise ValueError("Feature-interleaving ratio must be in [0, 1].")
        if self.interleaving.strategy not in {
            "mutual_information",
            "random",
            "fixed",
        }:
            raise ValueError("Unsupported feature-interleaving strategy.")
        if self.text.backend not in {"bioclinicalbert", "tiny"}:
            raise ValueError("text.backend must be 'bioclinicalbert' or 'tiny'.")
        if self.text.hidden_dim % self.text.attention_heads != 0:
            raise ValueError("text.hidden_dim must be divisible by attention_heads.")
        if self.loss.temperature <= 0:
            raise ValueError("temperature must be positive.")
        if not 0.0 <= self.loss.condition_alpha <= 1.0:
            raise ValueError("condition_alpha must be in [0, 1].")
        if self.loss.condition_beta < 0 or self.loss.condition_weight < 0:
            raise ValueError("Condition-loss weights must be non-negative.")
        if self.loss.condition_distribution not in {"kl", "mse", "kl_mse"}:
            raise ValueError("condition_distribution must be kl, mse, or kl_mse.")
        if self.loss.condition_mse_weight < 0 or self.loss.condition_norm_weight < 0:
            raise ValueError("Condition MSE and norm weights must be non-negative.")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "UHMFIConfig":
        text_data = dict(data.get("text", {}))
        if "freeze_layers" in text_data:
            text_data["freeze_layers"] = tuple(int(v) for v in _tuple(text_data["freeze_layers"]))
        config = cls(
            vision=VisionConfig(**dict(data.get("vision", {}))),
            text=TextConfig(**text_data),
            interleaving=InterleavingConfig(**dict(data.get("interleaving", {}))),
            loss=LossConfig(**dict(data.get("loss", {}))),
        )
        config.validate()
        return config

    @classmethod
    def from_yaml(cls, path: str | Path) -> "UHMFIConfig":
        data, _ = _load_yaml(path)
        if "model" in data:
            data = dict(data["model"])
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ImageTransformConfig:
    resize: int = 256
    crop_size: int = 224
    horizontal_flip: float = 0.5
    normalize: str = "half"


@dataclass
class LoaderConfig:
    batch_size: int = 64
    num_workers: int = 8
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2


@dataclass
class OptimizerConfig:
    name: str = "adamw"
    learning_rate: float = 5e-5
    weight_decay: float = 1e-6
    beta1: float = 0.9
    beta2: float = 0.999


@dataclass
class SchedulerConfig:
    name: str = "plateau"
    warmup_steps: int = 0
    minimum_learning_rate: float = 0.0
    plateau_factor: float = 0.5
    plateau_patience: int = 5


@dataclass
class TrainerConfig:
    output_dir: str = "output/pretrain"
    epochs: int = 50
    device: str = "cuda"
    seed: int = 42
    deterministic: bool = True
    amp: bool = True
    amp_dtype: str = "float16"
    amp_init_scale: float = 65536.0
    gradient_clip: float = 0.25
    gradient_accumulation: int = 1
    log_every_steps: int = 50
    validate_every_epochs: int = 1
    save_every_epochs: int = 1
    early_stopping_patience: int = 10
    resume: str | None = None
    max_train_steps: int | None = None
    max_validation_steps: int | None = None


@dataclass
class WandbConfig:
    enabled: bool = False
    project: str = "uhm-fi"
    entity: str | None = None
    run_name: str | None = None
    mode: str = "online"
    tags: tuple[str, ...] = ("pretraining",)
    notes: str | None = None


@dataclass
class PretrainingDataConfig:
    # The cache-backed manifest is the executable default.  The classified
    # report manifests are intermediate inputs used while rebuilding data and
    # do not contain the image cache columns required by the trainer.
    csv_path: str = "data/pretrain.csv"
    image_column: str = "path"
    text_column: str = "report"
    split_column: str = "split"
    fine_label_column: str = "fine_multi_hot"
    coarse_label_column: str = "coarse_multi_hot"
    train_split: str = "train"
    validation_split: str = "valid"
    strict_paths: bool = True
    allow_truncated_images: bool = False
    cache_path_column: str | None = None
    cache_index_column: str | None = None
    label_mode: str = "both"
    # Empty filters retain every record.  These fields support the reviewer-
    # requested single-organ/cross-organ and anatomy-specific comparisons
    # without generating derivative CSV files.
    source_column: str = "source"
    include_sources: tuple[str, ...] = ()
    include_fine_labels: tuple[str, ...] = ()
    include_coarse_labels: tuple[str, ...] = ()
    label_filter_mode: str = "any"
    max_records: int | None = None
    sampling_strategy: str = "head"
    sampling_seed: int = 42
    image: ImageTransformConfig = field(default_factory=ImageTransformConfig)
    loader: LoaderConfig = field(default_factory=lambda: LoaderConfig(batch_size=128))


@dataclass
class PretrainingConfig:
    model: UHMFIConfig = field(default_factory=UHMFIConfig)
    data: PretrainingDataConfig = field(default_factory=PretrainingDataConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    def validate(self) -> None:
        self.model.validate()
        _validate_runtime(self.data.image, self.data.loader, self.optimizer, self.trainer)
        if not self.data.csv_path:
            raise ValueError("A pre-training CSV path is required.")
        if self.data.label_mode not in {"both", "fine_only", "coarse_only"}:
            raise ValueError("label_mode must be both, fine_only, or coarse_only.")
        if self.data.label_filter_mode not in {"any", "all"}:
            raise ValueError("label_filter_mode must be any or all.")
        if self.data.sampling_strategy not in {"head", "random"}:
            raise ValueError("sampling_strategy must be head or random.")
        if self.data.max_records is not None and self.data.max_records <= 0:
            raise ValueError("max_records must be positive when provided.")
        if bool(self.data.cache_path_column) != bool(self.data.cache_index_column):
            raise ValueError(
                "cache_path_column and cache_index_column must be configured together."
            )
        for name, values in (
            ("include_sources", self.data.include_sources),
            ("include_fine_labels", self.data.include_fine_labels),
            ("include_coarse_labels", self.data.include_coarse_labels),
        ):
            if any(not str(value).strip() for value in values):
                raise ValueError(f"{name} cannot contain empty values.")
        if self.wandb.mode not in {"online", "offline", "disabled"}:
            raise ValueError("wandb.mode must be online, offline, or disabled.")
        if self.wandb.enabled and not self.wandb.project.strip():
            raise ValueError("wandb.project is required when W&B logging is enabled.")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PretrainingConfig":
        data_config = dict(data.get("data", {}))
        image = ImageTransformConfig(**dict(data_config.pop("image", {})))
        loader = LoaderConfig(**dict(data_config.pop("loader", {})))
        for key in (
            "include_sources",
            "include_fine_labels",
            "include_coarse_labels",
        ):
            if key in data_config:
                data_config[key] = tuple(str(value) for value in _tuple(data_config[key]))
        wandb_data = dict(data.get("wandb", {}))
        if "tags" in wandb_data:
            wandb_data["tags"] = tuple(str(tag) for tag in _tuple(wandb_data["tags"]))
        config = cls(
            model=UHMFIConfig.from_dict(dict(data.get("model", {}))),
            data=PretrainingDataConfig(image=image, loader=loader, **data_config),
            optimizer=OptimizerConfig(**dict(data.get("optimizer", {}))),
            scheduler=SchedulerConfig(**dict(data.get("scheduler", {}))),
            trainer=TrainerConfig(**dict(data.get("trainer", {}))),
            wandb=WandbConfig(**wandb_data),
        )
        config.validate()
        return config

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PretrainingConfig":
        data, config_path = _load_yaml(path)
        config = cls.from_dict(data)
        if data.get("model_config"):
            model_path = _resolve_relative(str(data["model_config"]), config_path)
            assert model_path is not None
            model_data, _ = _load_yaml(model_path)
            if "model" in model_data:
                model_data = dict(model_data["model"])
            config.model = UHMFIConfig.from_dict(
                _deep_merge(model_data, dict(data.get("model", {})))
            )
        config.data.csv_path = _resolve_relative(config.data.csv_path, config_path) or ""
        config.trainer.output_dir = _resolve_relative(config.trainer.output_dir, config_path) or ""
        config.trainer.resume = _resolve_relative(config.trainer.resume, config_path)
        config.validate()
        return config

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DownstreamDataConfig:
    csv_path: str = ""
    image_column: str = "path"
    label_column: str = "label"
    mask_column: str | None = None
    rle_column: str | None = None
    split_column: str = "split"
    train_split: str = "train"
    validation_split: str = "valid"
    test_split: str = "test"
    train_fraction: float = 1.0
    # The historical SIIM/GLoRIA protocol retains every positive training
    # image and deterministically downsamples negative images to the same
    # count before applying ``train_fraction``.  This is opt-in so that other
    # segmentation datasets keep their natural class distribution.
    balance_segmentation_train: bool = False
    strict_paths: bool = True
    enforce_disjoint_paths: bool = True
    deduplicate_classification_paths: bool = True
    cache_path_column: str | None = None
    cache_index_column: str | None = None
    mask_cache_path_column: str | None = None
    mask_cache_index_column: str | None = None
    max_records: int | None = None
    image: ImageTransformConfig = field(default_factory=ImageTransformConfig)
    loader: LoaderConfig = field(default_factory=LoaderConfig)


@dataclass
class DownstreamTaskConfig:
    type: str = "classification"
    num_classes: int = 1
    protocol: str = "linear_probe"
    feature_source: str = "global_feature"
    classifier_dropout: float = 0.0
    decoder_channels: int = 256
    segmentation_loss: str = "bce_dice"
    fixed_fine_labels: tuple[float, ...] = ()
    fixed_coarse_labels: tuple[float, ...] = ()


@dataclass
class DownstreamConfig:
    model: UHMFIConfig = field(default_factory=UHMFIConfig)
    pretrained_checkpoint: str = ""
    # New-format ablation checkpoints contain the exact encoder architecture.
    # Loading it prevents removed CSAM/HMCS branches from being silently
    # re-created with random parameters during downstream evaluation.
    use_checkpoint_model_config: bool = True
    data: DownstreamDataConfig = field(default_factory=DownstreamDataConfig)
    task: DownstreamTaskConfig = field(default_factory=DownstreamTaskConfig)
    optimizer: OptimizerConfig = field(
        default_factory=lambda: OptimizerConfig(learning_rate=5e-4)
    )
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    trainer: TrainerConfig = field(
        default_factory=lambda: TrainerConfig(output_dir="output/downstream")
    )
    wandb: WandbConfig = field(
        default_factory=lambda: WandbConfig(tags=("downstream",))
    )

    def validate(self) -> None:
        self.model.validate()
        _validate_runtime(self.data.image, self.data.loader, self.optimizer, self.trainer)
        if self.task.type not in {"classification", "segmentation"}:
            raise ValueError("Downstream task type must be classification or segmentation.")
        if self.task.protocol not in {"linear_probe", "fine_tune", "frozen_encoder"}:
            raise ValueError("Unsupported downstream protocol.")
        if self.task.num_classes <= 0:
            raise ValueError("num_classes must be positive.")
        if self.task.feature_source not in {"global_feature", "image_embedding"}:
            raise ValueError("feature_source must be global_feature or image_embedding.")
        if self.task.segmentation_loss not in {"bce", "dice", "bce_dice", "ce_dice"}:
            raise ValueError("Unsupported segmentation loss.")
        if bool(self.task.fixed_fine_labels) != bool(self.task.fixed_coarse_labels):
            raise ValueError("Fixed fine and coarse labels must be provided together.")
        if self.task.fixed_fine_labels and (
            len(self.task.fixed_fine_labels) != self.model.vision.num_fine_labels
            or len(self.task.fixed_coarse_labels) != self.model.vision.num_coarse_labels
        ):
            raise ValueError("Fixed downstream label vectors have incorrect dimensions.")
        if not 0.0 < self.data.train_fraction <= 1.0:
            raise ValueError("train_fraction must be in (0, 1].")
        if self.data.balance_segmentation_train and self.task.type != "segmentation":
            raise ValueError(
                "balance_segmentation_train is only valid for segmentation tasks."
            )
        if bool(self.data.cache_path_column) != bool(self.data.cache_index_column):
            raise ValueError(
                "cache_path_column and cache_index_column must be configured together."
            )
        if bool(self.data.mask_cache_path_column) != bool(self.data.mask_cache_index_column):
            raise ValueError(
                "mask_cache_path_column and mask_cache_index_column must be configured together."
            )
        if not self.data.csv_path:
            raise ValueError("A downstream CSV path is required.")
        if self.wandb.mode not in {"online", "offline", "disabled"}:
            raise ValueError("wandb.mode must be online, offline, or disabled.")
        if self.wandb.enabled and not self.wandb.project.strip():
            raise ValueError("wandb.project is required when W&B logging is enabled.")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DownstreamConfig":
        data_config = dict(data.get("data", {}))
        image = ImageTransformConfig(**dict(data_config.pop("image", {})))
        loader = LoaderConfig(**dict(data_config.pop("loader", {})))
        task_data = dict(data.get("task", {}))
        wandb_data = dict(data.get("wandb", {}))
        if "tags" in wandb_data:
            wandb_data["tags"] = tuple(
                str(tag) for tag in _tuple(wandb_data["tags"])
            )
        else:
            wandb_data["tags"] = ("downstream",)
        for key in ("fixed_fine_labels", "fixed_coarse_labels"):
            if key in task_data:
                task_data[key] = tuple(float(v) for v in _tuple(task_data[key]))
        config = cls(
            model=UHMFIConfig.from_dict(dict(data.get("model", {}))),
            pretrained_checkpoint=str(data.get("pretrained_checkpoint", "")),
            use_checkpoint_model_config=bool(
                data.get("use_checkpoint_model_config", True)
            ),
            data=DownstreamDataConfig(image=image, loader=loader, **data_config),
            task=DownstreamTaskConfig(**task_data),
            optimizer=OptimizerConfig(**dict(data.get("optimizer", {}))),
            scheduler=SchedulerConfig(**dict(data.get("scheduler", {}))),
            trainer=TrainerConfig(**dict(data.get("trainer", {}))),
            wandb=WandbConfig(**wandb_data),
        )
        config.validate()
        return config

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DownstreamConfig":
        data, config_path = _load_yaml(path)
        config = cls.from_dict(data)
        if data.get("model_config"):
            model_path = _resolve_relative(str(data["model_config"]), config_path)
            assert model_path is not None
            model_data, _ = _load_yaml(model_path)
            if "model" in model_data:
                model_data = dict(model_data["model"])
            config.model = UHMFIConfig.from_dict(
                _deep_merge(model_data, dict(data.get("model", {})))
            )
        config.data.csv_path = _resolve_relative(config.data.csv_path, config_path) or ""
        config.pretrained_checkpoint = (
            _resolve_relative(config.pretrained_checkpoint, config_path) or ""
        )
        config.trainer.output_dir = _resolve_relative(config.trainer.output_dir, config_path) or ""
        config.trainer.resume = _resolve_relative(config.trainer.resume, config_path)
        config.validate()
        return config

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _validate_runtime(
    image: ImageTransformConfig,
    loader: LoaderConfig,
    optimizer: OptimizerConfig,
    trainer: TrainerConfig,
) -> None:
    if image.resize <= 0 or image.crop_size <= 0 or image.crop_size > image.resize:
        raise ValueError("Image resize/crop dimensions are invalid.")
    if not 0.0 <= image.horizontal_flip <= 1.0:
        raise ValueError("horizontal_flip must be in [0, 1].")
    if image.normalize not in {"half", "imagenet", "none"}:
        raise ValueError("normalize must be half, imagenet, or none.")
    if loader.batch_size <= 0 or loader.num_workers < 0:
        raise ValueError("Invalid DataLoader configuration.")
    if optimizer.name.lower() not in {"adam", "adamw", "sgd"}:
        raise ValueError("Unsupported optimizer.")
    if optimizer.learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")
    if (
        trainer.epochs <= 0
        or trainer.gradient_accumulation <= 0
        or trainer.log_every_steps <= 0
    ):
        raise ValueError("Invalid trainer epoch/accumulation configuration.")
    if trainer.device not in {"auto", "cpu", "cuda"}:
        raise ValueError("trainer.device must be auto, cpu, or cuda.")
    if trainer.amp_dtype not in {"float16", "bfloat16"}:
        raise ValueError("amp_dtype must be float16 or bfloat16.")
    if not 0.0 < trainer.amp_init_scale < float("inf"):
        raise ValueError("amp_init_scale must be finite and positive.")
