"""CSV-backed medical image datasets for pre-training and downstream tasks.

The implementation intentionally avoids a pandas dependency.  DICOM support is
enabled when ``pydicom`` is installed; ordinary PNG/JPEG/TIFF inputs work with
Pillow alone.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import math
import random
import re
import threading
import warnings
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageFile
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from .annotation import COARSE_LABELS, FINE_LABELS
from .config import (
    DownstreamConfig,
    ImageTransformConfig,
    LoaderConfig,
    PretrainingConfig,
    TextConfig,
)


class TruncatedImageWarning(UserWarning):
    """An explicitly permitted truncated image was decoded successfully."""


_PIL_IMAGE_LOAD_LOCK = threading.Lock()


def read_csv_records(
    path: str | Path,
    *,
    split_column: str | None = None,
    split: str | None = None,
    required_columns: Iterable[str] = (),
    max_records: int | None = None,
) -> list[dict[str, str]]:
    """Read selected CSV rows while preserving exact column names."""

    csv_path = Path(path).expanduser()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file does not exist: {csv_path}")
    records: list[dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or ())
        missing = set(required_columns) - fieldnames
        if split_column is not None and split_column not in fieldnames:
            missing.add(split_column)
        if missing:
            raise KeyError(
                f"CSV {csv_path} is missing required columns: {sorted(missing)}"
            )
        for row in reader:
            if split_column is not None and split is not None:
                if str(row.get(split_column, "")).strip().lower() != split.lower():
                    continue
            records.append({key: value or "" for key, value in row.items()})
            if max_records is not None and len(records) >= max_records:
                break
    if not records:
        split_message = "" if split is None else f" for split '{split}'"
        raise ValueError(f"No records found in {csv_path}{split_message}.")
    return records


def parse_numeric_vector(value: str | Sequence[Any], expected: int) -> Tensor:
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError) as exc:
            raise ValueError(f"Cannot parse multi-hot vector: {value[:80]!r}") from exc
    else:
        parsed = value
    tensor = torch.as_tensor(parsed, dtype=torch.float32)
    if tensor.ndim != 1 or tensor.numel() != expected:
        raise ValueError(
            f"Expected a vector with {expected} values, got {tuple(tensor.shape)}."
        )
    return tensor


def _requested_label_indices(
    requested: Sequence[str],
    labels: Sequence[str],
    kind: str,
) -> tuple[int, ...]:
    lookup = {label.casefold(): index for index, label in enumerate(labels)}
    unknown = sorted(
        {str(label) for label in requested if str(label).casefold() not in lookup}
    )
    if unknown:
        raise ValueError(
            f"Unknown {kind} labels in pre-training filter: {unknown}. "
            f"Available labels: {list(labels)}"
        )
    return tuple(lookup[str(label).casefold()] for label in requested)


def filter_pretraining_records(
    records: Sequence[Mapping[str, str]],
    config: PretrainingConfig,
) -> list[Mapping[str, str]]:
    """Apply source/anatomy filters used by pre-training ablations."""

    data = config.data
    allowed_sources = {source.casefold() for source in data.include_sources}
    fine_indices = _requested_label_indices(
        data.include_fine_labels, FINE_LABELS, "fine"
    )
    coarse_indices = _requested_label_indices(
        data.include_coarse_labels, COARSE_LABELS, "coarse"
    )
    selected: list[Mapping[str, str]] = []
    for record in records:
        if allowed_sources and record[data.source_column].strip().casefold() not in allowed_sources:
            continue
        label_matches: list[bool] = []
        if fine_indices:
            fine = parse_numeric_vector(
                record[data.fine_label_column],
                config.model.vision.num_fine_labels,
            )
            label_matches.extend(bool(fine[index] > 0) for index in fine_indices)
        if coarse_indices:
            coarse = parse_numeric_vector(
                record[data.coarse_label_column],
                config.model.vision.num_coarse_labels,
            )
            label_matches.extend(bool(coarse[index] > 0) for index in coarse_indices)
        if label_matches:
            matched = (
                all(label_matches)
                if data.label_filter_mode == "all"
                else any(label_matches)
            )
            if not matched:
                continue
        selected.append(record)
    return selected


def sample_pretraining_records(
    records: Sequence[Mapping[str, str]],
    config: PretrainingConfig,
    *,
    namespace: str,
) -> list[Mapping[str, str]]:
    """Apply a deterministic post-filter cap for size-matched ablations."""

    limit = config.data.max_records
    if limit is None or len(records) <= limit:
        return list(records)
    if config.data.sampling_strategy == "head":
        return list(records[:limit])
    digest = hashlib.blake2b(
        f"{config.data.sampling_seed}:{namespace}".encode("utf-8"), digest_size=8
    ).digest()
    rng = random.Random(int.from_bytes(digest, "little"))
    indices = sorted(rng.sample(range(len(records)), limit))
    return [records[index] for index in indices]


def _normalize_array(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros(values.shape, dtype=np.uint8)
    valid = values[finite]
    lower, upper = np.percentile(valid, (0.5, 99.5))
    if upper <= lower:
        lower, upper = float(valid.min()), float(valid.max())
    if upper <= lower:
        return np.zeros(values.shape, dtype=np.uint8)
    values = np.clip((values - lower) / (upper - lower), 0.0, 1.0)
    values[~finite] = 0.0
    return (values * 255.0).round().astype(np.uint8)


def _is_truncated_image_error(error: OSError) -> bool:
    return str(error).lower().startswith("image file is truncated")


def _load_pillow_image(
    image_path: Path,
    *,
    allow_truncated_images: bool,
) -> Image.Image:
    # Pillow exposes truncated-image handling as process-wide state.  Guard the
    # strict attempt, optional retry, and restoration so concurrent callers do
    # not inherit a temporary permissive setting.
    with _PIL_IMAGE_LOAD_LOCK:
        previous_setting = ImageFile.LOAD_TRUNCATED_IMAGES
        try:
            ImageFile.LOAD_TRUNCATED_IMAGES = False
            try:
                with Image.open(image_path) as image:
                    return image.convert("L").copy()
            except OSError as error:
                if not (
                    allow_truncated_images and _is_truncated_image_error(error)
                ):
                    raise OSError(
                        f"Unable to decode image {image_path}: {error}"
                    ) from error

                strict_error = error
                ImageFile.LOAD_TRUNCATED_IMAGES = True
                try:
                    with Image.open(image_path) as image:
                        recovered = image.convert("L").copy()
                except OSError as recovery_error:
                    raise OSError(
                        "Unable to recover truncated image "
                        f"{image_path}: {recovery_error}"
                    ) from recovery_error
                warnings.warn(
                    f"Recovered truncated image {image_path}: {strict_error}",
                    TruncatedImageWarning,
                    stacklevel=2,
                )
                return recovered
        finally:
            ImageFile.LOAD_TRUNCATED_IMAGES = previous_setting


def load_medical_image(
    path: str | Path,
    *,
    allow_truncated_images: bool = False,
) -> Image.Image:
    image_path = Path(path).expanduser()
    if not image_path.exists():
        raise FileNotFoundError(f"Image does not exist: {image_path}")
    if image_path.suffix.lower() == ".dcm":
        try:
            import pydicom
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "pydicom is required to read DICOM inputs. Install requirements.txt."
            ) from exc
        dataset = pydicom.dcmread(str(image_path))
        try:
            array = dataset.pixel_array.astype(np.float32)
        except RuntimeError as exc:
            transfer_syntax = getattr(
                getattr(dataset, "file_meta", None),
                "TransferSyntaxUID",
                "unknown",
            )
            raise RuntimeError(
                "Unable to decode DICOM pixel data for "
                f"{image_path} (TransferSyntaxUID={transfer_syntax}). "
                "Install the compressed-pixel decoders with "
                "`python -m pip install -U \"pylibjpeg[all]\"`."
            ) from exc
        slope = float(getattr(dataset, "RescaleSlope", 1.0))
        intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
        array = array * slope + intercept
        array = _normalize_array(array)
        if str(getattr(dataset, "PhotometricInterpretation", "")) == "MONOCHROME1":
            array = 255 - array
        return Image.fromarray(array)
    return _load_pillow_image(
        image_path,
        allow_truncated_images=allow_truncated_images,
    )


def load_binary_mask(path: str | Path) -> Image.Image:
    mask = load_medical_image(path)
    array = np.asarray(mask, dtype=np.uint8)
    binary = (array > 0).astype(np.uint8) * 255
    return Image.fromarray(binary)


def _resize_and_pad(tensor: Tensor, size: int, interpolation: InterpolationMode) -> Tensor:
    _, height, width = tensor.shape
    scale = size / float(max(height, width))
    resized_height = max(1, int(round(height * scale)))
    resized_width = max(1, int(round(width * scale)))
    tensor = TF.resize(
        tensor,
        [resized_height, resized_width],
        interpolation=interpolation,
        antialias=interpolation != InterpolationMode.NEAREST,
    )
    pad_height = size - resized_height
    pad_width = size - resized_width
    left = pad_width // 2
    right = pad_width - left
    top = pad_height // 2
    bottom = pad_height - top
    return TF.pad(tensor, [left, top, right, bottom], fill=0)


class MedicalImageTransform:
    """Shared image and paired image-mask preprocessing."""

    def __init__(self, config: ImageTransformConfig, training: bool) -> None:
        self.config = config
        self.training = training

    def _parameters(self) -> tuple[int, int, bool]:
        maximum = self.config.resize - self.config.crop_size
        top = random.randint(0, maximum) if self.training and maximum > 0 else maximum // 2
        left = random.randint(0, maximum) if self.training and maximum > 0 else maximum // 2
        flip = self.training and random.random() < self.config.horizontal_flip
        return top, left, flip

    def _normalize(self, image: Tensor) -> Tensor:
        if self.config.normalize == "none":
            return image
        if self.config.normalize == "half":
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]
        else:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        return TF.normalize(image, mean, std)

    def prepare_image(self, image: Image.Image) -> Tensor:
        """Apply deterministic decoding-boundary preprocessing before augmentation."""

        image_tensor = TF.pil_to_tensor(image).float().div_(255.0)
        return _resize_and_pad(
            image_tensor, self.config.resize, InterpolationMode.BILINEAR
        )

    def prepare_mask(self, mask: Image.Image) -> Tensor:
        mask_tensor = TF.pil_to_tensor(mask).float().div_(255.0)
        return _resize_and_pad(
            mask_tensor, self.config.resize, InterpolationMode.NEAREST
        )

    def _augment_prepared(
        self,
        image_tensor: Tensor,
        mask_tensor: Tensor | None = None,
    ) -> Tensor | tuple[Tensor, Tensor]:
        expected = (1, self.config.resize, self.config.resize)
        if tuple(image_tensor.shape) != expected:
            raise ValueError(
                f"Prepared image must have shape {expected}, got "
                f"{tuple(image_tensor.shape)}."
            )
        if mask_tensor is not None and tuple(mask_tensor.shape) != expected:
            raise ValueError(
                f"Prepared mask must have shape {expected}, got "
                f"{tuple(mask_tensor.shape)}."
            )

        top, left, flip = self._parameters()
        crop = self.config.crop_size
        image_tensor = TF.crop(image_tensor, top, left, crop, crop)
        if mask_tensor is not None:
            mask_tensor = TF.crop(mask_tensor, top, left, crop, crop)
        if flip:
            image_tensor = TF.hflip(image_tensor)
            if mask_tensor is not None:
                mask_tensor = TF.hflip(mask_tensor)

        image_tensor = image_tensor.expand(3, -1, -1).contiguous()
        image_tensor = self._normalize(image_tensor)
        if mask_tensor is None:
            return image_tensor
        return image_tensor, mask_tensor.gt(0.5).float()

    def from_prepared_image(self, image_tensor: Tensor) -> Tensor:
        """Apply online random augmentation to a cached prepared image."""

        transformed = self._augment_prepared(image_tensor)
        assert isinstance(transformed, Tensor)
        return transformed

    def from_prepared_pair(
        self,
        image_tensor: Tensor,
        mask_tensor: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Apply identical online augmentation to cached image and mask tensors."""

        transformed = self._augment_prepared(image_tensor, mask_tensor)
        assert isinstance(transformed, tuple)
        return transformed

    def __call__(
        self,
        image: Image.Image,
        mask: Image.Image | None = None,
    ) -> Tensor | tuple[Tensor, Tensor]:
        image_tensor = self.prepare_image(image)
        mask_tensor = self.prepare_mask(mask) if mask is not None else None
        return self._augment_prepared(image_tensor, mask_tensor)


class HashingTokenizer:
    """Deterministic offline tokenizer for environments without a text model."""

    token_pattern = re.compile(r"[A-Za-z0-9_]+")

    def __init__(self, vocab_size: int, max_length: int) -> None:
        if vocab_size < 4:
            raise ValueError("vocab_size must be at least 4.")
        self.vocab_size = vocab_size
        self.max_length = max_length

    def _token_id(self, token: str) -> int:
        digest = hashlib.blake2b(token.lower().encode("utf-8"), digest_size=8).digest()
        return 2 + int.from_bytes(digest, "little") % (self.vocab_size - 2)

    def __call__(self, texts: Sequence[str]) -> dict[str, Tensor]:
        input_ids = torch.zeros((len(texts), self.max_length), dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for row, text in enumerate(texts):
            tokens = [1]
            tokens.extend(self._token_id(token) for token in self.token_pattern.findall(text))
            tokens = tokens[: self.max_length]
            input_ids[row, : len(tokens)] = torch.tensor(tokens, dtype=torch.long)
            attention_mask[row, : len(tokens)] = 1
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": torch.zeros_like(input_ids),
        }


def build_tokenizer(config: TextConfig) -> Any:
    if config.backend == "tiny":
        return HashingTokenizer(config.vocab_size, config.max_length)
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("transformers is required for BioClinicalBERT.") from exc
    return AutoTokenizer.from_pretrained(
        config.model_name,
        local_files_only=config.local_files_only,
        use_fast=True,
    )


class NpyShardCache:
    """Process-local lazy reader for fixed-shape float32 NPY shards."""

    def __init__(self) -> None:
        self._shards: dict[str, np.ndarray] = {}

    def __getstate__(self) -> dict[str, Any]:
        # DataLoader workers must open their own memory maps.
        return {"_shards": {}}

    def load(
        self,
        path: str | Path,
        index: int,
        *,
        expected_size: int,
    ) -> Tensor:
        cache_path = Path(path).expanduser()
        key = str(cache_path)
        shard = self._shards.get(key)
        if shard is None:
            if not cache_path.is_file():
                raise FileNotFoundError(f"Image cache shard does not exist: {cache_path}")
            try:
                shard = np.load(cache_path, mmap_mode="r", allow_pickle=False)
            except (OSError, ValueError) as error:
                raise OSError(
                    f"Unable to open image cache shard {cache_path}: {error}"
                ) from error
            expected_tail = (1, expected_size, expected_size)
            if shard.ndim != 4 or tuple(shard.shape[1:]) != expected_tail:
                raise ValueError(
                    f"Cache shard {cache_path} must have shape "
                    f"[N, {expected_tail[0]}, {expected_tail[1]}, {expected_tail[2]}], "
                    f"got {tuple(shard.shape)}."
                )
            if shard.dtype != np.float32:
                raise ValueError(
                    f"Cache shard {cache_path} must use float32, got {shard.dtype}."
                )
            self._shards[key] = shard
        if index < 0 or index >= int(shard.shape[0]):
            raise IndexError(
                f"Cache index {index} is outside shard {cache_path} "
                f"with {shard.shape[0]} records."
            )
        # Copy from the read-only memmap into worker-owned writable memory.
        return torch.from_numpy(np.array(shard[index], dtype=np.float32, copy=True))


class PretrainingCSVDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        records: Sequence[Mapping[str, str]],
        config: PretrainingConfig,
        training: bool,
    ) -> None:
        self.records = list(records)
        self.data_config = config.data
        self.model_config = config.model
        self.transform = MedicalImageTransform(config.data.image, training)
        self.image_cache = NpyShardCache() if config.data.cache_path_column else None

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image_path = record[self.data_config.image_column]
        if self.image_cache is None:
            image = self.transform(
                load_medical_image(
                    image_path,
                    allow_truncated_images=self.data_config.allow_truncated_images,
                )
            )
        else:
            assert self.data_config.cache_path_column is not None
            assert self.data_config.cache_index_column is not None
            cache_path = record[self.data_config.cache_path_column]
            try:
                cache_index = int(record[self.data_config.cache_index_column])
            except ValueError as error:
                raise ValueError(
                    f"Invalid cache index for {image_path}: "
                    f"{record[self.data_config.cache_index_column]!r}"
                ) from error
            prepared = self.image_cache.load(
                cache_path,
                cache_index,
                expected_size=self.data_config.image.resize,
            )
            image = self.transform.from_prepared_image(prepared)
        assert isinstance(image, Tensor)
        fine_labels = parse_numeric_vector(
            record[self.data_config.fine_label_column],
            self.model_config.vision.num_fine_labels,
        )
        coarse_labels = parse_numeric_vector(
            record[self.data_config.coarse_label_column],
            self.model_config.vision.num_coarse_labels,
        )
        if self.data_config.label_mode == "fine_only":
            coarse_labels.zero_()
        elif self.data_config.label_mode == "coarse_only":
            fine_labels.zero_()
        return {
            "images": image,
            "text": record[self.data_config.text_column],
            "fine_labels": fine_labels,
            "coarse_labels": coarse_labels,
            "paths": image_path,
        }


class PretrainingCollator:
    def __init__(self, tokenizer: Any, text_config: TextConfig) -> None:
        self.tokenizer = tokenizer
        self.text_config = text_config

    def __call__(self, samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        texts = [str(sample["text"]) for sample in samples]
        if isinstance(self.tokenizer, HashingTokenizer):
            tokens = self.tokenizer(texts)
        else:
            tokens = self.tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=self.text_config.max_length,
                return_tensors="pt",
            )
            if "token_type_ids" not in tokens:
                tokens["token_type_ids"] = torch.zeros_like(tokens["input_ids"])
        return {
            "images": torch.stack([sample["images"] for sample in samples]),
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
            "token_type_ids": tokens["token_type_ids"],
            "fine_labels": torch.stack([sample["fine_labels"] for sample in samples]),
            "coarse_labels": torch.stack([sample["coarse_labels"] for sample in samples]),
            "paths": [sample["paths"] for sample in samples],
        }


def _parse_label(value: str) -> int:
    stripped = value.strip()
    try:
        return int(float(stripped))
    except ValueError as exc:
        raise ValueError(
            f"Classification labels must be numeric. Received {value!r}."
        ) from exc


def _stratified_fraction(
    records: Sequence[Mapping[str, str]],
    label_column: str,
    fraction: float,
    seed: int,
) -> list[Mapping[str, str]]:
    if fraction >= 1.0:
        return list(records)
    groups: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for record in records:
        groups[str(record[label_column])].append(record)
    selected: list[Mapping[str, str]] = []
    rng = random.Random(seed)
    for group in groups.values():
        shuffled = list(group)
        rng.shuffle(shuffled)
        count = max(1, int(math.floor(len(shuffled) * fraction + 0.5)))
        selected.extend(shuffled[:count])
    rng.shuffle(selected)
    return selected


def _deduplicate_classification_records(
    records: Sequence[Mapping[str, str]],
    image_column: str,
    label_column: str,
) -> list[Mapping[str, str]]:
    unique: dict[str, Mapping[str, str]] = {}
    for record in records:
        path = record[image_column]
        previous = unique.get(path)
        if previous is not None and previous[label_column] != record[label_column]:
            raise ValueError(
                f"Conflicting labels for the same image path: {path!r} "
                f"({previous[label_column]!r} vs {record[label_column]!r})"
            )
        unique.setdefault(path, record)
    return list(unique.values())


def _assert_disjoint_paths(
    split_records: Mapping[str, Sequence[Mapping[str, str]]],
    image_column: str,
) -> None:
    names = list(split_records)
    path_sets = {
        name: {record[image_column] for record in records}
        for name, records in split_records.items()
    }
    overlaps: list[str] = []
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            shared = path_sets[left].intersection(path_sets[right])
            if shared:
                examples = sorted(shared)[:5]
                overlaps.append(
                    f"{left}/{right}: {len(shared)} shared image paths, examples={examples}"
                )
    if overlaps:
        raise ValueError(
            "Downstream split leakage detected. Rebuild the CSV with image/patient-level "
            "grouping before training. " + " | ".join(overlaps)
        )


class ClassificationCSVDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        records: Sequence[Mapping[str, str]],
        config: DownstreamConfig,
        training: bool,
    ) -> None:
        self.records = list(records)
        self.config = config
        self.transform = MedicalImageTransform(config.data.image, training)
        self.image_cache = NpyShardCache() if config.data.cache_path_column else None

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        path = record[self.config.data.image_column]
        if self.image_cache is None:
            image = self.transform(load_medical_image(path))
        else:
            assert self.config.data.cache_path_column is not None
            assert self.config.data.cache_index_column is not None
            prepared = self.image_cache.load(
                record[self.config.data.cache_path_column],
                int(record[self.config.data.cache_index_column]),
                expected_size=self.config.data.image.resize,
            )
            image = self.transform.from_prepared_image(prepared)
        assert isinstance(image, Tensor)
        label = _parse_label(record[self.config.data.label_column])
        return {
            "images": image,
            "labels": torch.tensor(label, dtype=torch.long),
            "paths": path,
        }


def decode_rle_mask(rle: str, height: int, width: int) -> Image.Image:
    """Decode the relative-offset RLE used by the SIIM training CSV.

    The first value of each pair is the number of pixels to skip from the end
    of the previous run, rather than an absolute one-based start position.
    This matches the dataset implementation used for the original experiments.
    """

    value = rle.strip()
    if value in {"", "-1", "nan", "None"}:
        return Image.fromarray(np.zeros((height, width), dtype=np.uint8))
    values = [int(item) for item in value.split()]
    if len(values) % 2 != 0:
        raise ValueError(f"Invalid run-length encoding: {value[:80]!r}")
    flat = np.zeros(height * width, dtype=np.uint8)
    current_position = 0
    for offset, length in zip(values[0::2], values[1::2]):
        if offset < 0 or length < 0:
            raise ValueError(f"RLE offsets and lengths must be non-negative: {value[:80]!r}")
        current_position += offset
        end = current_position + length
        if end > flat.size:
            raise ValueError(
                "RLE run exceeds the mask dimensions: "
                f"start={current_position}, length={length}, pixels={flat.size}."
            )
        flat[current_position:end] = 255
        current_position = end
    # SIIM encodes masks in column-major order.
    array = flat.reshape((width, height)).T
    return Image.fromarray(array)


def _mask_paths(value: str) -> list[str]:
    stripped = value.strip()
    if not stripped:
        return []
    if stripped.startswith("["):
        try:
            parsed = ast.literal_eval(stripped)
        except (ValueError, SyntaxError) as exc:
            raise ValueError(f"Cannot parse mask path list: {value!r}") from exc
        return [str(path) for path in parsed]
    return [stripped]


class SegmentationCSVDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        records: Sequence[Mapping[str, str]],
        config: DownstreamConfig,
        training: bool,
    ) -> None:
        grouped: dict[str, list[Mapping[str, str]]] = defaultdict(list)
        for record in records:
            grouped[record[config.data.image_column]].append(record)
        self.records = list(grouped.items())
        self.config = config
        self.transform = MedicalImageTransform(config.data.image, training)
        self.image_cache = NpyShardCache() if config.data.cache_path_column else None
        self.mask_cache = NpyShardCache() if config.data.mask_cache_path_column else None

    def __len__(self) -> int:
        return len(self.records)

    def _mask(self, image: Image.Image, rows: Sequence[Mapping[str, str]]) -> Image.Image:
        width, height = image.size
        union = np.zeros((height, width), dtype=np.uint8)
        if self.config.data.mask_column:
            for row in rows:
                for path in _mask_paths(row[self.config.data.mask_column]):
                    mask = load_binary_mask(path).resize((width, height), Image.Resampling.NEAREST)
                    union = np.maximum(union, np.asarray(mask, dtype=np.uint8))
        elif self.config.data.rle_column:
            for row in rows:
                mask = decode_rle_mask(
                    row[self.config.data.rle_column], height=height, width=width
                )
                union = np.maximum(union, np.asarray(mask, dtype=np.uint8))
        else:
            raise ValueError("Segmentation data require mask_column or rle_column.")
        return Image.fromarray(union)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path, rows = self.records[index]
        if self.image_cache is None:
            image = load_medical_image(path)
            mask = self._mask(image, rows)
            transformed = self.transform(image, mask)
        else:
            assert self.config.data.cache_path_column is not None
            assert self.config.data.cache_index_column is not None
            first = rows[0]
            prepared_image = self.image_cache.load(
                first[self.config.data.cache_path_column],
                int(first[self.config.data.cache_index_column]),
                expected_size=self.config.data.image.resize,
            )
            if self.mask_cache is not None:
                assert self.config.data.mask_cache_path_column is not None
                assert self.config.data.mask_cache_index_column is not None
                prepared_mask = torch.zeros_like(prepared_image)
                for row in rows:
                    row_mask = self.mask_cache.load(
                        row[self.config.data.mask_cache_path_column],
                        int(row[self.config.data.mask_cache_index_column]),
                        expected_size=self.config.data.image.resize,
                    )
                    prepared_mask = torch.maximum(prepared_mask, row_mask)
                transformed = self.transform.from_prepared_pair(
                    prepared_image, prepared_mask
                )
            else:
                # The image cache still removes the expensive image resize. A
                # fallback mask decode is retained for manifests without a
                # prepared mask cache.
                image = load_medical_image(path)
                mask = self._mask(image, rows)
                prepared_mask = self.transform.prepare_mask(mask)
                transformed = self.transform.from_prepared_pair(
                    prepared_image, prepared_mask
                )
        assert isinstance(transformed, tuple)
        image_tensor, mask_tensor = transformed
        return {"images": image_tensor, "masks": mask_tensor, "paths": path}


def _rle_has_foreground(value: str) -> bool:
    return value.strip() not in {"", "-1", "nan", "None"}


def _segmentation_group_has_foreground(
    rows: Sequence[Mapping[str, str]],
    config: DownstreamConfig,
) -> bool:
    if config.data.rle_column:
        return any(_rle_has_foreground(row[config.data.rle_column]) for row in rows)
    if config.data.mask_column:
        # Mask manifests such as CBIS-DDSM contain rows only for annotated
        # lesions.  An empty path represents a true empty mask if one is ever
        # included in a balanced protocol.
        return any(bool(_mask_paths(row[config.data.mask_column])) for row in rows)
    raise ValueError("Segmentation data require mask_column or rle_column.")


def _sample_segmentation_training_records(
    records: Sequence[Mapping[str, str]],
    config: DownstreamConfig,
) -> list[Mapping[str, str]]:
    """Sample segmentation records at image level with optional 1:1 balance."""

    grouped: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for record in records:
        grouped[record[config.data.image_column]].append(record)
    image_groups = list(grouped.values())
    rng = random.Random(config.trainer.seed)

    if config.data.balance_segmentation_train:
        positive_groups = [
            group
            for group in image_groups
            if _segmentation_group_has_foreground(group, config)
        ]
        negative_groups = [
            group
            for group in image_groups
            if not _segmentation_group_has_foreground(group, config)
        ]
        if not positive_groups or not negative_groups:
            raise ValueError(
                "Balanced segmentation sampling requires both positive and negative "
                "training images."
            )
        if len(negative_groups) < len(positive_groups):
            raise ValueError(
                "Balanced segmentation sampling cannot match all positive images: "
                f"positive={len(positive_groups)}, negative={len(negative_groups)}."
            )
        rng.shuffle(positive_groups)
        rng.shuffle(negative_groups)
        negative_groups = negative_groups[: len(positive_groups)]
        # Match the historical loader exactly: form the balanced pool first,
        # then draw the requested fraction from that pool as a whole.
        image_groups = positive_groups + negative_groups
        rng.shuffle(image_groups)
        count = max(1, int(len(image_groups) * config.data.train_fraction))
        selected_groups = image_groups[:count]
    else:
        rng.shuffle(image_groups)
        count = max(1, int(len(image_groups) * config.data.train_fraction))
        selected_groups = image_groups[:count]

    rng.shuffle(selected_groups)
    return [record for group in selected_groups for record in group]


def _loader(
    dataset: Dataset[Any],
    loader_config: LoaderConfig,
    *,
    shuffle: bool,
    collate_fn: Any = None,
) -> DataLoader[Any]:
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": loader_config.batch_size,
        "shuffle": shuffle,
        "num_workers": loader_config.num_workers,
        "pin_memory": loader_config.pin_memory,
        "drop_last": shuffle,
        "collate_fn": collate_fn,
    }
    if loader_config.num_workers > 0:
        kwargs["persistent_workers"] = loader_config.persistent_workers
        kwargs["prefetch_factor"] = loader_config.prefetch_factor
    return DataLoader(**kwargs)


def build_pretraining_loaders(
    config: PretrainingConfig,
) -> tuple[DataLoader[Any], DataLoader[Any]]:
    columns = [
        config.data.image_column,
        config.data.text_column,
        config.data.fine_label_column,
        config.data.coarse_label_column,
    ]
    if config.data.cache_path_column:
        assert config.data.cache_index_column is not None
        columns.extend(
            (config.data.cache_path_column, config.data.cache_index_column)
        )
    if config.data.include_sources:
        columns.append(config.data.source_column)
    has_filters = bool(
        config.data.include_sources
        or config.data.include_fine_labels
        or config.data.include_coarse_labels
    )
    postprocess_records = has_filters or config.data.sampling_strategy == "random"
    read_limit = None if postprocess_records else config.data.max_records
    train_records = read_csv_records(
        config.data.csv_path,
        split_column=config.data.split_column,
        split=config.data.train_split,
        required_columns=columns,
        max_records=read_limit,
    )
    validation_records = read_csv_records(
        config.data.csv_path,
        split_column=config.data.split_column,
        split=config.data.validation_split,
        required_columns=columns,
        max_records=read_limit,
    )
    if has_filters:
        train_records = list(filter_pretraining_records(train_records, config))
        validation_records = list(filter_pretraining_records(validation_records, config))
    if postprocess_records:
        train_records = list(
            sample_pretraining_records(
                train_records,
                config,
                namespace=config.data.train_split,
            )
        )
        validation_records = list(
            sample_pretraining_records(
                validation_records,
                config,
                namespace=config.data.validation_split,
            )
        )
    if has_filters or postprocess_records:
        for split, records in (
            (config.data.train_split, train_records),
            (config.data.validation_split, validation_records),
        ):
            if not records:
                raise ValueError(
                    f"Pre-training filters selected no records for split {split!r}."
                )
    tokenizer = build_tokenizer(config.model.text)
    collator = PretrainingCollator(tokenizer, config.model.text)
    return (
        _loader(
            PretrainingCSVDataset(train_records, config, training=True),
            config.data.loader,
            shuffle=True,
            collate_fn=collator,
        ),
        _loader(
            PretrainingCSVDataset(validation_records, config, training=False),
            config.data.loader,
            shuffle=False,
            collate_fn=collator,
        ),
    )


def build_downstream_loaders(
    config: DownstreamConfig,
) -> tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any]]:
    required = [config.data.image_column]
    if config.task.type == "classification":
        required.append(config.data.label_column)
    elif config.data.mask_column:
        required.append(config.data.mask_column)
    elif config.data.rle_column:
        required.append(config.data.rle_column)
    for column in (
        config.data.cache_path_column,
        config.data.cache_index_column,
        config.data.mask_cache_path_column,
        config.data.mask_cache_index_column,
    ):
        if column:
            required.append(column)

    def records(split: str) -> list[dict[str, str]]:
        return read_csv_records(
            config.data.csv_path,
            split_column=config.data.split_column,
            split=split,
            required_columns=required,
            max_records=config.data.max_records,
        )

    train_records: Sequence[Mapping[str, str]] = records(config.data.train_split)
    validation_records: Sequence[Mapping[str, str]] = records(
        config.data.validation_split
    )
    test_records: Sequence[Mapping[str, str]] = records(config.data.test_split)
    if config.data.enforce_disjoint_paths:
        _assert_disjoint_paths(
            {
                config.data.train_split: train_records,
                config.data.validation_split: validation_records,
                config.data.test_split: test_records,
            },
            config.data.image_column,
        )
    if config.task.type == "classification":
        if config.data.deduplicate_classification_paths:
            train_records = _deduplicate_classification_records(
                train_records, config.data.image_column, config.data.label_column
            )
            validation_records = _deduplicate_classification_records(
                validation_records, config.data.image_column, config.data.label_column
            )
            test_records = _deduplicate_classification_records(
                test_records, config.data.image_column, config.data.label_column
            )
        train_records = _stratified_fraction(
            train_records,
            config.data.label_column,
            config.data.train_fraction,
            config.trainer.seed,
        )
        dataset_type: type[Dataset[dict[str, Any]]] = ClassificationCSVDataset
    else:
        if config.data.balance_segmentation_train or config.data.train_fraction < 1.0:
            train_records = _sample_segmentation_training_records(train_records, config)
        dataset_type = SegmentationCSVDataset

    return (
        _loader(dataset_type(train_records, config, True), config.data.loader, shuffle=True),
        _loader(
            dataset_type(validation_records, config, False),
            config.data.loader,
            shuffle=False,
        ),
        _loader(
            dataset_type(test_records, config, False),
            config.data.loader,
            shuffle=False,
        ),
    )
