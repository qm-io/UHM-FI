"""NumPy shard caches for downstream image and segmentation manifests."""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from .cache import build_pretraining_cache
from .config import DownstreamConfig, ImageTransformConfig
from .data import (
    MedicalImageTransform,
    _mask_paths,
    decode_rle_mask,
    load_binary_mask,
    load_medical_image,
)


MASK_CACHE_PATH_COLUMN = "mask_cache_path"
MASK_CACHE_INDEX_COLUMN = "mask_cache_index"

_MASK_TRANSFORM: MedicalImageTransform | None = None
_MASK_IMAGE_COLUMN = "path"
_MASK_COLUMN: str | None = None
_MASK_RLE_COLUMN: str | None = None


def _medical_image_size(path: str | Path) -> tuple[int, int]:
    """Read image dimensions without decoding the image pixels when possible."""

    image_path = Path(path).expanduser()
    if image_path.suffix.lower() == ".dcm":
        try:
            import pydicom
        except ImportError as exc:  # pragma: no cover - required by DICOM runs
            raise RuntimeError("pydicom is required to inspect DICOM dimensions.") from exc
        dataset = pydicom.dcmread(
            str(image_path),
            stop_before_pixels=True,
            specific_tags=["Rows", "Columns"],
        )
        try:
            return int(dataset.Columns), int(dataset.Rows)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError(f"DICOM image has no valid Rows/Columns: {image_path}") from exc
    with Image.open(image_path) as image:
        return image.size


def _init_mask_worker(
    image_config: dict[str, Any],
    image_column: str,
    mask_column: str | None,
    rle_column: str | None,
) -> None:
    global _MASK_TRANSFORM, _MASK_IMAGE_COLUMN, _MASK_COLUMN, _MASK_RLE_COLUMN
    torch.set_num_threads(1)
    _MASK_TRANSFORM = MedicalImageTransform(
        ImageTransformConfig(**image_config), training=False
    )
    _MASK_IMAGE_COLUMN = image_column
    _MASK_COLUMN = mask_column
    _MASK_RLE_COLUMN = rle_column


def _prepare_mask_row(row: dict[str, str]) -> np.ndarray:
    if _MASK_TRANSFORM is None:
        raise RuntimeError("Mask cache worker was not initialized.")
    width, height = _medical_image_size(row[_MASK_IMAGE_COLUMN])
    union = np.zeros((height, width), dtype=np.uint8)
    if _MASK_COLUMN:
        for mask_path in _mask_paths(row[_MASK_COLUMN]):
            mask = load_binary_mask(mask_path).resize(
                (width, height), Image.Resampling.NEAREST
            )
            union = np.maximum(union, np.asarray(mask, dtype=np.uint8))
    elif _MASK_RLE_COLUMN:
        mask = decode_rle_mask(
            row[_MASK_RLE_COLUMN], height=height, width=width
        )
        union = np.maximum(union, np.asarray(mask, dtype=np.uint8))
    else:
        raise ValueError("A mask or RLE column is required for mask caching.")
    tensor = _MASK_TRANSFORM.prepare_mask(Image.fromarray(union)).contiguous()
    array = tensor.numpy()
    expected = (
        1,
        _MASK_TRANSFORM.config.resize,
        _MASK_TRANSFORM.config.resize,
    )
    if tuple(array.shape) != expected or array.dtype != np.float32:
        raise ValueError(
            f"Prepared mask has invalid shape/dtype for {row[_MASK_IMAGE_COLUMN]}: "
            f"{array.shape}, {array.dtype}."
        )
    return array


def _read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or ())
        rows = list(reader)
    if not fields or not rows:
        raise ValueError(f"CSV is empty or missing a header: {path}")
    return fields, rows


def build_mask_cache(
    *,
    input_csv: str | Path,
    output_csv: str | Path,
    cache_root: str | Path,
    image_config: ImageTransformConfig,
    image_column: str,
    mask_column: str | None,
    rle_column: str | None,
    workers: int = 12,
    shard_size: int = 4096,
) -> dict[str, Any]:
    source = Path(input_csv).expanduser().resolve()
    destination = Path(output_csv).expanduser().resolve()
    root = Path(cache_root).expanduser().resolve()
    fields, rows = _read_rows(source)
    if destination == source:
        raise ValueError("Mask cache CSV must not overwrite the source CSV.")
    if mask_column is None and rle_column is None:
        raise ValueError("A mask column or RLE column is required.")
    for column in (MASK_CACHE_PATH_COLUMN, MASK_CACHE_INDEX_COLUMN):
        if column in fields:
            raise ValueError(f"CSV already contains cache column {column!r}.")
    shards_dir = root / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = root / "metadata.json"
    specification = {
        "cache_version": "uhm-fi-downstream-mask-float32-v2",
        "input_csv": str(source),
        "row_count": len(rows),
        "shard_size": shard_size,
        "shard_count": (len(rows) + shard_size - 1) // shard_size,
        "dtype": "float32",
        "shape": [1, image_config.resize, image_config.resize],
        "image_column": image_column,
        "mask_column": mask_column,
        "rle_column": rle_column,
        "rle_encoding": "relative_offset" if rle_column else None,
    }

    # Shape and dtype alone cannot distinguish masks decoded with an obsolete
    # RLE convention.  Reuse shards only when the complete cache specification
    # matches the metadata written by this implementation.
    reuse_existing_shards = False
    if metadata_path.is_file():
        try:
            existing_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            existing_metadata = {}
        reuse_existing_shards = all(
            existing_metadata.get(key) == value
            for key, value in specification.items()
        )

    pool: Any = None
    if workers > 0:
        pool = mp.get_context("fork").Pool(
            processes=workers,
            initializer=_init_mask_worker,
            initargs=(
                vars(image_config),
                image_column,
                mask_column,
                rle_column,
            ),
        )
    else:
        _init_mask_worker(
            vars(image_config), image_column, mask_column, rle_column
        )
    started = time.monotonic()
    try:
        for shard_index in range(0, len(rows), shard_size):
            chunk = rows[shard_index : shard_index + shard_size]
            shard_number = shard_index // shard_size
            final_path = shards_dir / f"mask_shard_{shard_number:05d}.npy"
            temporary_path = shards_dir / f".mask_shard_{shard_number:05d}.npy.tmp"
            expected_shape = (len(chunk), 1, image_config.resize, image_config.resize)
            if reuse_existing_shards and final_path.is_file():
                try:
                    existing = np.load(final_path, mmap_mode="r", allow_pickle=False)
                    if existing.dtype == np.float32 and tuple(existing.shape) == expected_shape:
                        continue
                except (OSError, ValueError):
                    pass
            temporary_path.unlink(missing_ok=True)
            shard = np.lib.format.open_memmap(
                temporary_path,
                mode="w+",
                dtype=np.float32,
                shape=expected_shape,
            )
            prepared = (
                pool.imap(_prepare_mask_row, chunk, chunksize=4)
                if pool is not None
                else map(_prepare_mask_row, chunk)
            )
            try:
                for offset, array in enumerate(prepared):
                    shard[offset] = array
            except BaseException:
                del shard
                temporary_path.unlink(missing_ok=True)
                raise
            shard.flush()
            del shard
            os.replace(temporary_path, final_path)
            processed = min(shard_index + len(chunk), len(rows))
            elapsed = max(time.monotonic() - started, 1e-9)
            print(
                json.dumps(
                    {
                        "event": "mask_cache_shard_completed",
                        "processed": processed,
                        "total": len(rows),
                        "samples_per_second": processed / elapsed,
                        "shard": shard_number,
                        "shard_path": str(final_path),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    output_fields = [*fields, MASK_CACHE_PATH_COLUMN, MASK_CACHE_INDEX_COLUMN]
    temporary_csv = destination.with_name(f".{destination.name}.tmp")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with temporary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields)
        writer.writeheader()
        for index, row in enumerate(rows):
            row = dict(row)
            shard_index, offset = divmod(index, shard_size)
            row[MASK_CACHE_PATH_COLUMN] = str(
                (root / "shards" / f"mask_shard_{shard_index:05d}.npy").resolve()
            )
            row[MASK_CACHE_INDEX_COLUMN] = str(offset)
            writer.writerow(row)
    os.replace(temporary_csv, destination)
    metadata = {
        **specification,
        "status": "complete",
        "output_csv": str(destination),
        "duration_seconds": time.monotonic() - started,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def prepare_downstream_cache(
    *,
    config_path: str | Path,
    input_csv: str | Path,
    output_csv: str | Path,
    cache_root: str | Path,
    workers: int,
    shard_size: int,
    execute: bool,
) -> dict[str, Any]:
    config = DownstreamConfig.from_yaml(config_path)
    source = Path(input_csv).expanduser().resolve()
    destination = Path(output_csv).expanduser().resolve()
    root = Path(cache_root).expanduser().resolve()
    summary = {
        "input_csv": str(source),
        "output_csv": str(destination),
        "cache_root": str(root),
        "image_column": config.data.image_column,
        "mask_column": config.data.mask_column,
        "rle_column": config.data.rle_column,
        "resize": config.data.image.resize,
        "workers": workers,
        "shard_size": shard_size,
    }
    if not execute:
        return {"status": "configuration_valid", "summary": summary}
    image_csv = destination
    image_result = build_pretraining_cache(
        input_csv=source,
        output_csv=image_csv,
        cache_root=root / "images",
        image_config=config.data.image,
        allow_truncated_images=False,
        image_column=config.data.image_column,
        workers=workers,
        shard_size=shard_size,
    )
    result: dict[str, Any] = {"status": "completed", "image_cache": image_result}
    if config.task.type == "segmentation":
        mask_result = build_mask_cache(
            input_csv=image_csv,
            output_csv=destination.with_name(f"{destination.stem}_mask.csv"),
            cache_root=root / "masks",
            image_config=config.data.image,
            image_column=config.data.image_column,
            mask_column=config.data.mask_column,
            rle_column=config.data.rle_column,
            workers=workers,
            shard_size=shard_size,
        )
        result["mask_cache"] = mask_result
        result["training_csv"] = mask_result["output_csv"]
    else:
        result["training_csv"] = str(image_csv)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Build downstream NumPy image/mask caches.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--shard-size", type=int, default=4096)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    options = vars(args)
    options["config_path"] = options.pop("config")
    print(json.dumps(prepare_downstream_cache(**options), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
