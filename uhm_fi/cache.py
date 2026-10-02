"""Resumable fixed-shape image cache generation for UHM-FI pre-training."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
import shutil
import time
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch

from .config import ImageTransformConfig, PretrainingConfig
from .data import MedicalImageTransform, NpyShardCache, load_medical_image


CACHE_VERSION = "uhm-fi-prepared-float32-v1"
DEFAULT_CACHE_PATH_COLUMN = "cache_path"
DEFAULT_CACHE_INDEX_COLUMN = "cache_index"

_WORKER_TRANSFORM: MedicalImageTransform | None = None
_WORKER_ALLOW_TRUNCATED = False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _csv_layout(path: Path, image_column: str) -> tuple[list[str], int]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or ())
        if image_column not in fieldnames:
            raise KeyError(f"CSV {path} does not contain image column {image_column!r}.")
        row_count = sum(1 for _ in reader)
    if row_count <= 0:
        raise ValueError(f"CSV {path} contains no records.")
    return fieldnames, row_count


def _path_chunks(
    path: Path,
    *,
    image_column: str,
    shard_size: int,
) -> Iterator[list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        chunk: list[str] = []
        for row in reader:
            chunk.append(row[image_column])
            if len(chunk) == shard_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk


def _init_worker(image_config: dict[str, Any], allow_truncated: bool) -> None:
    global _WORKER_ALLOW_TRUNCATED, _WORKER_TRANSFORM
    torch.set_num_threads(1)
    _WORKER_TRANSFORM = MedicalImageTransform(
        ImageTransformConfig(**image_config), training=False
    )
    _WORKER_ALLOW_TRUNCATED = allow_truncated


def _prepare_path(path: str) -> np.ndarray:
    if _WORKER_TRANSFORM is None:
        raise RuntimeError("Cache worker was not initialized.")
    image = load_medical_image(
        path,
        allow_truncated_images=_WORKER_ALLOW_TRUNCATED,
    )
    tensor = _WORKER_TRANSFORM.prepare_image(image).contiguous()
    array = tensor.numpy()
    expected = (
        1,
        _WORKER_TRANSFORM.config.resize,
        _WORKER_TRANSFORM.config.resize,
    )
    if tuple(array.shape) != expected or array.dtype != np.float32:
        raise ValueError(
            f"Prepared image {path} has invalid shape/dtype: "
            f"{array.shape}, {array.dtype}."
        )
    if not np.isfinite(array).all():
        raise ValueError(f"Prepared image contains non-finite values: {path}")
    return array


def _valid_shard(path: Path, count: int, size: int) -> bool:
    if not path.is_file():
        return False
    try:
        shard = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError):
        return False
    return (
        shard.dtype == np.float32
        and tuple(shard.shape) == (count, 1, size, size)
        and path.stat().st_size > count * size * size * np.dtype(np.float32).itemsize
    )


def _progress(
    *,
    event: str,
    processed: int,
    total: int,
    started: float,
    **extra: Any,
) -> None:
    elapsed = max(time.monotonic() - started, 1e-9)
    rate = processed / elapsed
    eta_seconds = (total - processed) / rate if rate > 0 else None
    finish = (
        datetime.now().astimezone() + timedelta(seconds=eta_seconds)
        if eta_seconds is not None
        else None
    )
    payload = {
        "event": event,
        "processed": processed,
        "total": total,
        "progress": processed / total,
        "samples_per_second": rate,
        "eta_seconds": eta_seconds,
        "estimated_finish_time": (
            finish.strftime("%Y-%m-%d %H:%M:%S %Z") if finish else None
        ),
        **extra,
    }
    print(json.dumps(payload, sort_keys=True), flush=True)


def _generate_cache_csv(
    input_csv: Path,
    output_csv: Path,
    *,
    fieldnames: Sequence[str],
    cache_root: Path,
    shard_size: int,
    cache_path_column: str,
    cache_index_column: str,
) -> int:
    temporary = output_csv.with_name(f".{output_csv.name}.tmp")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_fields = [*fieldnames, cache_path_column, cache_index_column]
    rows = 0
    with input_csv.open("r", encoding="utf-8-sig", newline="") as source, temporary.open(
        "w", encoding="utf-8", newline=""
    ) as destination:
        reader = csv.DictReader(source)
        writer = csv.DictWriter(destination, fieldnames=output_fields)
        writer.writeheader()
        for row_index, row in enumerate(reader):
            shard_index, cache_index = divmod(row_index, shard_size)
            row[cache_path_column] = str(
                (cache_root / "shards" / f"shard_{shard_index:05d}.npy").resolve()
            )
            row[cache_index_column] = str(cache_index)
            writer.writerow(row)
            rows += 1
    os.replace(temporary, output_csv)
    return rows


def build_pretraining_cache(
    *,
    input_csv: str | Path,
    output_csv: str | Path,
    cache_root: str | Path,
    image_config: ImageTransformConfig,
    allow_truncated_images: bool,
    image_column: str = "path",
    cache_path_column: str = DEFAULT_CACHE_PATH_COLUMN,
    cache_index_column: str = DEFAULT_CACHE_INDEX_COLUMN,
    workers: int = 12,
    shard_size: int = 4096,
) -> dict[str, Any]:
    """Build or resume a float32 prepared-image cache and derivative CSV."""

    if workers < 0:
        raise ValueError("workers must be non-negative.")
    if shard_size <= 0:
        raise ValueError("shard_size must be positive.")
    source = Path(input_csv).expanduser().resolve()
    destination = Path(output_csv).expanduser().resolve()
    root = Path(cache_root).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Input CSV does not exist: {source}")
    if source == destination:
        raise ValueError("Cache CSV must not overwrite the source CSV.")
    fieldnames, row_count = _csv_layout(source, image_column)
    for column in (cache_path_column, cache_index_column):
        if column in fieldnames:
            raise ValueError(f"Input CSV already contains cache column {column!r}.")

    shards_dir = root / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    required_bytes = row_count * image_config.resize**2 * np.dtype(np.float32).itemsize
    completed_bytes = sum(path.stat().st_size for path in shards_dir.glob("shard_*.npy"))
    remaining_bytes = max(required_bytes - completed_bytes, 0)
    one_shard_bytes = shard_size * image_config.resize**2 * np.dtype(np.float32).itemsize
    free_bytes = shutil.disk_usage(root).free
    if free_bytes < remaining_bytes + one_shard_bytes:
        raise OSError(
            "Insufficient free space for cache: "
            f"need at least {remaining_bytes + one_shard_bytes} bytes, "
            f"have {free_bytes} bytes."
        )

    metadata_path = root / "metadata.json"
    specification = {
        "cache_version": CACHE_VERSION,
        "input_csv": str(source),
        "input_csv_sha256": _sha256(source),
        "row_count": row_count,
        "shard_size": shard_size,
        "shard_count": math.ceil(row_count / shard_size),
        "dtype": "float32",
        "shape": [1, image_config.resize, image_config.resize],
        "image_config": asdict(image_config),
        "allow_truncated_images": allow_truncated_images,
        "cache_path_column": cache_path_column,
        "cache_index_column": cache_index_column,
    }
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        for key, value in specification.items():
            if existing.get(key) != value:
                raise ValueError(
                    f"Existing cache metadata mismatch for {key}: "
                    f"{existing.get(key)!r} != {value!r}"
                )
    _write_json_atomic(metadata_path, {**specification, "status": "building"})

    pool: Any = None
    if workers > 0:
        pool = mp.get_context("fork").Pool(
            processes=workers,
            initializer=_init_worker,
            initargs=(asdict(image_config), allow_truncated_images),
        )
    else:
        _init_worker(asdict(image_config), allow_truncated_images)

    started = time.monotonic()
    processed = 0
    try:
        for shard_index, paths in enumerate(
            _path_chunks(source, image_column=image_column, shard_size=shard_size)
        ):
            final_path = shards_dir / f"shard_{shard_index:05d}.npy"
            temporary_path = shards_dir / f".shard_{shard_index:05d}.npy.tmp"
            if _valid_shard(final_path, len(paths), image_config.resize):
                processed += len(paths)
                _progress(
                    event="cache_shard_skipped",
                    processed=processed,
                    total=row_count,
                    started=started,
                    shard=shard_index,
                    records=len(paths),
                )
                continue
            temporary_path.unlink(missing_ok=True)
            shard = np.lib.format.open_memmap(
                temporary_path,
                mode="w+",
                dtype=np.float32,
                shape=(len(paths), 1, image_config.resize, image_config.resize),
            )
            prepared = (
                pool.imap(_prepare_path, paths, chunksize=4)
                if pool is not None
                else map(_prepare_path, paths)
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
            processed += len(paths)
            _progress(
                event="cache_shard_completed",
                processed=processed,
                total=row_count,
                started=started,
                shard=shard_index,
                records=len(paths),
                shard_path=str(final_path),
            )
    except BaseException:
        if pool is not None:
            pool.terminate()
            pool.join()
            pool = None
        raise
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    if processed != row_count:
        raise RuntimeError(f"Cached {processed} records, expected {row_count}.")
    csv_rows = _generate_cache_csv(
        source,
        destination,
        fieldnames=fieldnames,
        cache_root=root,
        shard_size=shard_size,
        cache_path_column=cache_path_column,
        cache_index_column=cache_index_column,
    )
    if csv_rows != row_count:
        raise RuntimeError(f"Wrote {csv_rows} CSV rows, expected {row_count}.")
    elapsed = time.monotonic() - started
    final_metadata = {
        **specification,
        "status": "complete",
        "output_csv": str(destination),
        "output_csv_sha256": _sha256(destination),
        "cache_bytes": sum(path.stat().st_size for path in shards_dir.glob("shard_*.npy")),
        "duration_seconds": elapsed,
    }
    _write_json_atomic(metadata_path, final_metadata)
    return final_metadata


def verify_pretraining_cache(
    *,
    cache_csv: str | Path,
    image_config: ImageTransformConfig,
    allow_truncated_images: bool,
    image_column: str = "path",
    cache_path_column: str = DEFAULT_CACHE_PATH_COLUMN,
    cache_index_column: str = DEFAULT_CACHE_INDEX_COLUMN,
    sample_count: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compare deterministic online preprocessing with cached float32 tensors."""

    csv_path = Path(cache_csv).expanduser().resolve()
    _, row_count = _csv_layout(csv_path, image_column)
    count = min(max(sample_count, 0), row_count)
    generator = np.random.default_rng(seed)
    selected = set(int(value) for value in generator.choice(row_count, count, replace=False))
    transform = MedicalImageTransform(image_config, training=False)
    cache = NpyShardCache()
    checked = 0
    maximum_error = 0.0
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row_index, row in enumerate(csv.DictReader(handle)):
            if row_index not in selected:
                continue
            online = transform.prepare_image(
                load_medical_image(
                    row[image_column],
                    allow_truncated_images=allow_truncated_images,
                )
            )
            cached = cache.load(
                row[cache_path_column],
                int(row[cache_index_column]),
                expected_size=image_config.resize,
            )
            error = float((online - cached).abs().max())
            maximum_error = max(maximum_error, error)
            if error != 0.0:
                raise ValueError(
                    f"Cache mismatch at row {row_index}: max_abs_error={error}, "
                    f"path={row[image_column]}"
                )
            checked += 1
    return {
        "status": "passed",
        "rows": row_count,
        "samples_checked": checked,
        "maximum_absolute_error": maximum_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a resumable float32 prepared-image cache for UHM-FI."
    )
    parser.add_argument("--config", default="configs/pretrain.yaml")
    parser.add_argument("--input-csv", default="data/pretrain_raw.csv")
    parser.add_argument(
        "--output-csv",
        default="data/pretrain.csv",
    )
    parser.add_argument("--cache-root", default="data/cache")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--shard-size", type=int, default=4096)
    parser.add_argument("--verify-samples", type=int, default=1000)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    config = PretrainingConfig.from_yaml(args.config)
    input_csv = Path(args.input_csv).expanduser().resolve()
    output_csv = Path(args.output_csv).expanduser().resolve()
    cache_root = Path(args.cache_root).expanduser().resolve()
    _, rows = _csv_layout(input_csv, config.data.image_column)
    required_bytes = rows * config.data.image.resize**2 * np.dtype(np.float32).itemsize
    summary = {
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "cache_root": str(cache_root),
        "rows": rows,
        "dtype": "float32",
        "shape": [1, config.data.image.resize, config.data.image.resize],
        "estimated_cache_bytes": required_bytes,
        "workers": args.workers,
        "shard_size": args.shard_size,
    }
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "configuration_valid",
                    "cache_started": False,
                    "summary": summary,
                    "safety_gate": "Pass --execute to build the cache.",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    result = build_pretraining_cache(
        input_csv=input_csv,
        output_csv=output_csv,
        cache_root=cache_root,
        image_config=config.data.image,
        allow_truncated_images=config.data.allow_truncated_images,
        image_column=config.data.image_column,
        workers=args.workers,
        shard_size=args.shard_size,
    )
    verification = verify_pretraining_cache(
        cache_csv=output_csv,
        image_config=config.data.image,
        allow_truncated_images=config.data.allow_truncated_images,
        image_column=config.data.image_column,
        sample_count=args.verify_samples,
        seed=config.trainer.seed,
    )
    print(
        json.dumps(
            {"status": "completed", "cache": result, "verification": verification},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
