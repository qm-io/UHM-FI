"""Generate the canonical NumPy-backed downstream configurations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


SPECS = {
    "rsna_linear": (
        "downstream_rsna_linear.yaml",
        "rsna_pneumonia.csv",
        False,
    ),
    "rsna_finetune": (
        "downstream_rsna_finetune.yaml",
        "rsna_pneumonia.csv",
        False,
    ),
    "mura_linear": (
        "downstream_mura_linear.yaml",
        "mura.csv",
        False,
    ),
    "mura_finetune": (
        "downstream_mura_finetune.yaml",
        "mura.csv",
        False,
    ),
    "cbis_cls_linear": (
        "downstream_cbis_linear.yaml",
        "cbis_ddsm_cls.csv",
        False,
    ),
    "cbis_cls_finetune": (
        "downstream_cbis_finetune.yaml",
        "cbis_ddsm_cls.csv",
        False,
    ),
    "cbis_segmentation": (
        "downstream_cbis_segmentation.yaml",
        "cbis_ddsm_seg.csv",
        True,
    ),
    "siim_segmentation": (
        "downstream_siim_segmentation.yaml",
        "pneumothorax.csv",
        True,
    ),
}


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return value


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(value, handle, sort_keys=False, allow_unicode=True)
    temporary.replace(path)


def prepare(
    *,
    project_root: Path,
    cache_manifest_dir: Path,
    checkpoint: Path,
    output_dir: Path,
    execute: bool,
) -> dict[str, Any]:
    cache_manifest_dir = cache_manifest_dir.expanduser().resolve()
    checkpoint = checkpoint.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not execute:
        return {
            "status": "configuration_valid",
            "configs": list(SPECS),
            "checkpoint": str(checkpoint),
            "cache_manifest_dir": str(cache_manifest_dir),
        }
    result: dict[str, Any] = {"status": "completed", "configs": []}
    for name, (canonical_filename, csv_name, segmentation) in SPECS.items():
        canonical_path = project_root / "configs" / canonical_filename
        cached = _load(canonical_path)
        cached["pretrained_checkpoint"] = str(checkpoint)
        if name == "siim_segmentation":
            # Keep the corrected relative-offset RLE cache visibly separate
            # from the obsolete absolute-start cache produced before the SIIM
            # decoder was aligned with the original GLoRIA implementation.
            cached_csv_name = "pneumothorax_mask_relative_rle.csv"
        else:
            cached_csv_name = (
                f"{Path(csv_name).stem}_mask.csv" if segmentation else csv_name
            )
        cached["data"]["csv_path"] = str(cache_manifest_dir / cached_csv_name)
        cached["data"]["cache_path_column"] = "cache_path"
        cached["data"]["cache_index_column"] = "cache_index"
        if segmentation:
            cached["data"]["mask_cache_path_column"] = "mask_cache_path"
            cached["data"]["mask_cache_index_column"] = "mask_cache_index"
        if name == "siim_segmentation":
            cached["data"]["balance_segmentation_train"] = True
        cached["data"]["loader"]["num_workers"] = 4
        cached["data"]["loader"]["prefetch_factor"] = 2
        cached["trainer"]["log_every_steps"] = 50
        cached["trainer"]["output_dir"] = str(output_dir / name)
        _write(canonical_path, cached)
        result["configs"].append(str(canonical_path))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate independent downstream YAML configs.")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--cache-manifest-dir", type=Path, default=Path("data/downstream"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("output/pretrain_full_50epoch_20260731/best.pt"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("output/downstream"))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                project_root=args.project_root.resolve(),
                cache_manifest_dir=args.cache_manifest_dir,
                checkpoint=args.checkpoint,
                output_dir=args.output_dir,
                execute=args.execute,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
