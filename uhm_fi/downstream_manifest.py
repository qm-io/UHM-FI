"""Create path-correct, group-disjoint downstream manifests.

The retained GLoRIA manifests contain machine-specific absolute paths and, for
some tasks, row-level splits that place the same patient/case/image in more
than one split.  This command creates independent UHM-FI manifests without
modifying the source CSVs.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from uhm_fi.cbis_split import assign_cbis_patient_splits, cbis_patient_id


DEFAULT_INPUT_DIR = Path("data/source_csv")
DEFAULT_OUTPUT_DIR = Path("data/downstream_raw")
DEFAULT_DATA_ROOT = Path("/path/to/downstream-data")


def _path_maps(data_root: Path) -> tuple[tuple[str, str], ...]:
    root = data_root.expanduser().resolve()
    return (
        (
            "/path/to/data/rsna-pneumonia-detection-challenge",
            str(root / "rsna-pneumonia-detection-challenge"),
        ),
        ("/path/to/data/MURA-v1.1", str(root / "MURA-v1.1")),
        (
            "/path/to/data/cbis-ddsm-breast-cancer-image-dataset",
            str(root / "cbis-ddsm-breast-cancer-image-dataset"),
        ),
        (
            "/path/to/data/siim_acr_pneumothorax_segmentation",
            str(root / "siim_acr_pneumothorax_segmentation"),
        ),
        (
            "/path/to/data/CBIS-DDSM",
            str(
                root
                / "CBIS-DDSM"
                / "manifest-ZkhPvrLo5216730872708713142"
                / "CBIS-DDSM"
            ),
        ),
        (
            "/path/to/data/CBIS-DDSM/manifest-ZkhPvrLo5216730872708713142/CBIS-DDSM",
            str(
                root
                / "CBIS-DDSM"
                / "manifest-ZkhPvrLo5216730872708713142"
                / "CBIS-DDSM"
            ),
        ),
    )


def _remap(value: str, maps: Iterable[tuple[str, str]]) -> str:
    result = str(value)
    for old, new in maps:
        result = result.replace(old, new)
    return result


def _read(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or ())
        if not fields:
            raise ValueError(f"CSV has no header: {path}")
        return fields, list(reader)


def _write(path: Path, fields: list[str], rows: list[Mapping[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _majority_split(rows: list[Mapping[str, str]], column: str) -> str:
    counts = Counter(str(row[column]).strip().lower() for row in rows)
    priority = {"train": 0, "valid": 1, "test": 2}
    return max(counts, key=lambda value: (counts[value], -priority.get(value, 9)))


def _assign_group_splits(
    rows: list[dict[str, str]],
    *,
    group_fn: Any,
    split_column: str,
) -> tuple[dict[int, str], dict[str, int]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[str(group_fn(row))].append(index)
    assignments: dict[int, str] = {}
    cross_groups = 0
    for members in grouped.values():
        split = _majority_split([rows[index] for index in members], split_column)
        original = {str(rows[index][split_column]).strip().lower() for index in members}
        if len(original) > 1:
            cross_groups += 1
        for index in members:
            assignments[index] = split
    return assignments, {"groups": len(grouped), "cross_split_groups_repaired": cross_groups}


def _patient_group(path: str) -> str:
    match = re.search(r"/(patient[^/]+)/", path, flags=re.IGNORECASE)
    return match.group(1).lower() if match else path


def _cbis_case_group(path: str) -> str:
    # JPEG manifests contain one case directory per image; using the parent
    # avoids duplicate rows while keeping distinct cases separate.
    return str(Path(path).parent)


def _collapse_binary_images(
    rows: list[dict[str, str]],
    *,
    path_column: str,
    label_column: str,
    split_column: str,
) -> tuple[list[dict[str, str]], dict[str, int]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row[path_column]].append(row)
    output: list[dict[str, str]] = []
    repaired = 0
    for path, members in grouped.items():
        row = dict(members[0])
        labels = [int(float(member[label_column])) for member in members]
        row[label_column] = str(max(labels))
        splits = {str(member[split_column]).strip().lower() for member in members}
        row[split_column] = _majority_split(members, split_column)
        repaired += int(len(splits) > 1)
        output.append(row)
    return output, {
        "input_rows": len(rows),
        "output_rows": len(output),
        "duplicate_image_groups_collapsed": len(rows) - len(output),
        "cross_split_image_groups_repaired": repaired,
    }


def _prepare_classification(
    name: str,
    rows: list[dict[str, str]],
    maps: tuple[tuple[str, str], ...],
) -> tuple[list[str], list[dict[str, str]], dict[str, Any]]:
    fields = list(rows[0])
    path_column = "path"
    split_column = "split"
    for row in rows:
        row[path_column] = _remap(row[path_column], maps)
    if name == "mura":
        assignments, audit = _assign_group_splits(
            rows,
            group_fn=lambda row: _patient_group(row[path_column]),
            split_column=split_column,
        )
        for index, row in enumerate(rows):
            row[split_column] = assignments[index]
        audit.update({"grouping": "MURA patient_id", "deduplicated": False})
        return fields, rows, audit
    collapsed, audit = _collapse_binary_images(
        rows,
        path_column=path_column,
        label_column="label",
        split_column=split_column,
    )
    if name == "cbis_cls":
        # Make the split decision at case-directory level as well, then retain
        # one row per image for the classifier.
        assignments, group_audit = _assign_group_splits(
            collapsed,
            group_fn=lambda row: _cbis_case_group(row[path_column]),
            split_column=split_column,
        )
        for index, row in enumerate(collapsed):
            row[split_column] = assignments[index]
        audit.update(group_audit)
        audit["grouping"] = "CBIS case directory"
    else:
        audit["grouping"] = "RSNA image_id"
    audit["deduplicated"] = True
    return fields, collapsed, audit


def _prepare_cbis_seg(
    rows: list[dict[str, str]], maps: tuple[tuple[str, str], ...], seed: int = 42
) -> tuple[list[str], list[dict[str, str]], dict[str, Any]]:
    fields = list(rows[0])
    for row in rows:
        row["full_mammo_path"] = _remap(row["full_mammo_path"], maps)
        row["roi_mask_path"] = _remap(row["roi_mask_path"], maps)
    rows, audit = assign_cbis_patient_splits(rows, seed=seed)
    audit["grouping"] = "CBIS patient_id parsed from subject_id"
    audit["deduplicated"] = False
    return fields, rows, audit


def _prepare_siim(
    rows: list[dict[str, str]], maps: tuple[tuple[str, str], ...]
) -> tuple[list[str], list[dict[str, str]], dict[str, Any]]:
    fields = list(rows[0])
    for row in rows:
        row["Path"] = _remap(row["Path"], maps)
    assignments, audit = _assign_group_splits(
        rows,
        group_fn=lambda row: row["ImageId"],
        split_column="Split",
    )
    for index, row in enumerate(rows):
        row["Split"] = assignments[index]
    audit["grouping"] = "SIIM ImageId"
    audit["deduplicated"] = False
    return fields, rows, audit


def _audit_rows(
    rows: list[Mapping[str, str]],
    *,
    path_column: str,
    split_column: str,
    group_fn: Any,
    extra_path_columns: Iterable[str] = (),
) -> dict[str, Any]:
    split_counts = Counter(str(row[split_column]).strip().lower() for row in rows)
    path_splits: dict[str, set[str]] = defaultdict(set)
    group_splits: dict[str, set[str]] = defaultdict(set)
    missing: list[str] = []
    missing_by_column: dict[str, list[str]] = defaultdict(list)
    path_columns = (path_column, *tuple(extra_path_columns))
    for row in rows:
        path = str(row[path_column])
        split = str(row[split_column]).strip().lower()
        path_splits[path].add(split)
        group_splits[str(group_fn(row))].add(split)
        if not Path(path).exists() and len(missing) < 20:
            missing.append(path)
        for column in path_columns:
            value = str(row.get(column, "")).strip()
            if not value:
                continue
            values: list[str]
            if value.startswith("["):
                try:
                    parsed = ast.literal_eval(value)
                    values = [str(item) for item in parsed]
                except (ValueError, SyntaxError):
                    values = [value]
            else:
                values = [value]
            for candidate in values:
                if not Path(candidate).exists() and len(missing_by_column[column]) < 20:
                    missing_by_column[column].append(candidate)
    return {
        "rows": len(rows),
        "split_counts": dict(split_counts),
        "unique_paths": len(path_splits),
        "cross_split_paths": sum(len(value) > 1 for value in path_splits.values()),
        "groups": len(group_splits),
        "cross_split_groups": sum(len(value) > 1 for value in group_splits.values()),
        "missing_path_examples": missing,
        "missing_path_examples_by_column": dict(missing_by_column),
    }


def prepare(
    *, input_dir: Path, output_dir: Path, data_root: Path, execute: bool, seed: int = 42
) -> dict[str, Any]:
    maps = _path_maps(data_root)
    specifications = {
        "rsna": ("rsna_pneumonia.csv", _prepare_classification, "path", "split", lambda row: row["path"]),
        "mura": ("mura.csv", _prepare_classification, "path", "split", lambda row: _patient_group(row["path"])),
        "cbis_cls": ("cbis_ddsm_cls.csv", _prepare_classification, "path", "split", lambda row: _cbis_case_group(row["path"])),
        "cbis_seg": ("cbis_ddsm_seg.csv", _prepare_cbis_seg, "full_mammo_path", "split", lambda row: cbis_patient_id(row["subject_id"])),
        "siim": ("pneumothorax.csv", _prepare_siim, "Path", "Split", lambda row: row["ImageId"]),
    }
    summary: dict[str, Any] = {"status": "configuration_valid", "datasets": {}}
    if not execute:
        for name, (filename, *_rest) in specifications.items():
            source = input_dir / filename
            fields, rows = _read(source)
            summary["datasets"][name] = {"input_csv": str(source.resolve()), "input_rows": len(rows), "fields": fields}
        return summary
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, (filename, prepare_fn, path_column, split_column, group_fn) in specifications.items():
        source = input_dir / filename
        fields, rows = _read(source)
        if name in {"rsna", "mura", "cbis_cls"}:
            fields, rows, transform_audit = prepare_fn(name, rows, maps)
        elif name == "cbis_seg":
            fields, rows, transform_audit = prepare_fn(rows, maps, seed)
        else:
            fields, rows, transform_audit = prepare_fn(rows, maps)
        output = output_dir / filename
        _write(output, fields, rows)
        extra_path_columns = ("roi_mask_path",) if name == "cbis_seg" else ()
        audit = _audit_rows(
            rows,
            path_column=path_column,
            split_column=split_column,
            group_fn=group_fn,
            extra_path_columns=extra_path_columns,
        )
        summary["datasets"][name] = {
            "input_csv": str(source.resolve()),
            "output_csv": str(output.resolve()),
            "path_maps": list(maps),
            "transform": transform_audit,
            "audit": audit,
        }
    summary["status"] = "completed"
    audit_path = output_dir / "manifest_audit.json"
    audit_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary["audit_path"] = str(audit_path.resolve())
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare path-correct group-disjoint downstream CSVs.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare(input_dir=args.input_dir, output_dir=args.output_dir, data_root=args.data_root, execute=args.execute, seed=args.seed), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
