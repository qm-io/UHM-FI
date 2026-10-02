"""Build a path-correct, group-disjoint UHM-FI pre-training manifest.

The retained CSV contains valid image-text-label rows, but its absolute paths
refer to an older machine and its non-MIMIC split labels were inverted after a
row-level split.  This module rebases paths and rebuilds splits at the strongest
available grouping level so that a patient/case never crosses train/validation.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


SOURCE_MIMIC = "MIMIC-CXR"
SOURCE_INBREAST = "INBreast"
SOURCE_RADIOPAEDIA = "Radiopaedia"
SOURCE_LOCAL = "Local"

DEFAULT_LOCAL_DATASET_DIRECTORY = "local_dataset"


def _dataset_directories(local_dataset_directory: str) -> dict[str, str]:
    """Return public dataset names plus a caller-supplied local dataset name."""

    local_dataset_directory = local_dataset_directory.strip().strip("/\\")
    if not local_dataset_directory or Path(local_dataset_directory).name != local_dataset_directory:
        raise ValueError("local_dataset_directory must be a single directory name.")
    return {
        "mimic-cxr-jpg-2.1.0.physionet.org": SOURCE_MIMIC,
        "INbreast_Release_1.0": SOURCE_INBREAST,
        "Radiopaedia": SOURCE_RADIOPAEDIA,
        local_dataset_directory: SOURCE_LOCAL,
    }

BIRADS_ORDER = {
    "0": 0,
    "1": 10,
    "2": 20,
    "3": 30,
    "4": 40,
    "4a": 41,
    "4b": 42,
    "4c": 43,
    "5": 50,
    "6": 60,
}

BIRADS_PATTERN = re.compile(
    r"\bbi\s*[- ]?\s*rads?\b"
    r"(?:\s*[:\-]?\s*(?:category)?)?\s*([0-6])\s*([abc]?)",
    re.IGNORECASE,
)
CALCIFICATION_PATTERN = re.compile(r"\b(?:micro)?calcif", re.IGNORECASE)
MASS_PATTERN = re.compile(
    r"\b(?:mass|masses|nodule|nodules|nodular|lump|lumps|lesion|lesions)\b",
    re.IGNORECASE,
)


@dataclass
class GroupInfo:
    source: str
    group_id: str
    image_count: int = 0
    reports: set[str] = field(default_factory=set)
    inbreast_birads: set[str] = field(default_factory=set)
    fine_union: list[int] = field(default_factory=lambda: [0] * 10)
    coarse_union: list[int] = field(default_factory=lambda: [0] * 4)
    official_splits: set[str] = field(default_factory=set)
    stratum: str = ""


def _normalise_birads(value: str) -> str:
    normalised = value.strip().lower().replace(" ", "")
    # Only BI-RADS 4 has the A/B/C subcategories. In prose such as
    # "BI-RADS 2 bilateral", the first letter of the following word must not
    # be interpreted as a category suffix.
    if normalised and normalised[0] != "4":
        normalised = normalised[0]
    if normalised not in BIRADS_ORDER:
        raise ValueError(f"Unsupported BI-RADS value: {value!r}")
    return normalised


def _highest_birads(values: set[str]) -> str:
    if not values:
        return "unknown"
    return max(values, key=lambda value: BIRADS_ORDER[value])


def breast_report_stratum(report: str) -> str:
    """Return a reproducible pathology proxy used only for split stratification."""

    birads = {
        _normalise_birads(f"{number}{suffix.lower()}")
        for number, suffix in BIRADS_PATTERN.findall(report)
    }
    highest = _highest_birads(birads)
    # Findings and impressions are normally at the end of these reports.
    summary = report[-1600:]
    has_mass = bool(MASS_PATTERN.search(summary))
    has_calcification = bool(CALCIFICATION_PATTERN.search(summary))
    if has_mass and has_calcification:
        finding = "mass+calcification"
    elif has_mass:
        finding = "mass"
    elif has_calcification:
        finding = "calcification"
    else:
        finding = "other"
    return f"birads={highest};finding={finding}"


def _parse_binary_vector(value: str, expected: int, *, row_number: int) -> tuple[int, ...]:
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError) as exc:
        raise ValueError(f"Invalid vector at CSV row {row_number}: {value[:80]!r}") from exc
    if not isinstance(parsed, list) or len(parsed) != expected:
        raise ValueError(
            f"Expected {expected} values at CSV row {row_number}, got {parsed!r}."
        )
    try:
        return tuple(int(float(item) > 0.0) for item in parsed)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Non-numeric vector at CSV row {row_number}: {parsed!r}") from exc


def _path_below_dataset_root(
    raw_path: str, data_root: Path, dataset_directories: Mapping[str, str]
) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        try:
            relative = path.relative_to(data_root)
            return data_root / relative
        except ValueError:
            pass

    parts = path.parts
    for index, part in enumerate(parts):
        if part in dataset_directories:
            return data_root.joinpath(*parts[index:])
    raise ValueError(f"Cannot identify a supported dataset in path: {raw_path}")


def identify_source_and_group(
    path: Path, data_root: Path, dataset_directories: Mapping[str, str]
) -> tuple[str, str, dict[str, str]]:
    """Identify the source and strongest available leakage-control group."""

    try:
        parts = path.relative_to(data_root).parts
    except ValueError as exc:
        raise ValueError(f"Path is outside the configured data root: {path}") from exc
    if not parts or parts[0] not in dataset_directories:
        raise ValueError(f"Unsupported pre-training path: {path}")

    source = dataset_directories[parts[0]]
    auxiliary: dict[str, str] = {}
    if source == SOURCE_MIMIC:
        patient = next((part for part in parts if re.fullmatch(r"p\d{8}", part)), None)
        if patient is None:
            raise ValueError(f"Cannot extract a MIMIC subject from path: {path}")
        auxiliary["subject_id"] = patient[1:]
        group_id = f"{SOURCE_MIMIC}/{patient}"
    elif source == SOURCE_INBREAST:
        if len(parts) < 4 or parts[1] != "Studies":
            raise ValueError(f"Unexpected INBreast path layout: {path}")
        study = parts[2]
        auxiliary["file_id"] = path.name.split("_", 1)[0]
        group_id = f"{SOURCE_INBREAST}/{study}"
    elif source == SOURCE_RADIOPAEDIA:
        if len(parts) < 3 or not parts[1].startswith("case_"):
            raise ValueError(f"Unexpected Radiopaedia path layout: {path}")
        # A case may contain multiple studies; keep the complete case together.
        group_id = f"{SOURCE_RADIOPAEDIA}/{parts[1]}"
    else:
        if len(parts) < 3:
            raise ValueError(f"Unexpected local-dataset path layout: {path}")
        group_id = f"{SOURCE_LOCAL}/{parts[1]}"
    return source, group_id, auxiliary


def _load_mimic_subject_splits(path: Path) -> dict[str, str]:
    subject_splits: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"subject_id", "split"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise KeyError(f"MIMIC split CSV is missing columns: {sorted(missing)}")
        for row in reader:
            subject = str(int(row["subject_id"]))
            split = row["split"].strip().lower()
            previous = subject_splits.setdefault(subject, split)
            if previous != split:
                raise ValueError(f"MIMIC subject {subject} crosses official splits.")
    return subject_splits


def _load_inbreast_birads(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        required = {"File Name", "Bi-Rads"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise KeyError(f"INBreast metadata is missing columns: {sorted(missing)}")
        for row in reader:
            file_id = row["File Name"].strip()
            values[file_id] = _normalise_birads(row["Bi-Rads"])
    return values


def _stable_key(seed: int, namespace: str, stratum: str, group_id: str) -> bytes:
    value = f"{seed}\0{namespace}\0{stratum}\0{group_id}".encode("utf-8")
    return hashlib.sha256(value).digest()


def allocate_stratified_validation(
    group_strata: Mapping[str, str],
    *,
    validation_fraction: float,
    seed: int,
    namespace: str,
    group_weights: Mapping[str, int] | None = None,
) -> set[str]:
    """Allocate an exact group fraction and optionally balance sample counts."""

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1).")
    strata: dict[str, list[str]] = defaultdict(list)
    for group_id, stratum in group_strata.items():
        strata[stratum].append(group_id)

    target = int(math.floor(len(group_strata) * validation_fraction + 0.5))
    allocation: dict[str, int] = {}
    remainder_order: list[tuple[float, bytes, str]] = []
    allocated = 0
    for stratum, group_ids in strata.items():
        quota = len(group_ids) * validation_fraction
        maximum = max(0, len(group_ids) - 1)
        count = min(int(math.floor(quota)), maximum)
        allocation[stratum] = count
        allocated += count
        remainder_order.append(
            (
                quota - math.floor(quota),
                _stable_key(seed, namespace, stratum, "__stratum__"),
                stratum,
            )
        )

    remainder_order.sort(key=lambda item: (-item[0], item[1]))
    while allocated < target:
        progressed = False
        for _, _, stratum in remainder_order:
            maximum = max(0, len(strata[stratum]) - 1)
            if allocation[stratum] >= maximum:
                continue
            allocation[stratum] += 1
            allocated += 1
            progressed = True
            if allocated == target:
                break
        if not progressed:
            break

    selected: set[str] = set()
    for stratum, group_ids in strata.items():
        ordered = sorted(
            group_ids,
            key=lambda group_id: _stable_key(seed, namespace, stratum, group_id),
        )
        selected.update(ordered[: allocation[stratum]])
    if len(selected) != target:
        raise RuntimeError(
            f"Could not allocate {target} validation groups for {namespace}; "
            f"allocated {len(selected)}."
        )

    if group_weights is not None:
        missing_weights = set(group_strata) - set(group_weights)
        if missing_weights:
            raise KeyError(f"Missing group weights: {sorted(missing_weights)[:5]}")
        target_weight = int(
            math.floor(
                sum(group_weights[group_id] for group_id in group_strata)
                * validation_fraction
                + 0.5
            )
        )
        selected_weight = sum(group_weights[group_id] for group_id in selected)
        while selected_weight != target_weight:
            current_error = abs(selected_weight - target_weight)
            best_swap: tuple[int, bytes, str, str, int] | None = None
            for stratum, group_ids in strata.items():
                selected_by_weight: dict[int, str] = {}
                training_by_weight: dict[int, str] = {}
                for group_id in group_ids:
                    weight = group_weights[group_id]
                    target = selected_by_weight if group_id in selected else training_by_weight
                    previous = target.get(weight)
                    if previous is None or _stable_key(
                        seed, namespace, stratum, group_id
                    ) < _stable_key(seed, namespace, stratum, previous):
                        target[weight] = group_id
                for selected_group in selected_by_weight.values():
                    for training_group in training_by_weight.values():
                        candidate_weight = (
                            selected_weight
                            - group_weights[selected_group]
                            + group_weights[training_group]
                        )
                        candidate_error = abs(candidate_weight - target_weight)
                        if candidate_error >= current_error:
                            continue
                        tie_breaker = _stable_key(
                            seed,
                            namespace,
                            stratum,
                            f"{selected_group}->{training_group}",
                        )
                        candidate = (
                            candidate_error,
                            tie_breaker,
                            selected_group,
                            training_group,
                            candidate_weight,
                        )
                        if best_swap is None or candidate < best_swap:
                            best_swap = candidate
            if best_swap is None:
                break
            _, _, selected_group, training_group, selected_weight = best_swap
            selected.remove(selected_group)
            selected.add(training_group)
    return selected


def _counter_dict(counter: Counter[str]) -> dict[str, int]:
    return {key: counter[key] for key in sorted(counter)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_pretraining_manifest(
    *,
    input_csv: Path,
    output_csv: Path,
    data_root: Path,
    mimic_split_csv: Path,
    inbreast_metadata_csv: Path,
    local_dataset_directory: str = DEFAULT_LOCAL_DATASET_DIRECTORY,
    seed: int = 42,
    force: bool = False,
) -> dict[str, Any]:
    """Create the corrected manifest and return its audit report."""

    input_csv = input_csv.expanduser().resolve()
    output_csv = output_csv.expanduser().resolve()
    data_root = data_root.expanduser().resolve()
    mimic_split_csv = mimic_split_csv.expanduser().resolve()
    inbreast_metadata_csv = inbreast_metadata_csv.expanduser().resolve()
    dataset_directories = _dataset_directories(local_dataset_directory)
    for required in (input_csv, data_root, mimic_split_csv, inbreast_metadata_csv):
        if not required.exists():
            raise FileNotFoundError(required)
    if output_csv.exists() and not force:
        raise FileExistsError(f"Output exists; pass --force to replace it: {output_csv}")

    csv.field_size_limit(sys.maxsize)
    mimic_subject_splits = _load_mimic_subject_splits(mimic_split_csv)
    inbreast_birads = _load_inbreast_birads(inbreast_metadata_csv)

    groups: dict[str, GroupInfo] = {}
    seen_paths: set[str] = set()
    source_counts: Counter[str] = Counter()
    original_split_counts: dict[str, Counter[str]] = defaultdict(Counter)
    missing_paths: list[str] = []
    duplicate_paths: list[str] = []
    row_count = 0

    with input_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        input_fields = list(reader.fieldnames or ())
        required_fields = {
            "path",
            "report",
            "split",
            "fine_multi_hot",
            "coarse_multi_hot",
        }
        missing_fields = required_fields - set(input_fields)
        if missing_fields:
            raise KeyError(f"Input CSV is missing columns: {sorted(missing_fields)}")

        for row_number, row in enumerate(reader, start=2):
            row_count += 1
            target_path = _path_below_dataset_root(
                row["path"].strip(), data_root, dataset_directories
            )
            path_text = str(target_path)
            if not target_path.is_file():
                if len(missing_paths) < 20:
                    missing_paths.append(path_text)
                continue
            if path_text in seen_paths:
                if len(duplicate_paths) < 20:
                    duplicate_paths.append(path_text)
                continue
            seen_paths.add(path_text)

            source, group_id, auxiliary = identify_source_and_group(
                target_path, data_root, dataset_directories
            )
            source_counts[source] += 1
            original_split_counts[source][row["split"].strip().lower()] += 1
            report = row["report"].strip()
            if not report:
                raise ValueError(f"Empty report at CSV row {row_number}.")
            fine = _parse_binary_vector(row["fine_multi_hot"], 10, row_number=row_number)
            coarse = _parse_binary_vector(row["coarse_multi_hot"], 4, row_number=row_number)

            group = groups.setdefault(group_id, GroupInfo(source=source, group_id=group_id))
            group.image_count += 1
            for index, value in enumerate(fine):
                group.fine_union[index] = max(group.fine_union[index], value)
            for index, value in enumerate(coarse):
                group.coarse_union[index] = max(group.coarse_union[index], value)

            if source in {SOURCE_INBREAST, SOURCE_LOCAL}:
                group.reports.add(report)
            if source == SOURCE_INBREAST:
                file_id = auxiliary["file_id"]
                if file_id not in inbreast_birads:
                    raise KeyError(f"INBreast file {file_id} is absent from metadata.")
                group.inbreast_birads.add(inbreast_birads[file_id])
            elif source == SOURCE_MIMIC:
                subject_id = str(int(auxiliary["subject_id"]))
                if subject_id not in mimic_subject_splits:
                    raise KeyError(f"MIMIC subject {subject_id} is absent from official split CSV.")
                group.official_splits.add(mimic_subject_splits[subject_id])

    if missing_paths:
        raise FileNotFoundError(
            f"At least {len(missing_paths)} manifest paths are missing; examples: {missing_paths}"
        )
    if duplicate_paths:
        raise ValueError(f"Duplicate image paths found; examples: {duplicate_paths}")
    if len(seen_paths) != row_count:
        raise RuntimeError("Manifest row/path accounting failed.")

    assignments: dict[str, str] = {}
    for group in groups.values():
        if group.source == SOURCE_MIMIC:
            if len(group.official_splits) != 1:
                raise ValueError(
                    f"MIMIC patient group {group.group_id} has official splits "
                    f"{sorted(group.official_splits)}."
                )
            official = next(iter(group.official_splits))
            group.stratum = f"official={official}"
            # UHM-FI trains with train/valid only. The official validate and test
            # patients are held out together and are never used for optimisation.
            assignments[group.group_id] = "train" if official == "train" else "valid"
        elif group.source == SOURCE_INBREAST:
            group.stratum = f"max_birads={_highest_birads(group.inbreast_birads)}"
        elif group.source == SOURCE_LOCAL:
            group.stratum = breast_report_stratum(" ".join(sorted(group.reports)))
        else:
            coarse = "".join(str(value) for value in group.coarse_union)
            fine = "".join(str(value) for value in group.fine_union)
            group.stratum = f"coarse={coarse};fine={fine}"

    split_rules = {
        SOURCE_MIMIC: "official patient split; validate+test held out as valid",
        SOURCE_INBREAST: "case-level 80/20, stratified by maximum official BI-RADS",
        SOURCE_RADIOPAEDIA: "case-level 80/20, stratified by hierarchical anatomy labels",
        SOURCE_LOCAL: "patient-level 70/30, stratified by report BI-RADS/finding proxy",
    }
    validation_fractions = {
        SOURCE_INBREAST: 0.20,
        SOURCE_RADIOPAEDIA: 0.20,
        SOURCE_LOCAL: 0.30,
    }
    for source, fraction in validation_fractions.items():
        group_strata = {
            group.group_id: group.stratum
            for group in groups.values()
            if group.source == source
        }
        validation_groups = allocate_stratified_validation(
            group_strata,
            validation_fraction=fraction,
            seed=seed,
            namespace=source,
            group_weights={
                group.group_id: group.image_count
                for group in groups.values()
                if group.source == source
            },
        )
        for group_id in group_strata:
            assignments[group_id] = "valid" if group_id in validation_groups else "train"

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    temporary_csv = output_csv.with_name(f".{output_csv.name}.tmp")
    metadata_fields = ("source", "group_id", "split_stratum", "original_split")
    output_fields = [field for field in input_fields if field not in metadata_fields]
    output_fields.extend(metadata_fields)

    split_counts: dict[str, Counter[str]] = defaultdict(Counter)
    group_split_counts: dict[str, Counter[str]] = defaultdict(Counter)
    try:
        with (
            input_csv.open("r", encoding="utf-8-sig", newline="") as source_handle,
            temporary_csv.open("w", encoding="utf-8", newline="") as output_handle,
        ):
            reader = csv.DictReader(source_handle)
            writer = csv.DictWriter(output_handle, fieldnames=output_fields)
            writer.writeheader()
            for row in reader:
                target_path = _path_below_dataset_root(
                    row["path"].strip(), data_root, dataset_directories
                )
                source, group_id, _ = identify_source_and_group(
                    target_path, data_root, dataset_directories
                )
                split = assignments[group_id]
                original_split = row["split"].strip().lower()
                row["path"] = str(target_path)
                row["split"] = split
                row["source"] = source
                row["group_id"] = group_id
                row["split_stratum"] = groups[group_id].stratum
                row["original_split"] = original_split
                writer.writerow({field: row.get(field, "") for field in output_fields})
                split_counts[source][split] += 1
        temporary_csv.replace(output_csv)
    except Exception:
        temporary_csv.unlink(missing_ok=True)
        raise

    for group in groups.values():
        group_split_counts[group.source][assignments[group.group_id]] += 1

    report: dict[str, Any] = {
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "output_sha256": _sha256(output_csv),
        "data_root": str(data_root),
        "seed": seed,
        "rows": row_count,
        "unique_paths": len(seen_paths),
        "unique_groups": len(groups),
        "all_paths_exist": True,
        "group_disjoint": True,
        "source_counts": _counter_dict(source_counts),
        "original_split_counts": {
            source: _counter_dict(counts)
            for source, counts in sorted(original_split_counts.items())
        },
        "new_split_counts": {
            source: _counter_dict(counts)
            for source, counts in sorted(split_counts.items())
        },
        "new_group_split_counts": {
            source: _counter_dict(counts)
            for source, counts in sorted(group_split_counts.items())
        },
        "split_rules": split_rules,
    }
    audit_path = output_csv.with_name(f"{output_csv.stem}.audit.json")
    with audit_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    report["audit_json"] = str(audit_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebase UHM-FI image paths and rebuild paper-aligned group splits."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/classified_reports_multi_hot.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/pretrain_split.csv"),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/path/to/pretrain-data"),
    )
    parser.add_argument(
        "--local-dataset-directory",
        default=DEFAULT_LOCAL_DATASET_DIRECTORY,
        help="Directory name for the private local dataset (default: local_dataset).",
    )
    parser.add_argument("--mimic-split-csv", type=Path, default=None)
    parser.add_argument("--inbreast-metadata-csv", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    mimic_split_csv = args.mimic_split_csv or (
        args.data_root
        / "mimic-cxr-jpg-2.1.0.physionet.org"
        / "mimic-cxr-2.0.0-split.csv"
    )
    inbreast_metadata_csv = args.inbreast_metadata_csv or (
        args.data_root / "INbreast_Release_1.0" / "INbreast.csv"
    )
    report = prepare_pretraining_manifest(
        input_csv=args.input,
        output_csv=args.output,
        data_root=args.data_root,
        mimic_split_csv=mimic_split_csv,
        inbreast_metadata_csv=inbreast_metadata_csv,
        local_dataset_directory=args.local_dataset_directory,
        seed=args.seed,
        force=args.force,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
