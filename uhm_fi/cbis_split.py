"""Patient-level split repair for CBIS-DDSM mass segmentation.

The retained GLoRIA CSV assigns CC and MLO views independently within the
official Mass-Test partition.  As a result, views from the same patient can be
placed in validation and test.  This module keeps Mass-Training untouched and
deterministically repartitions Mass-Test at patient level while preserving the
existing validation/test image totals as closely as the group sizes permit.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .manifest import allocate_stratified_validation


_PATIENT_PATTERN = re.compile(r"(P_\d+)", flags=re.IGNORECASE)


def cbis_patient_id(subject_id: str) -> str:
    match = _PATIENT_PATTERN.search(subject_id)
    if match is None:
        raise ValueError(f"Cannot extract a CBIS patient ID from {subject_id!r}.")
    return match.group(1).upper()


def _partition(subject_id: str) -> str:
    lowered = subject_id.casefold()
    if lowered.startswith("mass-training_"):
        return "training"
    if lowered.startswith("mass-test_"):
        return "heldout"
    return "unknown"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assign_cbis_patient_splits(
    rows: Sequence[Mapping[str, str]],
    *,
    seed: int = 42,
    validation_fraction: float | None = None,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Return copied rows with a deterministic patient-disjoint split."""

    if not rows:
        raise ValueError("CBIS segmentation rows cannot be empty.")
    required = {"subject_id", "split"}
    missing = required - set(rows[0])
    if missing:
        raise KeyError(f"CBIS rows are missing columns: {sorted(missing)}")

    patient_rows: dict[str, list[int]] = defaultdict(list)
    patient_partitions: dict[str, set[str]] = defaultdict(set)
    copied = [dict(row) for row in rows]
    for index, row in enumerate(copied):
        patient = cbis_patient_id(row["subject_id"])
        patient_rows[patient].append(index)
        patient_partitions[patient].add(_partition(row["subject_id"]))

    ambiguous = {
        patient: sorted(partitions)
        for patient, partitions in patient_partitions.items()
        if len(partitions - {"unknown"}) > 1
    }
    if ambiguous:
        raise ValueError(
            "CBIS patient identifiers cross Mass-Training/Mass-Test namespaces: "
            f"{list(ambiguous.items())[:10]}"
        )

    training_patients: set[str] = set()
    heldout_patients: set[str] = set()
    for patient, indices in patient_rows.items():
        partitions = patient_partitions[patient]
        if "training" in partitions:
            training_patients.add(patient)
        elif "heldout" in partitions:
            heldout_patients.add(patient)
        else:
            original = {copied[index]["split"].strip().lower() for index in indices}
            if "train" in original:
                training_patients.add(patient)
            else:
                heldout_patients.add(patient)

    if training_patients.intersection(heldout_patients):
        raise RuntimeError("A CBIS patient was assigned to both source partitions.")
    if not heldout_patients:
        raise ValueError("No held-out CBIS patients were found.")

    heldout_row_count = sum(len(patient_rows[patient]) for patient in heldout_patients)
    original_valid_rows = sum(
        1
        for patient in heldout_patients
        for index in patient_rows[patient]
        if copied[index]["split"].strip().lower() == "valid"
    )
    if validation_fraction is None:
        validation_fraction = original_valid_rows / heldout_row_count
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("CBIS held-out validation_fraction must be in (0, 1).")

    strata: dict[str, str] = {}
    weights: dict[str, int] = {}
    for patient in heldout_patients:
        labels = sorted(
            {
                copied[index].get("label", "unknown").strip().casefold()
                for index in patient_rows[patient]
            }
        )
        strata[patient] = "+".join(labels) or "unknown"
        weights[patient] = len(patient_rows[patient])
    validation_patients = allocate_stratified_validation(
        strata,
        validation_fraction=validation_fraction,
        seed=seed,
        namespace="CBIS-DDSM-mass-segmentation-heldout",
        group_weights=weights,
    )
    test_patients = heldout_patients - validation_patients

    changed_rows = 0
    for patient, indices in patient_rows.items():
        if patient in training_patients:
            split = "train"
        elif patient in validation_patients:
            split = "valid"
        elif patient in test_patients:
            split = "test"
        else:  # pragma: no cover - guarded by the partition accounting above
            raise RuntimeError(f"Unassigned CBIS patient: {patient}")
        for index in indices:
            changed_rows += int(copied[index]["split"].strip().lower() != split)
            copied[index]["split"] = split

    patient_splits: dict[str, set[str]] = defaultdict(set)
    image_split_counts: Counter[str] = Counter()
    patient_split_counts: Counter[str] = Counter()
    for patient, indices in patient_rows.items():
        splits = {copied[index]["split"] for index in indices}
        patient_splits[patient].update(splits)
        if len(splits) != 1:
            raise RuntimeError(f"CBIS patient still crosses splits: {patient}")
        split = next(iter(splits))
        patient_split_counts[split] += 1
        image_split_counts[split] += len(indices)

    audit = {
        "seed": seed,
        "validation_fraction_of_heldout_images": validation_fraction,
        "rows": len(copied),
        "patients": len(patient_rows),
        "changed_rows": changed_rows,
        "image_split_counts": dict(sorted(image_split_counts.items())),
        "patient_split_counts": dict(sorted(patient_split_counts.items())),
        "cross_split_patients": sum(len(splits) > 1 for splits in patient_splits.values()),
        "source_partition": (
            "Mass-Training remains train; Mass-Test is repartitioned into valid/test "
            "at patient level."
        ),
    }
    return copied, audit


def build_cbis_patient_split_csv(
    *,
    input_csv: str | Path,
    output_csv: str | Path,
    seed: int = 42,
    force: bool = False,
) -> dict[str, Any]:
    source = Path(input_csv).expanduser().resolve()
    destination = Path(output_csv).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if destination.exists() and not force:
        raise FileExistsError(f"Output exists; pass --force to replace it: {destination}")
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or ())
        rows = list(reader)
    repaired, audit = assign_cbis_patient_splits(rows, seed=seed)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(repaired)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    result = {
        "status": "complete",
        "input_csv": str(source),
        "input_csv_sha256": _sha256(source),
        "output_csv": str(destination),
        "output_csv_sha256": _sha256(destination),
        "cache_reused": True,
        "cache_reuse_basis": "Only the split column changed; image/mask cache paths and indices are preserved.",
        **audit,
    }
    audit_path = destination.with_name(f"{destination.stem}.audit.json")
    audit_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result["audit_json"] = str(audit_path)
    return result
