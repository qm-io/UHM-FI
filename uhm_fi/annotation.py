"""Deterministic report annotation used to create UHM-FI condition labels.

No recoverable LLM invocation or prompt exists in the retained project.  This
module therefore implements the evidenced keyword/regular-expression pipeline
only and fixes the label order used by the 10-D fine and 4-D coarse vectors.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


FINE_LABELS = (
    "Breast",
    "Cardiac",
    "Central Nervous System",
    "Chest",
    "Gastrointestinal",
    "Head & Neck",
    "Musculoskeletal",
    "Spine",
    "Uncategorized",
    "Vascular",
)
COARSE_LABELS = ("Bone", "Breast", "Chest", "Uncategorized")


CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "Breast": (
        r"breasts?", r"mammo", r"mammogram", r"mammography", r"nipple",
        r"areola", r"axilla", r"axillary", r"pectoral", r"fibroadenoma",
        r"breast calcification", r"bi-rads", r"birads", r"ductal", r"lobular",
        r"implant", r"augmentation", r"gynecomastia", r"galactogram",
        r"tomosynthesis", r"breast mri", r"breast ultrasound",
    ),
    "Cardiac": (
        r"cardiac", r"heart", r"coronary", r"myocard", r"pericard",
        r"ventricle", r"atrium", r"aortic valve", r"mitral valve",
        r"tricuspid", r"pulmonary valve", r"septum", r"ejection fraction",
        r"cardiomegaly", r"echocardiogram", r"ischemia", r"infarct",
        r"pacemaker", r"cardiac mri",
    ),
    "Central Nervous System": (
        r"brain", r"cerebr", r"cerebell", r"central nervous", r"pituitary",
        r"hypothalamus", r"thalamus", r"basal gangli", r"corpus callosum",
        r"brainstem", r"midbrain", r"pons", r"medulla", r"mening",
        r"white matter", r"gray matter", r"hydrocephalus", r"stroke",
        r"intracranial hemorrhage", r"ct head", r"mri brain",
    ),
    "Chest": (
        r"chest", r"thorax", r"thoracic", r"lungs?", r"pulmon", r"pleura",
        r"mediastin", r"hilar", r"airway", r"trachea", r"bronch",
        r"alveol", r"lung nodule", r"lung mass", r"infiltrate",
        r"consolidation", r"atelectasis", r"pneumothorax", r"effusion",
        r"empyema", r"fibrosis", r"emphysema", r"copd", r"tuberculosis",
        r"cxr", r"ct chest",
    ),
    "Gastrointestinal": (
        r"gastrointestin", r"bowel", r"colon", r"rectum", r"esophag",
        r"stomach", r"gastric", r"duoden", r"jejun", r"ileum", r"cecum",
        r"appendix", r"periton", r"hernia", r"obstruction", r"diverticul",
        r"crohn", r"ulcerative colitis", r"enteritis", r"gastritis",
        r"small bowel", r"large bowel", r"ct abdomen", r"mri abdomen",
    ),
    "Head & Neck": (
        r"head", r"neck", r"sinus", r"nasal", r"oral", r"pharynx",
        r"larynx", r"thyroid", r"salivary", r"parotid", r"submandibular",
        r"tongue", r"palate", r"tonsil", r"vocal cord", r"mandible",
        r"maxilla", r"orbit", r"ocular", r"mastoid", r"skull base",
        r"nasopharynx", r"oropharynx", r"hypopharynx", r"ct neck",
    ),
    "Musculoskeletal": (
        r"musculoskel", r"bones?", r"joint", r"muscle", r"tendon",
        r"ligament", r"shoulder", r"knee", r"hip", r"elbow", r"wrist",
        r"ankle", r"foot", r"hand", r"clavicle", r"scapula", r"humerus",
        r"radius", r"ulna", r"femur", r"tibia", r"fibula", r"patella",
        r"pelvis", r"fracture", r"dislocation", r"arthritis",
        r"osteoporosis", r"osteomyelitis", r"prosthesis", r"orthopedic",
        r"ct bone", r"mri joint",
    ),
    "Spine": (
        r"spine", r"spinal", r"vertebr", r"cervical spine",
        r"thoracic spine", r"lumbar spine", r"sacrum", r"coccyx",
        r"intervertebral", r"cauda equina", r"conus", r"nerve root",
        r"foramen", r"facet", r"lamina", r"spinous process", r"stenosis",
        r"herniation", r"protrusion", r"extrusion", r"syringomyelia",
        r"mri spine", r"ct spine",
    ),
    "Vascular": (
        r"vascular", r"veins?", r"venous", r"artery", r"arterial", r"aorta",
        r"carotid", r"femoral", r"popliteal", r"iliac", r"subclavian",
        r"brachial", r"radial", r"ulnar", r"mesenteric", r"renal artery",
        r"portal", r"vena cava", r"jugular", r"varicose", r"thrombosis",
        r"thromboembolism", r"aneurysm", r"dissection", r"atherosclerosis",
        r"angioplasty", r"angiogram",
    ),
}


COARSE_MAPPING = {
    "Uncategorized": "Uncategorized",
    "Breast": "Breast",
    "Chest": "Chest",
    "Musculoskeletal": "Bone",
    "Spine": "Bone",
    "Cardiac": "Uncategorized",
    "Central Nervous System": "Uncategorized",
    "Gastrointestinal": "Uncategorized",
    "Head & Neck": "Uncategorized",
    "Vascular": "Uncategorized",
}


COMPILED_KEYWORDS = {
    category: tuple(re.compile(rf"\b(?:{pattern})\b", re.IGNORECASE) for pattern in patterns)
    for category, patterns in CATEGORY_KEYWORDS.items()
}


def extract_findings_and_impression(report: str) -> str:
    """Return explicit Findings/Impression sections when headings are present."""

    normalized = report.replace("\r\n", "\n").replace("\r", "\n")
    pattern = re.compile(
        r"(?is)\b(findings?|impression)\s*:\s*(.*?)(?=\n\s*[A-Z][A-Z ]+\s*:|\Z)"
    )
    sections = [match.group(2).strip() for match in pattern.finditer(normalized)]
    return " ".join(section for section in sections if section) or report.strip()


def _context_allowed(category: str, text: str) -> bool:
    if category == "Breast":
        return any(token in text for token in ("breast", "mammo", "birads", "bi-rads"))
    if category == "Chest":
        return any(token in text for token in ("lung", "thorax", "pleura", "chest"))
    if category == "Vascular":
        return any(
            token in text
            for token in ("artery", "arterial", "vein", "vascular", "thrombosis", "aneurysm")
        )
    return True


def annotate_report(report: str) -> dict[str, Any]:
    text = extract_findings_and_impression(str(report)).lower()
    fine = [
        category
        for category, patterns in COMPILED_KEYWORDS.items()
        if any(pattern.search(text) for pattern in patterns)
        and _context_allowed(category, text)
    ]
    if not fine:
        fine = ["Uncategorized"]
    coarse = sorted({COARSE_MAPPING[category] for category in fine})
    if len(coarse) > 1 and "Uncategorized" in coarse:
        coarse.remove("Uncategorized")
    fine = sorted(fine, key=FINE_LABELS.index)
    coarse = sorted(coarse, key=COARSE_LABELS.index)
    return {
        "fine_labels": fine,
        "coarse_labels": coarse,
        "fine_multi_hot": [int(label in fine) for label in FINE_LABELS],
        "coarse_multi_hot": [int(label in coarse) for label in COARSE_LABELS],
    }


def annotate_csv(
    input_path: str | Path,
    output_path: str | Path,
    text_column: str = "report",
) -> dict[str, Any]:
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    counts = {label: 0 for label in FINE_LABELS}
    rows = 0
    with input_path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if text_column not in (reader.fieldnames or ()):
            raise KeyError(f"Missing text column {text_column!r} in {input_path}")
        fields = list(reader.fieldnames or ())
        additions = (
            "coarse_labels",
            "fine_labels",
            "coarse_multi_hot",
            "fine_multi_hot",
        )
        fields.extend(field for field in additions if field not in fields)
        with output_path.open("w", encoding="utf-8", newline="") as destination:
            writer = csv.DictWriter(destination, fieldnames=fields)
            writer.writeheader()
            for row in reader:
                annotation = annotate_report(row[text_column])
                row.update({key: str(value) for key, value in annotation.items()})
                writer.writerow(row)
                rows += 1
                for label in annotation["fine_labels"]:
                    counts[label] += 1
    return {
        "input": str(input_path.resolve()),
        "output": str(output_path.resolve()),
        "rows": rows,
        "fine_label_counts": counts,
        "fine_label_order": list(FINE_LABELS),
        "coarse_label_order": list(COARSE_LABELS),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate UHM-FI report labels.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--text-column", default="report")
    args = parser.parse_args()
    print(
        json.dumps(
            annotate_csv(args.input, args.output, args.text_column),
            indent=2,
            sort_keys=True,
        )
    )
