"""Convert existing categorical label lists to the fixed UHM-FI vector order."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import sys
from pathlib import Path

# Keep direct invocations such as
# ``python data_processor/multi_hot_encoder.py`` working from any directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uhm_fi.annotation import COARSE_LABELS, FINE_LABELS


def convert_csv(input_path: str, output_path: str) -> dict[str, object]:
    source_path = Path(input_path)
    destination_path = Path(output_path)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with source_path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        required = {"coarse_labels", "fine_labels"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise KeyError(f"Missing label columns: {sorted(missing)}")
        fields = list(reader.fieldnames or ())
        for field in ("coarse_multi_hot", "fine_multi_hot"):
            if field not in fields:
                fields.append(field)
        with destination_path.open("w", encoding="utf-8", newline="") as destination:
            writer = csv.DictWriter(destination, fieldnames=fields)
            writer.writeheader()
            for row in reader:
                coarse = set(ast.literal_eval(row["coarse_labels"]))
                fine = set(ast.literal_eval(row["fine_labels"]))
                row["coarse_multi_hot"] = str(
                    [int(label in coarse) for label in COARSE_LABELS]
                )
                row["fine_multi_hot"] = str(
                    [int(label in fine) for label in FINE_LABELS]
                )
                writer.writerow(row)
                rows += 1
    return {
        "input": str(source_path.resolve()),
        "output": str(destination_path.resolve()),
        "rows": rows,
        "coarse_label_order": list(COARSE_LABELS),
        "fine_label_order": list(FINE_LABELS),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(convert_csv(args.input, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
