"""Compatibility wrapper for the reproducible annotation pipeline.

Prefer ``python annotate_reports.py --input ... --output ...`` from the project
root.  This file remains so older commands importing ``classify_medical_report``
continue to work without pandas.
"""

from __future__ import annotations

import sys
from pathlib import Path

# When this compatibility script is executed by path, Python adds
# ``data_processor/`` rather than the project root to ``sys.path``.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uhm_fi.annotation import annotate_report, main


def classify_medical_report(report_text: str) -> tuple[list[str], list[str]]:
    annotation = annotate_report(report_text)
    return annotation["coarse_labels"], annotation["fine_labels"]


if __name__ == "__main__":
    main()
