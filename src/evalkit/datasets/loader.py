"""
Dataset loader — load evaluation cases from files.

Supports JSON, JSONL, and CSV formats.
"""

import csv
import json
import logging
from pathlib import Path
from typing import Optional

from ..models import EvalCase
from ..suite import EvalSuite

logger = logging.getLogger(__name__)


def load_cases_json(path: Path) -> list[EvalCase]:
    """Load cases from a JSON file (list of objects)."""
    data = json.loads(Path(path).read_text())
    if not isinstance(data, list):
        data = [data]
    return [_dict_to_case(d) for d in data]


def load_cases_jsonl(path: Path) -> list[EvalCase]:
    """Load cases from a JSONL file (one JSON per line)."""
    cases = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(_dict_to_case(json.loads(line)))
    return cases


def load_cases_csv(path: Path) -> list[EvalCase]:
    """
    Load cases from a CSV file.
    
    Expected columns: input, expected_output, context (optional).
    """
    cases = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            cases.append(EvalCase(
                input=row.get("input", ""),
                expected_output=row.get("expected_output") or row.get("expected") or None,
                context=row.get("context") or None,
            ))
    return cases


def load_cases(path: str | Path) -> list[EvalCase]:
    """
    Auto-detect format and load cases.
    
    Supports: .json, .jsonl, .csv
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".json":
        return load_cases_json(path)
    elif suffix == ".jsonl":
        return load_cases_jsonl(path)
    elif suffix == ".csv":
        return load_cases_csv(path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Use .json, .jsonl, or .csv")


def load_suite(
    path: str | Path,
    name: Optional[str] = None,
    metrics: Optional[list] = None,
) -> EvalSuite:
    """
    Load a suite from a data file.
    
    Args:
        path: Path to dataset file.
        name: Suite name (defaults to filename).
        metrics: Metrics to add (can be added later).
    """
    path = Path(path)
    cases = load_cases(path)
    suite_name = name or path.stem

    suite = EvalSuite(name=suite_name, cases=cases)
    if metrics:
        suite.add_metrics(metrics)

    logger.info("Loaded suite '%s' with %d cases from %s", suite_name, len(cases), path)
    return suite


def _dict_to_case(d: dict) -> EvalCase:
    """Convert a dict to an EvalCase."""
    input_text = d.get("input", d.get("question", d.get("prompt", "")))
    case_id = d.get("case_id", d.get("id", None))

    kwargs: dict = dict(
        input=input_text,
        expected_output=d.get("expected_output", d.get("expected", d.get("answer", None))),
        context=d.get("context", d.get("contexts", None)),
        metadata=d.get("metadata", {}),
    )
    if case_id is not None:
        kwargs["case_id"] = case_id

    return EvalCase(**kwargs)
