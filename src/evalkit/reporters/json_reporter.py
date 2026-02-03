"""
JSON reporter — machine-readable evaluation results.
"""

import json
from pathlib import Path

from ..models import EvalSuiteResult


class JSONReporter:
    """Exports evaluation results as JSON."""

    def __init__(self, indent: int = 2):
        self._indent = indent

    def to_json(self, result: EvalSuiteResult) -> str:
        """Convert result to JSON string."""
        return json.dumps(result.to_dict(), indent=self._indent, default=str)

    def save(self, result: EvalSuiteResult, path: Path) -> Path:
        """Save result to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(result))
        return path

    def save_comparison(
        self,
        results: dict[str, EvalSuiteResult],
        path: Path,
    ) -> Path:
        """Save comparison results to JSON."""
        data = {
            "comparison": True,
            "models": {name: r.to_dict() for name, r in results.items()},
            "ranking": sorted(
                results.keys(),
                key=lambda k: results[k].avg_score,
                reverse=True,
            ),
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=self._indent, default=str))
        return path
