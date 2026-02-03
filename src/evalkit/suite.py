"""
Evaluation suite — a collection of test cases and metrics.

A suite defines WHAT to evaluate and HOW to score it.
The runner handles actually executing it against models.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .metrics.base import EvalMetric
from .models import EvalCase


@dataclass
class EvalSuite:
    """
    A named collection of evaluation cases and metrics.

    Usage:
        suite = EvalSuite(
            name="RAG Quality",
            cases=[
                EvalCase(input="What is X?", expected_output="Y", context="..."),
            ],
            metrics=[ExactMatch(), ContainsAny(keywords=["Y"])],
        )
    """

    name: str
    cases: list[EvalCase] = field(default_factory=list)
    metrics: list[EvalMetric] = field(default_factory=list)
    description: str = ""
    metadata: dict = field(default_factory=dict)

    def add_case(self, case: EvalCase) -> EvalSuite:
        """Add a case (fluent API)."""
        self.cases.append(case)
        return self

    def add_cases(self, cases: list[EvalCase]) -> EvalSuite:
        """Add multiple cases (fluent API)."""
        self.cases.extend(cases)
        return self

    def add_metric(self, metric: EvalMetric) -> EvalSuite:
        """Add a metric (fluent API)."""
        self.metrics.append(metric)
        return self

    def add_metrics(self, metrics: list[EvalMetric]) -> EvalSuite:
        """Add multiple metrics (fluent API)."""
        self.metrics.extend(metrics)
        return self

    def __len__(self) -> int:
        return len(self.cases)

    def __repr__(self) -> str:
        return (
            f"EvalSuite(name='{self.name}', "
            f"cases={len(self.cases)}, "
            f"metrics={len(self.metrics)})"
        )
