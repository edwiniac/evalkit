"""
Core data models for EvalKit.

Defines the fundamental types: EvalCase, MetricResult, EvalResult, EvalSuiteResult.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class Verdict(str, Enum):
    """Pass/fail verdict for a metric evaluation."""
    PASS = "pass"
    FAIL = "fail"
    ERROR = "error"
    SKIP = "skip"


@dataclass
class EvalCase:
    """
    A single evaluation test case.
    
    Attributes:
        input: The prompt/question to send to the model.
        expected_output: The ideal/reference answer (optional).
        context: Retrieved context for RAG evaluation (optional).
        metadata: Arbitrary tags for filtering/grouping.
        case_id: Unique identifier (auto-generated if not set).
    """
    input: str
    expected_output: Optional[str] = None
    context: Optional[str | list[str]] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    case_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    @property
    def context_str(self) -> str:
        """Flatten context to a single string."""
        if self.context is None:
            return ""
        if isinstance(self.context, list):
            return "\n\n".join(self.context)
        return self.context

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "input": self.input,
            "expected_output": self.expected_output,
            "context": self.context,
            "metadata": self.metadata,
        }


@dataclass
class ModelResponse:
    """Response from a model for a given input."""
    text: str
    model: str
    latency_ms: float = 0.0
    token_count: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    raw: Any = None  # Original API response

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "model": self.model,
            "latency_ms": round(self.latency_ms, 1),
            "token_count": self.token_count,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost_usd": round(self.cost_usd, 6),
        }


@dataclass
class MetricResult:
    """
    Result of a single metric evaluation on one case.
    
    Attributes:
        metric_name: Name of the metric that produced this result.
        score: Numeric score (0.0 to 1.0).
        verdict: Pass/fail determination.
        reason: Human-readable explanation.
        threshold: The threshold used for pass/fail.
    """
    metric_name: str
    score: float  # 0.0 to 1.0
    verdict: Verdict = Verdict.PASS
    reason: str = ""
    threshold: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "metric_name": self.metric_name,
            "score": round(self.score, 4),
            "verdict": self.verdict.value,
            "reason": self.reason,
            "threshold": self.threshold,
            "metadata": self.metadata,
        }


@dataclass
class CaseResult:
    """Result of evaluating one case across all metrics."""
    case: EvalCase
    response: ModelResponse
    metric_results: list[MetricResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True if all metrics passed."""
        return all(
            r.verdict == Verdict.PASS
            for r in self.metric_results
            if r.verdict != Verdict.SKIP
        )

    @property
    def avg_score(self) -> float:
        scores = [r.score for r in self.metric_results if r.verdict != Verdict.SKIP]
        return sum(scores) / len(scores) if scores else 0.0

    def to_dict(self) -> dict:
        return {
            "case": self.case.to_dict(),
            "response": self.response.to_dict(),
            "metric_results": [r.to_dict() for r in self.metric_results],
            "passed": self.passed,
            "avg_score": round(self.avg_score, 4),
        }


@dataclass
class EvalSuiteResult:
    """
    Result of a complete evaluation suite run.
    
    Contains all case results plus aggregate statistics.
    """
    suite_name: str
    model: str
    case_results: list[CaseResult] = field(default_factory=list)
    run_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_cases(self) -> int:
        return len(self.case_results)

    @property
    def passed_cases(self) -> int:
        return sum(1 for r in self.case_results if r.passed)

    @property
    def failed_cases(self) -> int:
        return self.total_cases - self.passed_cases

    @property
    def pass_rate(self) -> float:
        return self.passed_cases / self.total_cases if self.total_cases else 0.0

    @property
    def avg_score(self) -> float:
        scores = [r.avg_score for r in self.case_results]
        return sum(scores) / len(scores) if scores else 0.0

    @property
    def avg_latency_ms(self) -> float:
        latencies = [r.response.latency_ms for r in self.case_results]
        return sum(latencies) / len(latencies) if latencies else 0.0

    @property
    def total_cost_usd(self) -> float:
        return sum(r.response.cost_usd for r in self.case_results)

    def metric_summary(self) -> dict[str, dict]:
        """Aggregate scores by metric name."""
        metrics: dict[str, list[float]] = {}
        for cr in self.case_results:
            for mr in cr.metric_results:
                if mr.verdict != Verdict.SKIP:
                    metrics.setdefault(mr.metric_name, []).append(mr.score)

        return {
            name: {
                "avg": round(sum(scores) / len(scores), 4),
                "min": round(min(scores), 4),
                "max": round(max(scores), 4),
                "count": len(scores),
            }
            for name, scores in metrics.items()
        }

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "suite_name": self.suite_name,
            "model": self.model,
            "total_cases": self.total_cases,
            "passed_cases": self.passed_cases,
            "failed_cases": self.failed_cases,
            "pass_rate": round(self.pass_rate, 4),
            "avg_score": round(self.avg_score, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "total_cost_usd": round(self.total_cost_usd, 6),
            "metric_summary": self.metric_summary(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "case_results": [r.to_dict() for r in self.case_results],
        }
