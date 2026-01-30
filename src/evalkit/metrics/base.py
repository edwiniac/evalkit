"""
Base metric interface.

All metrics (deterministic, LLM-judge, statistical) implement this ABC.
"""

from abc import ABC, abstractmethod
from typing import Optional

from ..models import EvalCase, MetricResult, ModelResponse, Verdict


class EvalMetric(ABC):
    """
    Abstract base class for evaluation metrics.
    
    Subclasses implement `score()` to evaluate a model response
    against a test case. Return a MetricResult with a 0-1 score.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        threshold: float = 0.5,
    ):
        self._name = name or self.__class__.__name__
        self._threshold = threshold

    @property
    def name(self) -> str:
        return self._name

    @property
    def threshold(self) -> float:
        return self._threshold

    @abstractmethod
    async def score(
        self,
        case: EvalCase,
        response: ModelResponse,
    ) -> MetricResult:
        """
        Evaluate a model response for a given case.
        
        Args:
            case: The evaluation test case.
            response: The model's response.
            
        Returns:
            MetricResult with score, verdict, and reason.
        """
        ...

    def _make_result(
        self,
        score: float,
        reason: str = "",
        **metadata,
    ) -> MetricResult:
        """Helper to create a MetricResult with auto verdict."""
        verdict = Verdict.PASS if score >= self._threshold else Verdict.FAIL
        return MetricResult(
            metric_name=self._name,
            score=score,
            verdict=verdict,
            reason=reason,
            threshold=self._threshold,
            metadata=metadata,
        )

    def _error_result(self, error: str) -> MetricResult:
        """Create an error result."""
        return MetricResult(
            metric_name=self._name,
            score=0.0,
            verdict=Verdict.ERROR,
            reason=f"Error: {error}",
            threshold=self._threshold,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(threshold={self._threshold})"
