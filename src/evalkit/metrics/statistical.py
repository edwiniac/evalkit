"""
Statistical metrics — BLEU, ROUGE, and semantic similarity.

These metrics compare the model response to reference text using
established NLP metrics. No LLM calls needed.
"""

import logging
from typing import Optional

from ..models import EvalCase, MetricResult, ModelResponse
from .base import EvalMetric

logger = logging.getLogger(__name__)


class BLEUScore(EvalMetric):
    """
    BLEU score — measures n-gram overlap with expected output.

    Standard MT metric adapted for LLM evaluation.
    Score range: 0.0 (no overlap) to 1.0 (identical).
    """

    def __init__(self, threshold: float = 0.3, name: Optional[str] = None):
        super().__init__(name=name or "BLEUScore", threshold=threshold)

    async def score(self, case: EvalCase, response: ModelResponse) -> MetricResult:
        if not case.expected_output:
            return self._error_result("No expected_output for BLEU calculation")

        try:
            from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
        except ImportError:
            return self._error_result("nltk required: pip install nltk")

        reference = case.expected_output.lower().split()
        hypothesis = response.text.lower().split()

        if not hypothesis:
            return self._make_result(0.0, reason="Empty response")

        try:
            smoothing = SmoothingFunction().method1
            bleu = sentence_bleu(
                [reference],
                hypothesis,
                weights=(0.5, 0.5),  # Bigram BLEU
                smoothing_function=smoothing,
            )
            return self._make_result(
                score=bleu,
                reason=f"BLEU score: {bleu:.4f}",
            )
        except Exception as e:
            return self._error_result(f"BLEU calculation failed: {e}")


class ROUGEScore(EvalMetric):
    """
    ROUGE-L score — measures longest common subsequence overlap.

    Good for summarization evaluation.
    Score range: 0.0 to 1.0.
    """

    def __init__(self, threshold: float = 0.3, name: Optional[str] = None):
        super().__init__(name=name or "ROUGEScore", threshold=threshold)

    async def score(self, case: EvalCase, response: ModelResponse) -> MetricResult:
        if not case.expected_output:
            return self._error_result("No expected_output for ROUGE calculation")

        try:
            from rouge_score.rouge_scorer import RougeScorer
        except ImportError:
            return self._error_result("rouge-score required: pip install rouge-score")

        try:
            scorer = RougeScorer(["rougeL"], use_stemmer=True)
            scores = scorer.score(case.expected_output, response.text)
            f1 = scores["rougeL"].fmeasure

            return self._make_result(
                score=f1,
                reason=f"ROUGE-L F1: {f1:.4f}",
                precision=scores["rougeL"].precision,
                recall=scores["rougeL"].recall,
            )
        except Exception as e:
            return self._error_result(f"ROUGE calculation failed: {e}")


class SemanticSimilarity(EvalMetric):
    """
    Semantic similarity — cosine similarity of text embeddings.

    Uses sentence-transformers for embedding, falls back to
    simple word overlap if unavailable.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        threshold: float = 0.7,
        name: Optional[str] = None,
    ):
        super().__init__(name=name or "SemanticSimilarity", threshold=threshold)
        self._model_name = model_name
        self._encoder = None

    async def score(self, case: EvalCase, response: ModelResponse) -> MetricResult:
        if not case.expected_output:
            return self._error_result("No expected_output for similarity calculation")

        # Try sentence-transformers first
        try:
            similarity = self._embedding_similarity(case.expected_output, response.text)
            return self._make_result(
                score=similarity,
                reason=f"Semantic similarity: {similarity:.4f}",
                method="embedding",
            )
        except ImportError:
            pass

        # Fallback: Jaccard word overlap
        similarity = self._jaccard_similarity(case.expected_output, response.text)
        return self._make_result(
            score=similarity,
            reason=(
                f"Word overlap similarity: {similarity:.4f}"
                " (install sentence-transformers for better results)"
            ),
            method="jaccard",
        )

    def _embedding_similarity(self, text_a: str, text_b: str) -> float:
        """Compute cosine similarity using sentence-transformers."""
        import numpy as np
        from sentence_transformers import SentenceTransformer

        if self._encoder is None:
            self._encoder = SentenceTransformer(self._model_name)

        embeddings = self._encoder.encode([text_a, text_b])
        cos_sim = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        return float(max(0.0, min(1.0, cos_sim)))

    @staticmethod
    def _jaccard_similarity(text_a: str, text_b: str) -> float:
        """Simple word-level Jaccard similarity."""
        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())
        if not words_a and not words_b:
            return 1.0
        if not words_a or not words_b:
            return 0.0
        intersection = words_a & words_b
        union = words_a | words_b
        return len(intersection) / len(union)


class LatencyMetric(EvalMetric):
    """
    Measures response latency.

    Score = 1.0 if under target, degrades as latency increases.
    """

    def __init__(
        self,
        target_ms: float = 1000,
        max_ms: float = 5000,
        threshold: float = 0.5,
        name: Optional[str] = None,
    ):
        super().__init__(name=name or "Latency", threshold=threshold)
        self._target = target_ms
        self._max = max_ms

    async def score(self, case: EvalCase, response: ModelResponse) -> MetricResult:
        latency = response.latency_ms

        if latency <= self._target:
            score = 1.0
        elif latency >= self._max:
            score = 0.0
        else:
            score = 1.0 - (latency - self._target) / (self._max - self._target)

        return self._make_result(
            score=score,
            reason=f"Latency: {latency:.0f}ms (target: {self._target:.0f}ms)",
            latency_ms=latency,
        )


class CostMetric(EvalMetric):
    """
    Measures response cost.

    Score = 1.0 if under budget, degrades as cost increases.
    """

    def __init__(
        self,
        budget_usd: float = 0.01,
        max_usd: float = 0.10,
        threshold: float = 0.5,
        name: Optional[str] = None,
    ):
        super().__init__(name=name or "Cost", threshold=threshold)
        self._budget = budget_usd
        self._max = max_usd

    async def score(self, case: EvalCase, response: ModelResponse) -> MetricResult:
        cost = response.cost_usd

        if cost <= self._budget:
            score = 1.0
        elif cost >= self._max:
            score = 0.0
        else:
            score = 1.0 - (cost - self._budget) / (self._max - self._budget)

        return self._make_result(
            score=score,
            reason=f"Cost: ${cost:.4f} (budget: ${self._budget:.4f})",
            cost_usd=cost,
        )
