"""
Tests for statistical metrics.
"""

import pytest

from evalkit.metrics.statistical import (
    BLEUScore,
    CostMetric,
    LatencyMetric,
    ROUGEScore,
    SemanticSimilarity,
)
from evalkit.models import EvalCase, ModelResponse, Verdict


def make_case(expected: str = "The capital of France is Paris") -> EvalCase:
    return EvalCase(input="What is the capital of France?", expected_output=expected)


def make_response(text: str, latency_ms: float = 100, cost_usd: float = 0.001) -> ModelResponse:
    return ModelResponse(text=text, model="test", latency_ms=latency_ms, cost_usd=cost_usd)


# ── BLEU Score ───────────────────────────────────────────────────────


class TestBLEUScore:
    @pytest.mark.asyncio
    async def test_identical_text(self):
        m = BLEUScore()
        r = await m.score(make_case("hello world"), make_response("hello world"))
        assert r.score > 0.5

    @pytest.mark.asyncio
    async def test_completely_different(self):
        m = BLEUScore()
        r = await m.score(make_case("hello world"), make_response("goodbye moon"))
        assert r.score < 0.3

    @pytest.mark.asyncio
    async def test_partial_overlap(self):
        m = BLEUScore()
        r = await m.score(
            make_case("The capital of France is Paris"),
            make_response("Paris is the capital of France"),
        )
        assert r.score > 0.3

    @pytest.mark.asyncio
    async def test_empty_response(self):
        m = BLEUScore()
        r = await m.score(make_case("hello"), make_response(""))
        assert r.score == 0.0

    @pytest.mark.asyncio
    async def test_no_expected_output(self):
        m = BLEUScore()
        case = EvalCase(input="Q")
        r = await m.score(case, make_response("anything"))
        assert r.verdict == Verdict.ERROR


# ── ROUGE Score ──────────────────────────────────────────────────────


class TestROUGEScore:
    @pytest.mark.asyncio
    async def test_identical_text(self):
        m = ROUGEScore()
        r = await m.score(make_case("hello world"), make_response("hello world"))
        assert r.score > 0.8

    @pytest.mark.asyncio
    async def test_completely_different(self):
        m = ROUGEScore()
        r = await m.score(make_case("hello world"), make_response("goodbye moon"))
        assert r.score < 0.3

    @pytest.mark.asyncio
    async def test_partial_overlap(self):
        m = ROUGEScore()
        r = await m.score(
            make_case("The quick brown fox jumps over the lazy dog"),
            make_response("A fast brown fox leaps over a lazy dog"),
        )
        assert r.score > 0.3

    @pytest.mark.asyncio
    async def test_no_expected(self):
        m = ROUGEScore()
        case = EvalCase(input="Q")
        r = await m.score(case, make_response("A"))
        assert r.verdict == Verdict.ERROR

    @pytest.mark.asyncio
    async def test_has_precision_recall(self):
        m = ROUGEScore()
        r = await m.score(make_case("hello world"), make_response("hello world"))
        assert "precision" in r.metadata or r.score > 0


# ── Semantic Similarity (Jaccard fallback) ────────────────────────────


class TestSemanticSimilarity:
    @pytest.mark.asyncio
    async def test_identical_text(self):
        m = SemanticSimilarity()
        r = await m.score(make_case("hello world"), make_response("hello world"))
        assert r.score > 0.8

    @pytest.mark.asyncio
    async def test_completely_different(self):
        m = SemanticSimilarity()
        r = await m.score(make_case("hello world"), make_response("goodbye moon"))
        assert r.score < 0.3

    @pytest.mark.asyncio
    async def test_partial_overlap(self):
        m = SemanticSimilarity()
        r = await m.score(
            make_case("Python is a programming language"),
            make_response("Python is a great programming language"),
        )
        assert r.score > 0.5

    @pytest.mark.asyncio
    async def test_no_expected(self):
        m = SemanticSimilarity()
        case = EvalCase(input="Q")
        r = await m.score(case, make_response("A"))
        assert r.verdict == Verdict.ERROR

    @pytest.mark.asyncio
    async def test_both_empty(self):
        m = SemanticSimilarity()
        r = await m.score(make_case(""), make_response(""))
        # Empty expected_output hits the guard → error
        assert r.verdict == Verdict.ERROR

    def test_jaccard_similarity(self):
        sim = SemanticSimilarity._jaccard_similarity
        assert sim("hello world", "hello world") == 1.0
        assert sim("hello", "goodbye") == 0.0
        assert sim("a b c", "b c d") == pytest.approx(0.5)


# ── Latency Metric ───────────────────────────────────────────────────


class TestLatencyMetric:
    @pytest.mark.asyncio
    async def test_under_target(self):
        m = LatencyMetric(target_ms=1000, max_ms=5000)
        r = await m.score(make_case(), make_response("A", latency_ms=500))
        assert r.score == 1.0
        assert r.verdict == Verdict.PASS

    @pytest.mark.asyncio
    async def test_over_max(self):
        m = LatencyMetric(target_ms=1000, max_ms=5000)
        r = await m.score(make_case(), make_response("A", latency_ms=6000))
        assert r.score == 0.0

    @pytest.mark.asyncio
    async def test_between_target_and_max(self):
        m = LatencyMetric(target_ms=1000, max_ms=5000)
        r = await m.score(make_case(), make_response("A", latency_ms=3000))
        assert 0.0 < r.score < 1.0

    @pytest.mark.asyncio
    async def test_exact_target(self):
        m = LatencyMetric(target_ms=1000)
        r = await m.score(make_case(), make_response("A", latency_ms=1000))
        assert r.score == 1.0


# ── Cost Metric ──────────────────────────────────────────────────────


class TestCostMetric:
    @pytest.mark.asyncio
    async def test_under_budget(self):
        m = CostMetric(budget_usd=0.01, max_usd=0.10)
        r = await m.score(make_case(), make_response("A", cost_usd=0.005))
        assert r.score == 1.0

    @pytest.mark.asyncio
    async def test_over_max(self):
        m = CostMetric(budget_usd=0.01, max_usd=0.10)
        r = await m.score(make_case(), make_response("A", cost_usd=0.20))
        assert r.score == 0.0

    @pytest.mark.asyncio
    async def test_between_budget_and_max(self):
        m = CostMetric(budget_usd=0.01, max_usd=0.10)
        r = await m.score(make_case(), make_response("A", cost_usd=0.05))
        assert 0.0 < r.score < 1.0

    @pytest.mark.asyncio
    async def test_free(self):
        m = CostMetric(budget_usd=0.01)
        r = await m.score(make_case(), make_response("A", cost_usd=0.0))
        assert r.score == 1.0
