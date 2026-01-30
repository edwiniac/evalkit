"""
Tests for core data models.
"""

import pytest

from evalkit.models import (
    CaseResult,
    EvalCase,
    EvalSuiteResult,
    MetricResult,
    ModelResponse,
    Verdict,
)


class TestEvalCase:
    def test_basic_case(self):
        case = EvalCase(input="What is 2+2?", expected_output="4")
        assert case.input == "What is 2+2?"
        assert case.expected_output == "4"
        assert case.case_id  # Auto-generated

    def test_context_string(self):
        case = EvalCase(input="Q", context="Some context")
        assert case.context_str == "Some context"

    def test_context_list(self):
        case = EvalCase(input="Q", context=["chunk1", "chunk2"])
        assert "chunk1" in case.context_str
        assert "chunk2" in case.context_str

    def test_context_none(self):
        case = EvalCase(input="Q")
        assert case.context_str == ""

    def test_metadata(self):
        case = EvalCase(input="Q", metadata={"category": "math"})
        assert case.metadata["category"] == "math"

    def test_to_dict(self):
        case = EvalCase(input="Q", expected_output="A", case_id="test123")
        d = case.to_dict()
        assert d["input"] == "Q"
        assert d["expected_output"] == "A"
        assert d["case_id"] == "test123"


class TestModelResponse:
    def test_basic_response(self):
        r = ModelResponse(text="Hello", model="gpt-4")
        assert r.text == "Hello"
        assert r.model == "gpt-4"

    def test_to_dict(self):
        r = ModelResponse(text="Hi", model="gpt-4", latency_ms=150.5, cost_usd=0.001)
        d = r.to_dict()
        assert d["text"] == "Hi"
        assert d["latency_ms"] == 150.5
        assert d["cost_usd"] == 0.001


class TestMetricResult:
    def test_pass_verdict(self):
        r = MetricResult(metric_name="test", score=0.9, verdict=Verdict.PASS)
        assert r.verdict == Verdict.PASS

    def test_fail_verdict(self):
        r = MetricResult(metric_name="test", score=0.2, verdict=Verdict.FAIL)
        assert r.verdict == Verdict.FAIL

    def test_to_dict(self):
        r = MetricResult(metric_name="ExactMatch", score=1.0, verdict=Verdict.PASS, reason="Match")
        d = r.to_dict()
        assert d["metric_name"] == "ExactMatch"
        assert d["score"] == 1.0
        assert d["verdict"] == "pass"


class TestCaseResult:
    def test_all_pass(self):
        cr = CaseResult(
            case=EvalCase(input="Q"),
            response=ModelResponse(text="A", model="test"),
            metric_results=[
                MetricResult("m1", 0.9, Verdict.PASS),
                MetricResult("m2", 0.8, Verdict.PASS),
            ],
        )
        assert cr.passed is True
        assert cr.avg_score == pytest.approx(0.85)

    def test_one_fail(self):
        cr = CaseResult(
            case=EvalCase(input="Q"),
            response=ModelResponse(text="A", model="test"),
            metric_results=[
                MetricResult("m1", 0.9, Verdict.PASS),
                MetricResult("m2", 0.2, Verdict.FAIL),
            ],
        )
        assert cr.passed is False

    def test_skip_ignored(self):
        cr = CaseResult(
            case=EvalCase(input="Q"),
            response=ModelResponse(text="A", model="test"),
            metric_results=[
                MetricResult("m1", 0.9, Verdict.PASS),
                MetricResult("m2", 0.0, Verdict.SKIP),
            ],
        )
        assert cr.passed is True
        assert cr.avg_score == pytest.approx(0.9)

    def test_empty_results(self):
        cr = CaseResult(
            case=EvalCase(input="Q"),
            response=ModelResponse(text="A", model="test"),
        )
        assert cr.passed is True
        assert cr.avg_score == 0.0


class TestEvalSuiteResult:
    def _make_result(self, scores: list[tuple[float, Verdict]]) -> EvalSuiteResult:
        case_results = []
        for score, verdict in scores:
            case_results.append(CaseResult(
                case=EvalCase(input="Q"),
                response=ModelResponse(text="A", model="test", latency_ms=100, cost_usd=0.001),
                metric_results=[MetricResult("m1", score, verdict)],
            ))
        return EvalSuiteResult(
            suite_name="Test",
            model="test",
            case_results=case_results,
        )

    def test_all_pass(self):
        r = self._make_result([(0.9, Verdict.PASS), (0.8, Verdict.PASS)])
        assert r.total_cases == 2
        assert r.passed_cases == 2
        assert r.failed_cases == 0
        assert r.pass_rate == 1.0

    def test_mixed(self):
        r = self._make_result([(0.9, Verdict.PASS), (0.2, Verdict.FAIL)])
        assert r.passed_cases == 1
        assert r.failed_cases == 1
        assert r.pass_rate == 0.5

    def test_avg_score(self):
        r = self._make_result([(0.8, Verdict.PASS), (0.6, Verdict.PASS)])
        assert r.avg_score == pytest.approx(0.7)

    def test_avg_latency(self):
        r = self._make_result([(0.9, Verdict.PASS)])
        assert r.avg_latency_ms == 100.0

    def test_total_cost(self):
        r = self._make_result([(0.9, Verdict.PASS), (0.8, Verdict.PASS)])
        assert r.total_cost_usd == pytest.approx(0.002)

    def test_metric_summary(self):
        r = self._make_result([(0.8, Verdict.PASS), (0.6, Verdict.PASS)])
        summary = r.metric_summary()
        assert "m1" in summary
        assert summary["m1"]["avg"] == pytest.approx(0.7)
        assert summary["m1"]["min"] == pytest.approx(0.6)
        assert summary["m1"]["max"] == pytest.approx(0.8)

    def test_to_dict(self):
        r = self._make_result([(0.9, Verdict.PASS)])
        d = r.to_dict()
        assert d["suite_name"] == "Test"
        assert d["total_cases"] == 1
        assert d["pass_rate"] == 1.0
        assert "case_results" in d

    def test_empty(self):
        r = EvalSuiteResult(suite_name="Empty", model="test")
        assert r.total_cases == 0
        assert r.pass_rate == 0.0
        assert r.avg_score == 0.0
