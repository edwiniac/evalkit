"""
Tests for the eval runner.
"""

import pytest

from evalkit.metrics.deterministic import ContainsAny, ExactMatch
from evalkit.models import EvalCase, EvalSuiteResult, ModelResponse, Verdict
from evalkit.runners.runner import EvalRunner
from evalkit.suite import EvalSuite

# ── Helpers ───────────────────────────────────────────────────────────


async def mock_model(input: str) -> ModelResponse:
    """Simple mock model that echoes a fixed answer."""
    answers = {
        "What is 2+2?": "4",
        "Capital of France?": "Paris",
        "Capital of Japan?": "Tokyo",
        "What color is the sky?": "The sky is blue",
    }
    return ModelResponse(
        text=answers.get(input, "I don't know"),
        model="mock",
        token_count=10,
        cost_usd=0.0001,
    )


async def failing_model(input: str) -> ModelResponse:
    raise RuntimeError("Model crashed")


def make_suite(cases=None, metrics=None) -> EvalSuite:
    return EvalSuite(
        name="Test Suite",
        cases=cases
        or [
            EvalCase(input="What is 2+2?", expected_output="4"),
            EvalCase(input="Capital of France?", expected_output="Paris"),
        ],
        metrics=metrics or [ExactMatch()],
    )


# ── Basic Running ────────────────────────────────────────────────────


class TestBasicRunner:
    @pytest.mark.asyncio
    async def test_run_simple_suite(self):
        runner = EvalRunner()
        result = await runner.run(make_suite(), mock_model, model_name="mock")

        assert isinstance(result, EvalSuiteResult)
        assert result.suite_name == "Test Suite"
        assert result.model == "mock"
        assert result.total_cases == 2

    @pytest.mark.asyncio
    async def test_all_pass(self):
        runner = EvalRunner()
        result = await runner.run(make_suite(), mock_model, model_name="mock")
        assert result.passed_cases == 2
        assert result.pass_rate == 1.0

    @pytest.mark.asyncio
    async def test_some_fail(self):
        suite = make_suite(
            cases=[
                EvalCase(input="What is 2+2?", expected_output="4"),
                EvalCase(input="Unknown question", expected_output="42"),
            ],
        )
        runner = EvalRunner()
        result = await runner.run(suite, mock_model, model_name="mock")
        assert result.passed_cases == 1
        assert result.failed_cases == 1

    @pytest.mark.asyncio
    async def test_empty_suite(self):
        suite = EvalSuite(name="Empty", cases=[], metrics=[ExactMatch()])
        runner = EvalRunner()
        result = await runner.run(suite, mock_model)
        assert result.total_cases == 0


# ── Multiple Metrics ─────────────────────────────────────────────────


class TestMultipleMetrics:
    @pytest.mark.asyncio
    async def test_two_metrics(self):
        suite = make_suite(
            cases=[EvalCase(input="Capital of France?", expected_output="Paris")],
            metrics=[ExactMatch(), ContainsAny(keywords=["Paris"])],
        )
        runner = EvalRunner()
        result = await runner.run(suite, mock_model)

        assert result.total_cases == 1
        cr = result.case_results[0]
        assert len(cr.metric_results) == 2
        assert all(mr.verdict == Verdict.PASS for mr in cr.metric_results)

    @pytest.mark.asyncio
    async def test_metric_summary(self):
        suite = make_suite(
            metrics=[ExactMatch(), ContainsAny(keywords=["Paris", "4"])],
        )
        runner = EvalRunner()
        result = await runner.run(suite, mock_model)

        summary = result.metric_summary()
        assert "ExactMatch" in summary
        assert "ContainsAny" in summary


# ── Error Handling ───────────────────────────────────────────────────


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_model_error_handled(self):
        runner = EvalRunner()
        result = await runner.run(make_suite(), failing_model, model_name="broken")

        # Should not crash — produces results with empty responses
        assert result.total_cases == 2

    @pytest.mark.asyncio
    async def test_metric_error_handled(self):
        class BrokenMetric(ExactMatch):
            async def score(self, case, response):
                raise ValueError("Metric broke")

        suite = make_suite(metrics=[BrokenMetric()])
        runner = EvalRunner()
        result = await runner.run(suite, mock_model)

        # Should not crash — produces error verdicts
        assert result.total_cases == 2
        for cr in result.case_results:
            assert any(mr.verdict == Verdict.ERROR for mr in cr.metric_results)


# ── Concurrent Execution ─────────────────────────────────────────────


class TestConcurrency:
    @pytest.mark.asyncio
    async def test_concurrent_run(self):
        runner = EvalRunner(concurrency=3)
        suite = make_suite(
            cases=[
                EvalCase(input="What is 2+2?", expected_output="4"),
                EvalCase(input="Capital of France?", expected_output="Paris"),
                EvalCase(input="Capital of Japan?", expected_output="Tokyo"),
            ],
        )
        result = await runner.run(suite, mock_model)
        assert result.total_cases == 3
        assert result.passed_cases == 3


# ── Model Comparison ─────────────────────────────────────────────────


class TestComparison:
    @pytest.mark.asyncio
    async def test_compare_models(self):
        async def good_model(input: str) -> ModelResponse:
            return ModelResponse(text="4" if "2+2" in input else "Paris", model="good")

        async def bad_model(input: str) -> ModelResponse:
            return ModelResponse(text="I don't know", model="bad")

        runner = EvalRunner()
        results = await runner.run_comparison(
            make_suite(),
            models={"good": good_model, "bad": bad_model},
        )

        assert "good" in results
        assert "bad" in results
        assert results["good"].pass_rate > results["bad"].pass_rate


# ── Callbacks ────────────────────────────────────────────────────────


class TestCallbacks:
    @pytest.mark.asyncio
    async def test_on_case_complete(self):
        called = []

        def on_complete(idx, result):
            called.append(idx)

        runner = EvalRunner(on_case_complete=on_complete)
        await runner.run(make_suite(), mock_model)
        assert len(called) == 2
        assert called == [0, 1]


# ── Metadata ─────────────────────────────────────────────────────────


class TestRunMetadata:
    @pytest.mark.asyncio
    async def test_has_timestamps(self):
        runner = EvalRunner()
        result = await runner.run(make_suite(), mock_model)
        assert result.started_at is not None
        assert result.finished_at is not None

    @pytest.mark.asyncio
    async def test_has_run_id(self):
        runner = EvalRunner()
        result = await runner.run(make_suite(), mock_model)
        assert result.run_id  # Non-empty

    @pytest.mark.asyncio
    async def test_latency_tracked(self):
        runner = EvalRunner()
        result = await runner.run(make_suite(), mock_model)
        for cr in result.case_results:
            assert cr.response.latency_ms >= 0
