"""
End-to-end integration tests.

Tests the full pipeline: define suite → run → report.
"""

import json

import pytest

from evalkit.adapters import static_model
from evalkit.datasets.loader import load_suite
from evalkit.metrics import (
    BLEUScore,
    ContainsAny,
    ExactMatch,
    Faithfulness,
    LengthRange,
    ROUGEScore,
)
from evalkit.models import EvalCase
from evalkit.reporters import ConsoleReporter, HTMLReporter, JSONReporter
from evalkit.runners import EvalRunner
from evalkit.suite import EvalSuite

# ── E2E: Full Pipeline ───────────────────────────────────────────────


class TestFullPipeline:
    @pytest.mark.asyncio
    async def test_define_run_report(self, tmp_path):
        """Define a suite, run it, generate all report types."""
        # Define
        suite = EvalSuite(
            name="Geography Quiz",
            cases=[
                EvalCase(input="Capital of France?", expected_output="Paris"),
                EvalCase(input="Capital of Japan?", expected_output="Tokyo"),
                EvalCase(input="Capital of Germany?", expected_output="Berlin"),
            ],
            metrics=[ExactMatch(), ContainsAny()],
        )

        # Model
        model = static_model(
            {
                "Capital of France?": "Paris",
                "Capital of Japan?": "Tokyo",
                "Capital of Germany?": "Munich",  # Wrong!
            }
        )

        # Run
        runner = EvalRunner()
        result = await runner.run(suite, model, model_name="static-geo")

        assert result.total_cases == 3
        assert result.passed_cases == 2
        assert result.failed_cases == 1

        # Console report
        console_output = ConsoleReporter(verbose=True).report(result)
        assert "Geography Quiz" in console_output
        assert "static-geo" in console_output

        # JSON report
        json_path = tmp_path / "result.json"
        JSONReporter().save(result, json_path)
        data = json.loads(json_path.read_text())
        assert data["total_cases"] == 3
        assert data["passed_cases"] == 2

        # HTML report
        html_path = tmp_path / "report.html"
        HTMLReporter().save(result, html_path)
        html = html_path.read_text()
        assert "Geography Quiz" in html
        assert "<!DOCTYPE html>" in html


# ── E2E: Model Comparison ────────────────────────────────────────────


class TestModelComparison:
    @pytest.mark.asyncio
    async def test_compare_two_models(self, tmp_path):
        suite = EvalSuite(
            name="Math Quiz",
            cases=[
                EvalCase(input="2+2", expected_output="4"),
                EvalCase(input="3*3", expected_output="9"),
            ],
            metrics=[ExactMatch()],
        )

        good = static_model({"2+2": "4", "3*3": "9"})
        bad = static_model({"2+2": "5", "3*3": "10"})

        runner = EvalRunner()
        results = await runner.run_comparison(suite, models={"good": good, "bad": bad})

        assert results["good"].pass_rate == 1.0
        assert results["bad"].pass_rate == 0.0

        # Save comparison
        JSONReporter().save_comparison(results, tmp_path / "comparison.json")
        data = json.loads((tmp_path / "comparison.json").read_text())
        assert data["ranking"][0] == "good"


# ── E2E: Dataset Loading Pipeline ────────────────────────────────────


class TestDatasetPipeline:
    @pytest.mark.asyncio
    async def test_load_and_eval(self, tmp_path):
        # Create dataset file
        data = [
            {"input": "Hello", "expected_output": "Hello"},
            {"input": "World", "expected_output": "World"},
        ]
        dataset_path = tmp_path / "echo_test.json"
        dataset_path.write_text(json.dumps(data))

        # Load suite
        suite = load_suite(dataset_path, metrics=[ExactMatch()])

        # Run
        echo = static_model({"Hello": "Hello", "World": "World"})
        result = await EvalRunner().run(suite, echo, model_name="echo")

        assert result.pass_rate == 1.0
        assert result.suite_name == "echo_test"


# ── E2E: Multiple Metrics ────────────────────────────────────────────


class TestMultiMetricPipeline:
    @pytest.mark.asyncio
    async def test_many_metrics(self):
        suite = EvalSuite(
            name="Multi-Metric",
            cases=[
                EvalCase(
                    input="Explain Python",
                    expected_output="Python is a programming language",
                ),
            ],
            metrics=[
                ExactMatch(),
                ContainsAny(keywords=["Python", "programming"]),
                LengthRange(min_chars=10, max_chars=500),
                BLEUScore(),
                ROUGEScore(),
            ],
        )

        model = static_model({"Explain Python": "Python is a high-level programming language"})
        result = await EvalRunner().run(suite, model)

        # Should have results for all 5 metrics
        cr = result.case_results[0]
        assert len(cr.metric_results) == 5

        summary = result.metric_summary()
        assert "ExactMatch" in summary
        assert "ContainsAny" in summary
        assert "BLEUScore" in summary
        assert "ROUGEScore" in summary
        assert "LengthRange" in summary


# ── E2E: LLM-as-Judge Pipeline ───────────────────────────────────────


class TestJudgePipeline:
    @pytest.mark.asyncio
    async def test_judge_metric_in_suite(self):
        """Test faithfulness metric with a mock judge."""

        # Mock judge that always returns high faithfulness
        async def mock_judge(prompt: str) -> str:
            return json.dumps(
                {
                    "score": 0.95,
                    "verdict": "pass",
                    "reason": "Response is grounded in context",
                }
            )

        suite = EvalSuite(
            name="RAG Quality",
            cases=[
                EvalCase(
                    input="What is Python?",
                    expected_output="A programming language",
                    context=(
                        "Python is a high-level programming language"
                        " created by Guido van Rossum."
                    ),
                ),
            ],
            metrics=[
                ExactMatch(),
                Faithfulness(judge_fn=mock_judge),
            ],
        )

        model = static_model({"What is Python?": "Python is a programming language"})
        result = await EvalRunner().run(suite, model)

        cr = result.case_results[0]
        assert len(cr.metric_results) == 2

        faith = next(mr for mr in cr.metric_results if mr.metric_name == "Faithfulness")
        assert faith.score == 0.95


# ── E2E: Serialization Roundtrip ─────────────────────────────────────


class TestSerializationRoundtrip:
    @pytest.mark.asyncio
    async def test_json_roundtrip(self, tmp_path):
        suite = EvalSuite(
            name="Roundtrip",
            cases=[EvalCase(input="Q", expected_output="A")],
            metrics=[ExactMatch()],
        )

        model = static_model({"Q": "A"})
        result = await EvalRunner().run(suite, model)

        # Save
        path = tmp_path / "result.json"
        JSONReporter().save(result, path)

        # Load and verify
        data = json.loads(path.read_text())
        assert data["suite_name"] == "Roundtrip"
        assert data["total_cases"] == 1
        assert data["passed_cases"] == 1
        assert len(data["case_results"]) == 1
        assert data["case_results"][0]["passed"] is True
