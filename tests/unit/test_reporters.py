"""
Tests for reporters.
"""

import json
from pathlib import Path

import pytest

from evalkit.models import (
    CaseResult,
    EvalCase,
    EvalSuiteResult,
    MetricResult,
    ModelResponse,
    Verdict,
)
from evalkit.reporters.console import ConsoleReporter
from evalkit.reporters.json_reporter import JSONReporter


def make_result() -> EvalSuiteResult:
    return EvalSuiteResult(
        suite_name="Test Suite",
        model="gpt-4",
        case_results=[
            CaseResult(
                case=EvalCase(input="What is 2+2?", expected_output="4"),
                response=ModelResponse(text="4", model="gpt-4", latency_ms=150, cost_usd=0.001),
                metric_results=[
                    MetricResult("ExactMatch", 1.0, Verdict.PASS, "Match"),
                    MetricResult("ContainsAny", 1.0, Verdict.PASS, "Found [4]"),
                ],
            ),
            CaseResult(
                case=EvalCase(input="Capital of France?", expected_output="Paris"),
                response=ModelResponse(text="London", model="gpt-4", latency_ms=200, cost_usd=0.002),
                metric_results=[
                    MetricResult("ExactMatch", 0.0, Verdict.FAIL, "Mismatch"),
                    MetricResult("ContainsAny", 0.0, Verdict.FAIL, "Not found"),
                ],
            ),
        ],
    )


class TestConsoleReporter:
    def test_report_not_empty(self):
        reporter = ConsoleReporter()
        output = reporter.report(make_result())
        assert len(output) > 0

    def test_contains_suite_name(self):
        output = ConsoleReporter().report(make_result())
        assert "Test Suite" in output

    def test_contains_model_name(self):
        output = ConsoleReporter().report(make_result())
        assert "gpt-4" in output

    def test_contains_pass_rate(self):
        output = ConsoleReporter().report(make_result())
        assert "1/2" in output or "50%" in output

    def test_contains_metrics(self):
        output = ConsoleReporter().report(make_result())
        assert "ExactMatch" in output

    def test_verbose_shows_cases(self):
        output = ConsoleReporter(verbose=True).report(make_result())
        assert "2+2" in output
        assert "France" in output

    def test_non_verbose_hides_cases(self):
        output = ConsoleReporter(verbose=False).report(make_result())
        assert "Case Details" not in output

    def test_print_comparison(self):
        reporter = ConsoleReporter()
        results = {
            "gpt-4": make_result(),
            "claude": make_result(),
        }
        # Should not crash
        reporter.print_comparison(results)


class TestJSONReporter:
    def test_to_json(self):
        reporter = JSONReporter()
        output = reporter.to_json(make_result())
        data = json.loads(output)
        assert data["suite_name"] == "Test Suite"
        assert data["total_cases"] == 2

    def test_save_to_file(self, tmp_path):
        reporter = JSONReporter()
        path = tmp_path / "result.json"
        reporter.save(make_result(), path)
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["model"] == "gpt-4"

    def test_save_creates_dirs(self, tmp_path):
        reporter = JSONReporter()
        path = tmp_path / "nested" / "dir" / "result.json"
        reporter.save(make_result(), path)
        assert path.exists()

    def test_save_comparison(self, tmp_path):
        reporter = JSONReporter()
        results = {"gpt-4": make_result(), "claude": make_result()}
        path = tmp_path / "comparison.json"
        reporter.save_comparison(results, path)

        data = json.loads(path.read_text())
        assert data["comparison"] is True
        assert "gpt-4" in data["models"]
        assert "ranking" in data
