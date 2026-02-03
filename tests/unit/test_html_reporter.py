"""
Tests for HTML reporter.
"""

from evalkit.models import (
    CaseResult,
    EvalCase,
    EvalSuiteResult,
    MetricResult,
    ModelResponse,
    Verdict,
)
from evalkit.reporters.html_reporter import HTMLReporter


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
                ],
            ),
            CaseResult(
                case=EvalCase(input="Capital of France?", expected_output="Paris"),
                response=ModelResponse(
                    text="London",
                    model="gpt-4",
                    latency_ms=200,
                    cost_usd=0.002,
                ),
                metric_results=[
                    MetricResult("ExactMatch", 0.0, Verdict.FAIL, "Mismatch"),
                ],
            ),
        ],
    )


class TestHTMLReporter:
    def test_generate_not_empty(self):
        html = HTMLReporter().generate(make_result())
        assert len(html) > 500

    def test_contains_suite_name(self):
        html = HTMLReporter().generate(make_result())
        assert "Test Suite" in html

    def test_contains_model(self):
        html = HTMLReporter().generate(make_result())
        assert "gpt-4" in html

    def test_contains_pass_rate(self):
        html = HTMLReporter().generate(make_result())
        assert "50%" in html

    def test_contains_cases(self):
        html = HTMLReporter().generate(make_result())
        assert "2+2" in html
        assert "France" in html

    def test_contains_verdicts(self):
        html = HTMLReporter().generate(make_result())
        assert "✅" in html
        assert "❌" in html

    def test_valid_html(self):
        html = HTMLReporter().generate(make_result())
        assert html.startswith("<!DOCTYPE html>")
        assert "</html>" in html

    def test_save_to_file(self, tmp_path):
        path = tmp_path / "report.html"
        HTMLReporter().save(make_result(), path)
        assert path.exists()
        content = path.read_text()
        assert "EvalKit Report" in content

    def test_save_creates_dirs(self, tmp_path):
        path = tmp_path / "nested" / "dir" / "report.html"
        HTMLReporter().save(make_result(), path)
        assert path.exists()

    def test_empty_result(self):
        r = EvalSuiteResult(suite_name="Empty", model="test")
        html = HTMLReporter().generate(r)
        assert "Empty" in html
