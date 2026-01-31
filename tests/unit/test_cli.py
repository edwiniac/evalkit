"""
Tests for the CLI.
"""

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from evalkit.cli import main


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def dataset_json(tmp_path):
    data = [
        {"input": "What is 2+2?", "expected_output": "4"},
        {"input": "Capital of France?", "expected_output": "Paris"},
    ]
    path = tmp_path / "test_data.json"
    path.write_text(json.dumps(data))
    return path


@pytest.fixture
def dataset_csv(tmp_path):
    csv = "input,expected_output\nWhat is 2+2?,4\nCapital of France?,Paris\n"
    path = tmp_path / "test_data.csv"
    path.write_text(csv)
    return path


class TestCLIRun:
    def test_run_basic(self, runner, dataset_json):
        result = runner.invoke(main, ["run", str(dataset_json)])
        assert result.exit_code == 0
        assert "EvalKit Report" in result.output

    def test_run_with_metrics(self, runner, dataset_json):
        result = runner.invoke(main, ["run", str(dataset_json), "-m", "exact,contains"])
        assert result.exit_code == 0

    def test_run_csv(self, runner, dataset_csv):
        result = runner.invoke(main, ["run", str(dataset_csv)])
        assert result.exit_code == 0

    def test_run_verbose(self, runner, dataset_json):
        result = runner.invoke(main, ["run", str(dataset_json), "-v"])
        assert result.exit_code == 0
        assert "Case" in result.output

    def test_run_json_output(self, runner, dataset_json, tmp_path):
        output = tmp_path / "result.json"
        result = runner.invoke(main, [
            "run", str(dataset_json), "-r", "json", "-o", str(output)
        ])
        assert result.exit_code == 0
        assert output.exists()
        data = json.loads(output.read_text())
        assert "suite_name" in data

    def test_run_html_output(self, runner, dataset_json, tmp_path):
        output = tmp_path / "report.html"
        result = runner.invoke(main, [
            "run", str(dataset_json), "-r", "html", "-o", str(output)
        ])
        assert result.exit_code == 0
        assert output.exists()

    def test_run_custom_name(self, runner, dataset_json):
        result = runner.invoke(main, ["run", str(dataset_json), "-n", "My Test"])
        assert result.exit_code == 0
        assert "My Test" in result.output

    def test_run_unknown_metric(self, runner, dataset_json):
        result = runner.invoke(main, ["run", str(dataset_json), "-m", "nonexistent"])
        assert result.exit_code != 0

    def test_run_nonexistent_file(self, runner):
        result = runner.invoke(main, ["run", "/nonexistent/file.json"])
        assert result.exit_code != 0


class TestCLIListMetrics:
    def test_list_metrics(self, runner):
        result = runner.invoke(main, ["list-metrics"])
        assert result.exit_code == 0
        assert "exact" in result.output
        assert "bleu" in result.output

    def test_lists_judge_metrics(self, runner):
        result = runner.invoke(main, ["list-metrics"])
        assert "faithfulness" in result.output.lower()


class TestCLIVersion:
    def test_version(self, runner):
        result = runner.invoke(main, ["--version"])
        assert "0.1.0" in result.output
