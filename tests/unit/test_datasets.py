"""
Tests for dataset loading.
"""

import json

import pytest

from evalkit.datasets.loader import (
    load_cases,
    load_cases_csv,
    load_cases_json,
    load_cases_jsonl,
    load_suite,
)
from evalkit.metrics.deterministic import ExactMatch
from evalkit.models import EvalCase


# ── JSON Loading ─────────────────────────────────────────────────────

class TestLoadJSON:
    def test_load_json_list(self, tmp_path):
        data = [
            {"input": "Q1", "expected_output": "A1"},
            {"input": "Q2", "expected_output": "A2"},
        ]
        path = tmp_path / "data.json"
        path.write_text(json.dumps(data))

        cases = load_cases_json(path)
        assert len(cases) == 2
        assert cases[0].input == "Q1"
        assert cases[1].expected_output == "A2"

    def test_load_json_single(self, tmp_path):
        data = {"input": "Q1", "expected_output": "A1"}
        path = tmp_path / "data.json"
        path.write_text(json.dumps(data))

        cases = load_cases_json(path)
        assert len(cases) == 1

    def test_load_with_context(self, tmp_path):
        data = [{"input": "Q", "expected_output": "A", "context": "Some context"}]
        path = tmp_path / "data.json"
        path.write_text(json.dumps(data))

        cases = load_cases_json(path)
        assert cases[0].context == "Some context"

    def test_load_with_metadata(self, tmp_path):
        data = [{"input": "Q", "metadata": {"category": "math"}}]
        path = tmp_path / "data.json"
        path.write_text(json.dumps(data))

        cases = load_cases_json(path)
        assert cases[0].metadata["category"] == "math"

    def test_alternative_keys(self, tmp_path):
        data = [{"question": "Q", "answer": "A"}]
        path = tmp_path / "data.json"
        path.write_text(json.dumps(data))

        cases = load_cases_json(path)
        assert cases[0].input == "Q"
        assert cases[0].expected_output == "A"


# ── JSONL Loading ────────────────────────────────────────────────────

class TestLoadJSONL:
    def test_load_jsonl(self, tmp_path):
        lines = [
            json.dumps({"input": "Q1", "expected_output": "A1"}),
            json.dumps({"input": "Q2", "expected_output": "A2"}),
        ]
        path = tmp_path / "data.jsonl"
        path.write_text("\n".join(lines))

        cases = load_cases_jsonl(path)
        assert len(cases) == 2

    def test_handles_empty_lines(self, tmp_path):
        lines = [
            json.dumps({"input": "Q1"}),
            "",
            json.dumps({"input": "Q2"}),
        ]
        path = tmp_path / "data.jsonl"
        path.write_text("\n".join(lines))

        cases = load_cases_jsonl(path)
        assert len(cases) == 2


# ── CSV Loading ──────────────────────────────────────────────────────

class TestLoadCSV:
    def test_load_csv(self, tmp_path):
        csv_content = "input,expected_output\nQ1,A1\nQ2,A2\n"
        path = tmp_path / "data.csv"
        path.write_text(csv_content)

        cases = load_cases_csv(path)
        assert len(cases) == 2
        assert cases[0].input == "Q1"
        assert cases[0].expected_output == "A1"

    def test_csv_with_context(self, tmp_path):
        csv_content = "input,expected_output,context\nQ1,A1,Some context\n"
        path = tmp_path / "data.csv"
        path.write_text(csv_content)

        cases = load_cases_csv(path)
        assert cases[0].context == "Some context"

    def test_csv_alternative_column(self, tmp_path):
        csv_content = "input,expected\nQ1,A1\n"
        path = tmp_path / "data.csv"
        path.write_text(csv_content)

        cases = load_cases_csv(path)
        assert cases[0].expected_output == "A1"


# ── Auto-detect Format ───────────────────────────────────────────────

class TestLoadCases:
    def test_auto_json(self, tmp_path):
        path = tmp_path / "data.json"
        path.write_text(json.dumps([{"input": "Q"}]))
        cases = load_cases(path)
        assert len(cases) == 1

    def test_auto_jsonl(self, tmp_path):
        path = tmp_path / "data.jsonl"
        path.write_text(json.dumps({"input": "Q"}))
        cases = load_cases(path)
        assert len(cases) == 1

    def test_auto_csv(self, tmp_path):
        path = tmp_path / "data.csv"
        path.write_text("input,expected_output\nQ,A\n")
        cases = load_cases(path)
        assert len(cases) == 1

    def test_unsupported_format(self, tmp_path):
        path = tmp_path / "data.xml"
        path.write_text("<data/>")
        with pytest.raises(ValueError, match="Unsupported"):
            load_cases(path)


# ── Suite Loading ────────────────────────────────────────────────────

class TestLoadSuite:
    def test_load_suite(self, tmp_path):
        path = tmp_path / "my_suite.json"
        path.write_text(json.dumps([
            {"input": "Q1", "expected_output": "A1"},
            {"input": "Q2", "expected_output": "A2"},
        ]))
        suite = load_suite(path)
        assert suite.name == "my_suite"
        assert len(suite) == 2

    def test_load_suite_custom_name(self, tmp_path):
        path = tmp_path / "data.json"
        path.write_text(json.dumps([{"input": "Q"}]))
        suite = load_suite(path, name="Custom Name")
        assert suite.name == "Custom Name"

    def test_load_suite_with_metrics(self, tmp_path):
        path = tmp_path / "data.json"
        path.write_text(json.dumps([{"input": "Q"}]))
        suite = load_suite(path, metrics=[ExactMatch()])
        assert len(suite.metrics) == 1
