"""
Tests for EvalSuite.
"""

from evalkit.metrics.deterministic import ExactMatch
from evalkit.models import EvalCase
from evalkit.suite import EvalSuite


class TestEvalSuite:
    def test_create_suite(self):
        suite = EvalSuite(name="Test")
        assert suite.name == "Test"
        assert len(suite) == 0

    def test_add_case(self):
        suite = EvalSuite(name="Test")
        suite.add_case(EvalCase(input="Q"))
        assert len(suite) == 1

    def test_add_cases(self):
        suite = EvalSuite(name="Test")
        suite.add_cases([EvalCase(input="Q1"), EvalCase(input="Q2")])
        assert len(suite) == 2

    def test_add_metric(self):
        suite = EvalSuite(name="Test")
        suite.add_metric(ExactMatch())
        assert len(suite.metrics) == 1

    def test_add_metrics(self):
        suite = EvalSuite(name="Test")
        suite.add_metrics([ExactMatch(), ExactMatch()])
        assert len(suite.metrics) == 2

    def test_fluent_api(self):
        suite = (
            EvalSuite(name="Fluent")
            .add_case(EvalCase(input="Q1"))
            .add_case(EvalCase(input="Q2"))
            .add_metric(ExactMatch())
        )
        assert len(suite) == 2
        assert len(suite.metrics) == 1

    def test_repr(self):
        suite = EvalSuite(name="Test", cases=[EvalCase(input="Q")], metrics=[ExactMatch()])
        assert "Test" in repr(suite)
        assert "1" in repr(suite)
