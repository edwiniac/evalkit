"""
Tests for LLM-as-Judge metrics.

Uses mock judge functions to test the metric logic without real API calls.
"""

import json

import pytest

from evalkit.metrics.llm_judge import (
    AnswerRelevance,
    Coherence,
    Correctness,
    Faithfulness,
    Hallucination,
    LLMJudgeMetric,
    Toxicity,
)
from evalkit.models import EvalCase, ModelResponse, Verdict

# ── Mock Judge Functions ──────────────────────────────────────────────


def make_judge(score: float, reason: str = "Test", **extra):
    """Create a mock judge that returns a fixed score."""
    response = {"score": score, "verdict": "pass" if score >= 0.5 else "fail", "reason": reason}
    response.update(extra)

    async def judge(prompt: str) -> str:
        return json.dumps(response)

    return judge


def make_failing_judge():
    async def judge(prompt: str) -> str:
        raise RuntimeError("Judge crashed")

    return judge


def make_bad_json_judge():
    async def judge(prompt: str) -> str:
        return "This is not JSON at all"

    return judge


def make_markdown_judge(score: float = 0.85):
    async def judge(prompt: str) -> str:
        return f'```json\n{{"score": {score}, "reason": "wrapped in markdown"}}\n```'

    return judge


# ── Fixtures ──────────────────────────────────────────────────────────


def make_case(**kwargs) -> EvalCase:
    defaults = {
        "input": "What is Python?",
        "expected_output": "A programming language",
        "context": "Python is a high-level programming language created by Guido van Rossum.",
    }
    defaults.update(kwargs)
    return EvalCase(**defaults)


def make_response(text: str = "Python is a programming language") -> ModelResponse:
    return ModelResponse(text=text, model="test")


# ── Base LLMJudgeMetric ──────────────────────────────────────────────


class TestLLMJudgeMetric:
    @pytest.mark.asyncio
    async def test_high_score_passes(self):
        m = LLMJudgeMetric(
            judge_fn=make_judge(0.9, "Good response"),
            name="TestJudge",
            prompt_template="{input} {response}",
        )
        r = await m.score(make_case(), make_response())
        assert r.score == 0.9
        assert r.verdict == Verdict.PASS
        assert r.reason == "Good response"

    @pytest.mark.asyncio
    async def test_low_score_fails(self):
        m = LLMJudgeMetric(
            judge_fn=make_judge(0.2, "Bad response"),
            name="TestJudge",
            threshold=0.5,
            prompt_template="{input} {response}",
        )
        r = await m.score(make_case(), make_response())
        assert r.score == 0.2
        assert r.verdict == Verdict.FAIL

    @pytest.mark.asyncio
    async def test_handles_judge_error(self):
        m = LLMJudgeMetric(
            judge_fn=make_failing_judge(),
            name="TestJudge",
            prompt_template="{input} {response}",
        )
        r = await m.score(make_case(), make_response())
        assert r.verdict == Verdict.ERROR

    @pytest.mark.asyncio
    async def test_handles_bad_json(self):
        m = LLMJudgeMetric(
            judge_fn=make_bad_json_judge(),
            name="TestJudge",
            prompt_template="{input} {response}",
        )
        r = await m.score(make_case(), make_response())
        assert r.verdict == Verdict.ERROR

    @pytest.mark.asyncio
    async def test_parses_markdown_json(self):
        m = LLMJudgeMetric(
            judge_fn=make_markdown_judge(0.85),
            name="TestJudge",
            prompt_template="{input} {response}",
        )
        r = await m.score(make_case(), make_response())
        assert r.score == 0.85

    @pytest.mark.asyncio
    async def test_clamps_score(self):
        m = LLMJudgeMetric(
            judge_fn=make_judge(1.5),  # Over 1.0
            name="TestJudge",
            prompt_template="{input} {response}",
        )
        r = await m.score(make_case(), make_response())
        assert r.score == 1.0

    @pytest.mark.asyncio
    async def test_extracts_metadata(self):
        m = LLMJudgeMetric(
            judge_fn=make_judge(0.8, "OK", unsupported_claims=["claim1"]),
            name="TestJudge",
            prompt_template="{input} {response}",
        )
        r = await m.score(make_case(), make_response())
        assert "unsupported_claims" in r.metadata


# ── Faithfulness ──────────────────────────────────────────────────────


class TestFaithfulness:
    @pytest.mark.asyncio
    async def test_faithful_response(self):
        m = Faithfulness(judge_fn=make_judge(0.95, "Fully grounded in context"))
        r = await m.score(make_case(), make_response())
        assert r.score == 0.95
        assert r.verdict == Verdict.PASS

    @pytest.mark.asyncio
    async def test_unfaithful_response(self):
        m = Faithfulness(judge_fn=make_judge(0.2, "Contains unsupported claims"))
        r = await m.score(make_case(), make_response("Python was invented in 2020"))
        assert r.score == 0.2
        assert r.verdict == Verdict.FAIL

    @pytest.mark.asyncio
    async def test_no_context_scores_zero(self):
        m = Faithfulness(judge_fn=make_judge(0.9))
        case = make_case(context=None)
        r = await m.score(case, make_response())
        assert r.score == 0.0
        assert "No context" in r.reason


# ── Answer Relevance ──────────────────────────────────────────────────


class TestAnswerRelevance:
    @pytest.mark.asyncio
    async def test_relevant_response(self):
        m = AnswerRelevance(judge_fn=make_judge(0.9, "Directly addresses the question"))
        r = await m.score(make_case(), make_response())
        assert r.verdict == Verdict.PASS

    @pytest.mark.asyncio
    async def test_irrelevant_response(self):
        m = AnswerRelevance(judge_fn=make_judge(0.1, "Off topic"))
        r = await m.score(make_case(), make_response("The weather is nice"))
        assert r.verdict == Verdict.FAIL


# ── Hallucination ────────────────────────────────────────────────────


class TestHallucination:
    @pytest.mark.asyncio
    async def test_no_hallucination(self):
        m = Hallucination(judge_fn=make_judge(1.0, "No fabricated info"))
        r = await m.score(make_case(), make_response())
        assert r.score == 1.0

    @pytest.mark.asyncio
    async def test_hallucination_detected(self):
        m = Hallucination(
            judge_fn=make_judge(
                0.2, "Contains fabricated facts", hallucinations=["Python was created in 2020"]
            )
        )
        r = await m.score(make_case(), make_response("Python was created in 2020"))
        assert r.score == 0.2
        assert r.verdict == Verdict.FAIL


# ── Coherence ────────────────────────────────────────────────────────


class TestCoherence:
    @pytest.mark.asyncio
    async def test_coherent(self):
        m = Coherence(judge_fn=make_judge(0.9, "Well structured"))
        r = await m.score(make_case(), make_response())
        assert r.verdict == Verdict.PASS

    @pytest.mark.asyncio
    async def test_incoherent(self):
        m = Coherence(judge_fn=make_judge(0.2, "Rambling"))
        r = await m.score(make_case(), make_response("word salad gibberish"))
        assert r.verdict == Verdict.FAIL


# ── Toxicity ─────────────────────────────────────────────────────────


class TestToxicity:
    @pytest.mark.asyncio
    async def test_safe_content(self):
        m = Toxicity(judge_fn=make_judge(1.0, "Completely safe"))
        r = await m.score(make_case(), make_response())
        assert r.verdict == Verdict.PASS

    @pytest.mark.asyncio
    async def test_toxic_content(self):
        m = Toxicity(
            judge_fn=make_judge(0.1, "Contains harmful content", toxic_elements=["profanity"])
        )
        r = await m.score(make_case(), make_response("bad words"))
        assert r.verdict == Verdict.FAIL
        assert r.score == 0.1


# ── Correctness ──────────────────────────────────────────────────────


class TestCorrectness:
    @pytest.mark.asyncio
    async def test_correct(self):
        m = Correctness(judge_fn=make_judge(1.0, "Matches expected"))
        r = await m.score(make_case(), make_response())
        assert r.verdict == Verdict.PASS

    @pytest.mark.asyncio
    async def test_incorrect(self):
        m = Correctness(judge_fn=make_judge(0.0, "Completely wrong"))
        r = await m.score(make_case(), make_response("Python is a snake"))
        assert r.verdict == Verdict.FAIL
