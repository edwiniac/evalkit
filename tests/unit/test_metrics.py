"""
Tests for deterministic metrics.
"""

import pytest

from evalkit.metrics.deterministic import (
    ContainsAll,
    ContainsAny,
    ExactMatch,
    IsJSON,
    LengthRange,
    RegexMatch,
)
from evalkit.models import EvalCase, ModelResponse, Verdict


def make_case(expected: str = "Paris", input: str = "What is the capital of France?") -> EvalCase:
    return EvalCase(input=input, expected_output=expected)


def make_response(text: str) -> ModelResponse:
    return ModelResponse(text=text, model="test")


# ── ExactMatch ───────────────────────────────────────────────────────


class TestExactMatch:
    @pytest.mark.asyncio
    async def test_exact_match(self):
        m = ExactMatch()
        r = await m.score(make_case("Paris"), make_response("Paris"))
        assert r.score == 1.0
        assert r.verdict == Verdict.PASS

    @pytest.mark.asyncio
    async def test_case_insensitive(self):
        m = ExactMatch(case_sensitive=False)
        r = await m.score(make_case("Paris"), make_response("paris"))
        assert r.score == 1.0

    @pytest.mark.asyncio
    async def test_case_sensitive(self):
        m = ExactMatch(case_sensitive=True)
        r = await m.score(make_case("Paris"), make_response("paris"))
        assert r.score == 0.0
        assert r.verdict == Verdict.FAIL

    @pytest.mark.asyncio
    async def test_strips_whitespace(self):
        m = ExactMatch()
        r = await m.score(make_case("Paris"), make_response("  Paris  "))
        assert r.score == 1.0

    @pytest.mark.asyncio
    async def test_no_strip(self):
        m = ExactMatch(strip=False)
        r = await m.score(make_case("Paris"), make_response("  Paris  "))
        assert r.score == 0.0

    @pytest.mark.asyncio
    async def test_no_expected_output(self):
        case = EvalCase(input="Q")
        r = await ExactMatch().score(case, make_response("A"))
        assert r.verdict == Verdict.ERROR

    @pytest.mark.asyncio
    async def test_mismatch(self):
        r = await ExactMatch().score(make_case("Paris"), make_response("London"))
        assert r.score == 0.0
        assert "Expected" in r.reason


# ── ContainsAny ──────────────────────────────────────────────────────


class TestContainsAny:
    @pytest.mark.asyncio
    async def test_contains_keyword(self):
        m = ContainsAny(keywords=["Paris", "France"])
        r = await m.score(make_case(), make_response("The capital is Paris"))
        assert r.score >= 0.5

    @pytest.mark.asyncio
    async def test_contains_all_keywords(self):
        m = ContainsAny(keywords=["Paris", "capital"])
        r = await m.score(make_case(), make_response("Paris is the capital"))
        assert r.score == 1.0

    @pytest.mark.asyncio
    async def test_contains_none(self):
        m = ContainsAny(keywords=["Tokyo", "Japan"])
        r = await m.score(make_case(), make_response("Paris is lovely"))
        assert r.score == 0.0
        assert r.verdict == Verdict.FAIL

    @pytest.mark.asyncio
    async def test_case_insensitive(self):
        m = ContainsAny(keywords=["PARIS"])
        r = await m.score(make_case(), make_response("paris is great"))
        assert r.score == 1.0

    @pytest.mark.asyncio
    async def test_falls_back_to_expected(self):
        m = ContainsAny()
        r = await m.score(make_case("Paris"), make_response("Paris"))
        assert r.score == 1.0

    @pytest.mark.asyncio
    async def test_no_keywords_no_expected(self):
        m = ContainsAny()
        case = EvalCase(input="Q")
        r = await m.score(case, make_response("A"))
        assert r.verdict == Verdict.ERROR


# ── ContainsAll ──────────────────────────────────────────────────────


class TestContainsAll:
    @pytest.mark.asyncio
    async def test_all_present(self):
        m = ContainsAll(keywords=["Python", "Java", "Rust"])
        r = await m.score(make_case(), make_response("I know Python, Java, and Rust"))
        assert r.score == 1.0
        assert r.verdict == Verdict.PASS

    @pytest.mark.asyncio
    async def test_some_missing(self):
        m = ContainsAll(keywords=["Python", "Java", "Go"])
        r = await m.score(make_case(), make_response("I know Python and Java"))
        assert r.score == pytest.approx(2 / 3)

    @pytest.mark.asyncio
    async def test_none_present(self):
        m = ContainsAll(keywords=["Rust", "Go"])
        r = await m.score(make_case(), make_response("I know Python"))
        assert r.score == 0.0

    @pytest.mark.asyncio
    async def test_no_keywords(self):
        m = ContainsAll()
        r = await m.score(make_case(), make_response("anything"))
        assert r.verdict == Verdict.ERROR


# ── RegexMatch ───────────────────────────────────────────────────────


class TestRegexMatch:
    @pytest.mark.asyncio
    async def test_matches_pattern(self):
        m = RegexMatch(pattern=r"\d{3}-\d{4}")
        r = await m.score(make_case(), make_response("Call 555-1234"))
        assert r.score == 1.0

    @pytest.mark.asyncio
    async def test_no_match(self):
        m = RegexMatch(pattern=r"\d{3}-\d{4}")
        r = await m.score(make_case(), make_response("No phone here"))
        assert r.score == 0.0

    @pytest.mark.asyncio
    async def test_case_insensitive(self):
        m = RegexMatch(pattern=r"paris")
        r = await m.score(make_case(), make_response("PARIS is great"))
        assert r.score == 1.0

    @pytest.mark.asyncio
    async def test_falls_back_to_expected(self):
        m = RegexMatch()
        r = await m.score(make_case("Paris"), make_response("Paris"))
        assert r.score == 1.0

    @pytest.mark.asyncio
    async def test_invalid_regex(self):
        m = RegexMatch(pattern="[invalid")
        r = await m.score(make_case(), make_response("test"))
        assert r.verdict == Verdict.ERROR

    @pytest.mark.asyncio
    async def test_no_pattern_no_expected(self):
        m = RegexMatch()
        case = EvalCase(input="Q")
        r = await m.score(case, make_response("A"))
        assert r.verdict == Verdict.ERROR


# ── IsJSON ───────────────────────────────────────────────────────────


class TestIsJSON:
    @pytest.mark.asyncio
    async def test_valid_json(self):
        m = IsJSON()
        r = await m.score(make_case(), make_response('{"key": "value"}'))
        assert r.score == 1.0

    @pytest.mark.asyncio
    async def test_invalid_json(self):
        m = IsJSON()
        r = await m.score(make_case(), make_response("not json at all"))
        assert r.score == 0.0

    @pytest.mark.asyncio
    async def test_json_in_markdown(self):
        m = IsJSON()
        text = '```json\n{"key": "value"}\n```'
        r = await m.score(make_case(), make_response(text))
        assert r.score == 1.0

    @pytest.mark.asyncio
    async def test_required_keys_present(self):
        m = IsJSON(required_keys=["name", "age"])
        r = await m.score(make_case(), make_response('{"name": "Ed", "age": 25}'))
        assert r.score == 1.0

    @pytest.mark.asyncio
    async def test_required_keys_missing(self):
        m = IsJSON(required_keys=["name", "age", "email"])
        r = await m.score(make_case(), make_response('{"name": "Ed"}'))
        assert r.score == pytest.approx(1 / 3)

    @pytest.mark.asyncio
    async def test_json_array(self):
        m = IsJSON(required_keys=["name"])
        r = await m.score(make_case(), make_response("[1, 2, 3]"))
        assert r.score == 0.5  # Valid JSON but not an object


# ── LengthRange ──────────────────────────────────────────────────────


class TestLengthRange:
    @pytest.mark.asyncio
    async def test_within_range(self):
        m = LengthRange(min_chars=5, max_chars=100)
        r = await m.score(make_case(), make_response("This is a good response"))
        assert r.score == 1.0

    @pytest.mark.asyncio
    async def test_too_short(self):
        m = LengthRange(min_chars=50, max_chars=200)
        r = await m.score(make_case(), make_response("Hi"))
        assert r.score < 1.0
        assert "short" in r.reason.lower()

    @pytest.mark.asyncio
    async def test_too_long(self):
        m = LengthRange(min_chars=0, max_chars=10)
        r = await m.score(make_case(), make_response("A" * 100))
        assert r.score < 1.0
        assert "long" in r.reason.lower()

    @pytest.mark.asyncio
    async def test_exact_min(self):
        m = LengthRange(min_chars=5, max_chars=5)
        r = await m.score(make_case(), make_response("Hello"))
        assert r.score == 1.0

    @pytest.mark.asyncio
    async def test_empty_response(self):
        m = LengthRange(min_chars=10, max_chars=100)
        r = await m.score(make_case(), make_response(""))
        assert r.score == 0.0
