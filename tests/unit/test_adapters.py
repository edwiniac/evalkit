"""
Tests for model adapters.
"""

import pytest

from evalkit.adapters import static_model, judge_from_model, _estimate_cost
from evalkit.models import ModelResponse


class TestStaticModel:
    @pytest.mark.asyncio
    async def test_known_input(self):
        model = static_model({"hello": "world"})
        r = await model("hello")
        assert isinstance(r, ModelResponse)
        assert r.text == "world"
        assert r.model == "static"

    @pytest.mark.asyncio
    async def test_unknown_input(self):
        model = static_model({"hello": "world"}, default="unknown")
        r = await model("bye")
        assert r.text == "unknown"

    @pytest.mark.asyncio
    async def test_default_response(self):
        model = static_model({})
        r = await model("anything")
        assert r.text == "I don't know"


class TestJudgeFromModel:
    @pytest.mark.asyncio
    async def test_converts_to_string(self):
        model = static_model({"prompt": "result"})
        judge = judge_from_model(model)
        result = await judge("prompt")
        assert isinstance(result, str)
        assert result == "result"


class TestPricing:
    def test_gpt4_pricing(self):
        cost = _estimate_cost("gpt-4", prompt_tokens=1000, completion_tokens=500)
        assert cost > 0

    def test_gpt4o_mini_pricing(self):
        cost = _estimate_cost("gpt-4o-mini", prompt_tokens=1000, completion_tokens=500)
        assert cost > 0
        # Should be cheaper than gpt-4 at scale
        gpt4_cost = _estimate_cost("gpt-4", 10000, 5000)
        mini_cost = _estimate_cost("gpt-4o-mini", 10000, 5000)
        assert mini_cost < gpt4_cost

    def test_claude_pricing(self):
        cost = _estimate_cost("claude-3-5-sonnet", 1000, 500)
        assert cost > 0

    def test_unknown_model_free(self):
        cost = _estimate_cost("unknown-model", 1000, 500)
        assert cost == 0.0

    def test_zero_tokens(self):
        cost = _estimate_cost("gpt-4", 0, 0)
        assert cost == 0.0
