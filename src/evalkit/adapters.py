"""
Model adapters — create callable functions for different LLM providers.

Each adapter returns an async function compatible with:
- EvalRunner's ModelFn (input → ModelResponse)
- LLMJudgeMetric's JudgeFn (prompt → str)
"""

import logging
import time
from typing import Optional

from .models import ModelResponse

logger = logging.getLogger(__name__)

# Token pricing (approximate, per 1K tokens)
PRICING = {
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    "claude-3-5-sonnet": {"input": 0.003, "output": 0.015},
    "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
    "claude-3-opus": {"input": 0.015, "output": 0.075},
}


def _estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Estimate cost in USD based on token counts."""
    model_lower = model.lower()
    # Sort by key length descending so "gpt-4o-mini" matches before "gpt-4"
    for key in sorted(PRICING.keys(), key=len, reverse=True):
        if key in model_lower:
            prices = PRICING[key]
            return (
                prompt_tokens / 1000 * prices["input"] + completion_tokens / 1000 * prices["output"]
            )
    return 0.0


def openai_model(
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
):
    """
    Create an OpenAI model function.

    Returns an async function: input → ModelResponse.

    Usage:
        model_fn = openai_model("gpt-4o-mini")
        response = await model_fn("What is 2+2?")
    """

    async def call(input_text: str) -> ModelResponse:
        import openai

        client = openai.AsyncOpenAI(api_key=api_key)
        start = time.monotonic()

        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": input_text}],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        latency = (time.monotonic() - start) * 1000
        text = response.choices[0].message.content or ""
        usage = response.usage

        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        cost = _estimate_cost(model, prompt_tokens, completion_tokens)

        return ModelResponse(
            text=text,
            model=model,
            latency_ms=latency,
            token_count=prompt_tokens + completion_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost,
            raw=response,
        )

    return call


def anthropic_model(
    model: str = "claude-3-5-sonnet-20241022",
    api_key: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
):
    """
    Create an Anthropic model function.

    Returns an async function: input → ModelResponse.
    """

    async def call(input_text: str) -> ModelResponse:
        import anthropic

        client = anthropic.AsyncAnthropic(api_key=api_key)
        start = time.monotonic()

        response = await client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": input_text}],
        )

        latency = (time.monotonic() - start) * 1000
        text = response.content[0].text if response.content else ""

        prompt_tokens = response.usage.input_tokens
        completion_tokens = response.usage.output_tokens
        cost = _estimate_cost(model, prompt_tokens, completion_tokens)

        return ModelResponse(
            text=text,
            model=model,
            latency_ms=latency,
            token_count=prompt_tokens + completion_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost,
            raw=response,
        )

    return call


def ollama_model(
    model: str = "llama3:8b",
    base_url: str = "http://localhost:11434",
    temperature: float = 0.0,
):
    """
    Create an Ollama model function (local).

    Returns an async function: input → ModelResponse.
    """

    async def call(input_text: str) -> ModelResponse:
        import httpx

        async with httpx.AsyncClient(base_url=base_url, timeout=120) as client:
            start = time.monotonic()

            resp = await client.post(
                "/api/generate",
                json={
                    "model": model,
                    "prompt": input_text,
                    "stream": False,
                    "options": {"temperature": temperature},
                },
            )
            resp.raise_for_status()
            data = resp.json()

            latency = (time.monotonic() - start) * 1000
            text = data.get("response", "")

            return ModelResponse(
                text=text,
                model=f"ollama:{model}",
                latency_ms=latency,
                token_count=data.get("eval_count", 0),
                prompt_tokens=data.get("prompt_eval_count", 0),
                completion_tokens=data.get("eval_count", 0),
                cost_usd=0.0,  # Local models are free
                raw=data,
            )

    return call


def static_model(responses: dict[str, str], default: str = "I don't know"):
    """
    Create a static mock model for testing.

    Args:
        responses: Dict of {input: output}.
        default: Default response for unknown inputs.
    """

    async def call(input_text: str) -> ModelResponse:
        text = responses.get(input_text, default)
        return ModelResponse(text=text, model="static", token_count=len(text.split()))

    return call


def judge_from_model(model_fn):
    """
    Convert a model function (input → ModelResponse) into a judge function (prompt → str).

    Useful for passing model adapters to LLM-as-Judge metrics.
    """

    async def judge(prompt: str) -> str:
        response = await model_fn(prompt)
        return response.text

    return judge
