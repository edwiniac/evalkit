"""
LLM-as-Judge metrics — uses one LLM to evaluate another.

Each metric sends a structured prompt to a judge model, asking it
to score a specific quality dimension (faithfulness, relevance, etc.)
and return a JSON score with reasoning.
"""

import json
import logging
from typing import Any, Awaitable, Callable, Optional

from ..models import EvalCase, MetricResult, ModelResponse
from . import judge_prompts
from .base import EvalMetric

logger = logging.getLogger(__name__)

# Type for a judge function: takes prompt string, returns text response
JudgeFn = Callable[[str], Awaitable[str]]


class LLMJudgeMetric(EvalMetric):
    """
    Base class for LLM-as-Judge metrics.

    Subclasses define the prompt template and parse the response.
    The judge function is injected — can be any LLM provider.
    """

    def __init__(
        self,
        judge_fn: JudgeFn,
        name: Optional[str] = None,
        threshold: float = 0.5,
        prompt_template: str = "",
    ):
        super().__init__(name=name, threshold=threshold)
        self._judge_fn = judge_fn
        self._prompt_template = prompt_template

    def _format_prompt(self, case: EvalCase, response: ModelResponse) -> str:
        """Format the prompt template with case and response data."""
        return self._prompt_template.format(
            input=case.input,
            response=response.text,
            expected=case.expected_output or "(not provided)",
            context=case.context_str or "(not provided)",
        )

    def _parse_judge_response(self, text: str) -> dict[str, Any]:
        """Extract JSON from judge model response."""
        text = text.strip()

        # Handle markdown fences
        if "```json" in text:
            start = text.index("```json") + 7
            end = text.index("```", start)
            text = text[start:end].strip()
        elif "```" in text:
            start = text.index("```") + 3
            end = text.index("```", start)
            text = text[start:end].strip()

        # Find JSON object
        brace_start = text.find("{")
        brace_end = text.rfind("}")
        if brace_start != -1 and brace_end != -1:
            text = text[brace_start : brace_end + 1]

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Failed to parse judge response: %s", text[:200])
            return {}

    async def score(self, case: EvalCase, response: ModelResponse) -> MetricResult:
        """Send prompt to judge and parse the result."""
        try:
            prompt = self._format_prompt(case, response)
            judge_response = await self._judge_fn(prompt)
            parsed = self._parse_judge_response(judge_response)

            if not parsed:
                return self._error_result("Could not parse judge response")

            score = float(parsed.get("score", 0.0))
            score = max(0.0, min(1.0, score))  # Clamp to [0, 1]
            reason = parsed.get("reason", "")

            # Extract additional metadata
            metadata = {k: v for k, v in parsed.items() if k not in ("score", "verdict", "reason")}

            return self._make_result(
                score=score,
                reason=reason,
                **metadata,
            )

        except Exception as e:
            logger.error("Judge metric '%s' failed: %s", self._name, e)
            return self._error_result(str(e))


class Faithfulness(LLMJudgeMetric):
    """
    Evaluates if the response is grounded in the provided context.

    High score = response only contains claims supported by the context.
    Requires: case.context to be set.
    """

    def __init__(self, judge_fn: JudgeFn, threshold: float = 0.7):
        super().__init__(
            judge_fn=judge_fn,
            name="Faithfulness",
            threshold=threshold,
            prompt_template=judge_prompts.FAITHFULNESS_PROMPT,
        )

    async def score(self, case: EvalCase, response: ModelResponse) -> MetricResult:
        if not case.context:
            return self._make_result(
                score=0.0,
                reason="No context provided — cannot evaluate faithfulness",
            )
        return await super().score(case, response)


class AnswerRelevance(LLMJudgeMetric):
    """
    Evaluates if the response is relevant to the question asked.

    High score = response directly addresses the question.
    """

    def __init__(self, judge_fn: JudgeFn, threshold: float = 0.7):
        super().__init__(
            judge_fn=judge_fn,
            name="AnswerRelevance",
            threshold=threshold,
            prompt_template=judge_prompts.ANSWER_RELEVANCE_PROMPT,
        )


class Hallucination(LLMJudgeMetric):
    """
    Detects hallucinations (fabricated information) in the response.

    High score = fewer hallucinations (good).
    """

    def __init__(self, judge_fn: JudgeFn, threshold: float = 0.7):
        super().__init__(
            judge_fn=judge_fn,
            name="Hallucination",
            threshold=threshold,
            prompt_template=judge_prompts.HALLUCINATION_PROMPT,
        )


class Coherence(LLMJudgeMetric):
    """
    Evaluates structural quality and logical flow of the response.

    High score = well-organized, easy to follow.
    """

    def __init__(self, judge_fn: JudgeFn, threshold: float = 0.6):
        super().__init__(
            judge_fn=judge_fn,
            name="Coherence",
            threshold=threshold,
            prompt_template=judge_prompts.COHERENCE_PROMPT,
        )


class Toxicity(LLMJudgeMetric):
    """
    Detects toxic, harmful, or inappropriate content.

    High score = safe content (good).
    """

    def __init__(self, judge_fn: JudgeFn, threshold: float = 0.8):
        super().__init__(
            judge_fn=judge_fn,
            name="Toxicity",
            threshold=threshold,
            prompt_template=judge_prompts.TOXICITY_PROMPT,
        )


class Correctness(LLMJudgeMetric):
    """
    Evaluates factual correctness against expected answer.

    High score = factually correct.
    """

    def __init__(self, judge_fn: JudgeFn, threshold: float = 0.7):
        super().__init__(
            judge_fn=judge_fn,
            name="Correctness",
            threshold=threshold,
            prompt_template=judge_prompts.CORRECTNESS_PROMPT,
        )
