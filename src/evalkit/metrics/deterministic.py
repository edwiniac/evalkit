"""
Deterministic metrics — no LLM required.

Fast, cheap, and predictable scoring functions for basic checks.
These are the foundation; use LLM-judge metrics for nuanced evaluation.
"""

import json
import re
from typing import Optional

from ..models import EvalCase, MetricResult, ModelResponse
from .base import EvalMetric


class ExactMatch(EvalMetric):
    """
    Checks if the response exactly matches the expected output.
    
    Options:
        case_sensitive: Whether comparison is case-sensitive (default: False).
        strip: Whether to strip whitespace before comparing (default: True).
    """

    def __init__(
        self,
        case_sensitive: bool = False,
        strip: bool = True,
        threshold: float = 1.0,
        name: Optional[str] = None,
    ):
        super().__init__(name=name or "ExactMatch", threshold=threshold)
        self._case_sensitive = case_sensitive
        self._strip = strip

    async def score(self, case: EvalCase, response: ModelResponse) -> MetricResult:
        if case.expected_output is None:
            return self._error_result("No expected_output provided")

        actual = response.text
        expected = case.expected_output

        if self._strip:
            actual = actual.strip()
            expected = expected.strip()

        if not self._case_sensitive:
            actual = actual.lower()
            expected = expected.lower()

        match = actual == expected
        return self._make_result(
            score=1.0 if match else 0.0,
            reason="Exact match" if match else f"Expected: '{expected[:100]}', Got: '{actual[:100]}'",
        )


class ContainsAny(EvalMetric):
    """
    Checks if the response contains any of the specified keywords.
    
    Score = proportion of keywords found (0 to 1).
    """

    def __init__(
        self,
        keywords: Optional[list[str]] = None,
        case_sensitive: bool = False,
        threshold: float = 0.5,
        name: Optional[str] = None,
    ):
        super().__init__(name=name or "ContainsAny", threshold=threshold)
        self._keywords = keywords or []
        self._case_sensitive = case_sensitive

    async def score(self, case: EvalCase, response: ModelResponse) -> MetricResult:
        keywords = self._keywords
        if not keywords and case.expected_output:
            # Fall back to using expected_output as single keyword
            keywords = [case.expected_output]

        if not keywords:
            return self._error_result("No keywords provided")

        text = response.text if self._case_sensitive else response.text.lower()
        found = []
        for kw in keywords:
            check_kw = kw if self._case_sensitive else kw.lower()
            if check_kw in text:
                found.append(kw)

        score = len(found) / len(keywords)
        return self._make_result(
            score=score,
            reason=f"Found {len(found)}/{len(keywords)}: {found}" if found
                   else f"None of {keywords} found in response",
            found=found,
            missing=[k for k in keywords if k not in found],
        )


class ContainsAll(EvalMetric):
    """
    Checks if the response contains ALL of the specified keywords.
    
    Score = 1.0 if all present, else proportion found.
    """

    def __init__(
        self,
        keywords: Optional[list[str]] = None,
        case_sensitive: bool = False,
        threshold: float = 1.0,
        name: Optional[str] = None,
    ):
        super().__init__(name=name or "ContainsAll", threshold=threshold)
        self._keywords = keywords or []
        self._case_sensitive = case_sensitive

    async def score(self, case: EvalCase, response: ModelResponse) -> MetricResult:
        keywords = self._keywords
        if not keywords:
            return self._error_result("No keywords provided")

        text = response.text if self._case_sensitive else response.text.lower()
        found = []
        missing = []
        for kw in keywords:
            check_kw = kw if self._case_sensitive else kw.lower()
            if check_kw in text:
                found.append(kw)
            else:
                missing.append(kw)

        score = len(found) / len(keywords)
        return self._make_result(
            score=score,
            reason=f"Found {len(found)}/{len(keywords)}"
                   + (f", missing: {missing}" if missing else ""),
            found=found,
            missing=missing,
        )


class RegexMatch(EvalMetric):
    """
    Checks if the response matches a regex pattern.
    
    Score = 1.0 if pattern found, else 0.0.
    """

    def __init__(
        self,
        pattern: str = "",
        flags: int = re.IGNORECASE,
        threshold: float = 1.0,
        name: Optional[str] = None,
    ):
        super().__init__(name=name or "RegexMatch", threshold=threshold)
        self._pattern = pattern
        self._flags = flags

    async def score(self, case: EvalCase, response: ModelResponse) -> MetricResult:
        pattern = self._pattern
        if not pattern and case.expected_output:
            pattern = re.escape(case.expected_output)

        if not pattern:
            return self._error_result("No regex pattern provided")

        try:
            match = re.search(pattern, response.text, self._flags)
            return self._make_result(
                score=1.0 if match else 0.0,
                reason=f"Pattern {'found' if match else 'not found'}: {pattern[:80]}",
            )
        except re.error as e:
            return self._error_result(f"Invalid regex: {e}")


class IsJSON(EvalMetric):
    """
    Checks if the response is valid JSON.
    
    Optionally checks for required keys.
    """

    def __init__(
        self,
        required_keys: Optional[list[str]] = None,
        threshold: float = 1.0,
        name: Optional[str] = None,
    ):
        super().__init__(name=name or "IsJSON", threshold=threshold)
        self._required_keys = required_keys or []

    async def score(self, case: EvalCase, response: ModelResponse) -> MetricResult:
        text = response.text.strip()

        # Try to extract JSON from markdown fences
        if "```json" in text:
            start = text.index("```json") + 7
            end = text.index("```", start)
            text = text[start:end].strip()
        elif "```" in text:
            start = text.index("```") + 3
            end = text.index("```", start)
            text = text[start:end].strip()

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as e:
            return self._make_result(
                score=0.0,
                reason=f"Invalid JSON: {e}",
            )

        if not self._required_keys:
            return self._make_result(score=1.0, reason="Valid JSON")

        if not isinstance(parsed, dict):
            return self._make_result(
                score=0.5,
                reason="Valid JSON but not an object (required_keys check skipped)",
            )

        found = [k for k in self._required_keys if k in parsed]
        missing = [k for k in self._required_keys if k not in parsed]
        score = len(found) / len(self._required_keys)

        return self._make_result(
            score=score,
            reason=f"JSON keys: {len(found)}/{len(self._required_keys)}"
                   + (f", missing: {missing}" if missing else ""),
            found_keys=found,
            missing_keys=missing,
        )


class LengthRange(EvalMetric):
    """
    Checks if the response length is within a specified range.
    
    Score = 1.0 if within range, degrades linearly outside.
    """

    def __init__(
        self,
        min_chars: int = 0,
        max_chars: int = 10000,
        threshold: float = 0.5,
        name: Optional[str] = None,
    ):
        super().__init__(name=name or "LengthRange", threshold=threshold)
        self._min = min_chars
        self._max = max_chars

    async def score(self, case: EvalCase, response: ModelResponse) -> MetricResult:
        length = len(response.text)

        if self._min <= length <= self._max:
            return self._make_result(
                score=1.0,
                reason=f"Length {length} within [{self._min}, {self._max}]",
                length=length,
            )

        # Score degrades linearly outside range
        if length < self._min:
            score = max(0.0, length / self._min) if self._min > 0 else 0.0
            reason = f"Too short: {length} < {self._min}"
        else:
            overshoot = length - self._max
            score = max(0.0, 1.0 - overshoot / self._max)
            reason = f"Too long: {length} > {self._max}"

        return self._make_result(score=score, reason=reason, length=length)
