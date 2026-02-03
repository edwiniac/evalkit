"""
Eval runner — executes evaluation suites against models.

Orchestrates:
1. Send each case to the model
2. Collect response
3. Run all metrics on the response
4. Aggregate results
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Awaitable, Callable, Optional

from ..metrics.base import EvalMetric
from ..models import (
    CaseResult,
    EvalCase,
    EvalSuiteResult,
    MetricResult,
    ModelResponse,
    Verdict,
)
from ..suite import EvalSuite

logger = logging.getLogger(__name__)

# Type for a model callable: takes input string, returns ModelResponse
ModelFn = Callable[[str], Awaitable[ModelResponse]]


class EvalRunner:
    """
    Runs evaluation suites against model functions.

    Usage:
        runner = EvalRunner()
        result = await runner.run(suite, model_fn, model_name="gpt-4")
    """

    def __init__(
        self,
        concurrency: int = 1,
        on_case_complete: Optional[Callable] = None,
    ):
        """
        Args:
            concurrency: Max concurrent model calls (1 = sequential).
            on_case_complete: Callback(case_index, case_result) after each case.
        """
        self._concurrency = concurrency
        self._on_case_complete = on_case_complete

    async def run(
        self,
        suite: EvalSuite,
        model_fn: ModelFn,
        model_name: str = "unknown",
    ) -> EvalSuiteResult:
        """
        Run a full evaluation suite.

        Args:
            suite: The evaluation suite to run.
            model_fn: Async callable that takes input string → ModelResponse.
            model_name: Name identifier for the model.

        Returns:
            EvalSuiteResult with all case results and aggregates.
        """
        logger.info(
            "Starting eval: '%s' (%d cases, %d metrics, model=%s)",
            suite.name,
            len(suite.cases),
            len(suite.metrics),
            model_name,
        )

        result = EvalSuiteResult(
            suite_name=suite.name,
            model=model_name,
            started_at=datetime.now(),
        )

        if self._concurrency <= 1:
            # Sequential execution
            for i, case in enumerate(suite.cases):
                case_result = await self._eval_case(case, suite.metrics, model_fn)
                result.case_results.append(case_result)

                if self._on_case_complete:
                    self._on_case_complete(i, case_result)

                logger.debug(
                    "Case %d/%d: %s (%.2f)",
                    i + 1,
                    len(suite.cases),
                    "PASS" if case_result.passed else "FAIL",
                    case_result.avg_score,
                )
        else:
            # Concurrent execution with semaphore
            sem = asyncio.Semaphore(self._concurrency)
            tasks = []

            async def bounded_eval(idx: int, case: EvalCase):
                async with sem:
                    cr = await self._eval_case(case, suite.metrics, model_fn)
                    if self._on_case_complete:
                        self._on_case_complete(idx, cr)
                    return idx, cr

            tasks = [bounded_eval(i, case) for i, case in enumerate(suite.cases)]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Sort by index to maintain order
            sorted_results = sorted(
                [(idx, cr) for idx, cr in results if not isinstance(cr, Exception)],
                key=lambda x: x[0],
            )
            result.case_results = [cr for _, cr in sorted_results]

        result.finished_at = datetime.now()

        logger.info(
            "Eval complete: '%s' — %d/%d passed (%.1f%%), avg score: %.2f",
            suite.name,
            result.passed_cases,
            result.total_cases,
            result.pass_rate * 100,
            result.avg_score,
        )

        return result

    async def run_comparison(
        self,
        suite: EvalSuite,
        models: dict[str, ModelFn],
    ) -> dict[str, EvalSuiteResult]:
        """
        Run a suite against multiple models for comparison.

        Args:
            suite: The evaluation suite.
            models: Dict of {model_name: model_fn}.

        Returns:
            Dict of {model_name: EvalSuiteResult}.
        """
        results = {}
        for name, fn in models.items():
            logger.info("Running comparison model: %s", name)
            results[name] = await self.run(suite, fn, model_name=name)
        return results

    async def _eval_case(
        self,
        case: EvalCase,
        metrics: list[EvalMetric],
        model_fn: ModelFn,
    ) -> CaseResult:
        """Evaluate a single case: get response, run all metrics."""
        # Get model response
        try:
            start = time.monotonic()
            response = await model_fn(case.input)
            response.latency_ms = (time.monotonic() - start) * 1000
        except Exception as e:
            logger.warning("Model call failed for case %s: %s", case.case_id, e)
            response = ModelResponse(
                text="",
                model="error",
                latency_ms=0,
            )

        # Run all metrics
        metric_results = []
        for metric in metrics:
            try:
                mr = await metric.score(case, response)
                metric_results.append(mr)
            except Exception as e:
                logger.warning(
                    "Metric '%s' failed for case %s: %s",
                    metric.name,
                    case.case_id,
                    e,
                )
                metric_results.append(
                    MetricResult(
                        metric_name=metric.name,
                        score=0.0,
                        verdict=Verdict.ERROR,
                        reason=f"Metric error: {e}",
                    )
                )

        return CaseResult(
            case=case,
            response=response,
            metric_results=metric_results,
        )
