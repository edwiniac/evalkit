"""
Console reporter — rich terminal output for evaluation results.
"""

from ..models import EvalSuiteResult, Verdict


class ConsoleReporter:
    """Prints evaluation results to the console with formatting."""

    def __init__(self, verbose: bool = False):
        self._verbose = verbose

    def report(self, result: EvalSuiteResult) -> str:
        """Generate a formatted console report string."""
        lines = []

        # Header
        lines.append("")
        lines.append("=" * 70)
        lines.append(f"  EvalKit Report: {result.suite_name}")
        lines.append(f"  Model: {result.model} | Run: {result.run_id}")
        lines.append("=" * 70)
        lines.append("")

        # Summary
        pass_icon = "✅" if result.pass_rate >= 0.8 else "⚠️" if result.pass_rate >= 0.5 else "❌"
        lines.append(
            f"  {pass_icon} Pass Rate: {result.passed_cases}"
            f"/{result.total_cases} ({result.pass_rate:.0%})"
        )
        lines.append(f"  📊 Avg Score: {result.avg_score:.2f}")
        lines.append(f"  ⏱  Avg Latency: {result.avg_latency_ms:.0f}ms")
        if result.total_cost_usd > 0:
            lines.append(f"  💰 Total Cost: ${result.total_cost_usd:.4f}")
        lines.append("")

        # Metric breakdown
        summary = result.metric_summary()
        if summary:
            lines.append("  Metrics:")
            lines.append("  " + "-" * 50)
            for name, stats in summary.items():
                bar = self._score_bar(stats["avg"])
                lines.append(
                    f"  {name:<25} {bar} {stats['avg']:.2f}  "
                    f"(min={stats['min']:.2f}, max={stats['max']:.2f})"
                )
            lines.append("")

        # Per-case details (verbose)
        if self._verbose:
            lines.append("  Case Details:")
            lines.append("  " + "-" * 50)
            for i, cr in enumerate(result.case_results):
                icon = "✅" if cr.passed else "❌"
                input_preview = cr.case.input[:60].replace("\n", " ")
                lines.append(f"  {icon} Case {i+1}: {input_preview}")
                for mr in cr.metric_results:
                    v_icon = (
                        "✓"
                        if mr.verdict == Verdict.PASS
                        else "✗" if mr.verdict == Verdict.FAIL else "!"
                    )
                    lines.append(
                        f"     {v_icon} {mr.metric_name}: {mr.score:.2f} — {mr.reason[:80]}"
                    )
            lines.append("")

        lines.append("=" * 70)
        lines.append("")

        return "\n".join(lines)

    def print(self, result: EvalSuiteResult) -> None:
        """Print the report to stdout."""
        print(self.report(result))

    def print_comparison(self, results: dict[str, EvalSuiteResult]) -> None:
        """Print a comparison table for multiple models."""
        lines = []
        lines.append("")
        lines.append("=" * 70)
        lines.append("  Model Comparison")
        lines.append("=" * 70)
        lines.append("")

        # Header
        lines.append(
            f"  {'Model':<20} {'Pass Rate':>10} {'Avg Score':>10} " f"{'Latency':>10} {'Cost':>10}"
        )
        lines.append("  " + "-" * 62)

        for name, result in results.items():
            lines.append(
                f"  {name:<20} {result.pass_rate:>9.0%} {result.avg_score:>10.2f} "
                f"{result.avg_latency_ms:>8.0f}ms ${result.total_cost_usd:>8.4f}"
            )

        lines.append("")
        lines.append("=" * 70)
        print("\n".join(lines))

    @staticmethod
    def _score_bar(score: float, width: int = 10) -> str:
        filled = int(score * width)
        return "█" * filled + "░" * (width - filled)
