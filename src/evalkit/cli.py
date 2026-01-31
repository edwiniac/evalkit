"""
EvalKit CLI — run evaluations from the command line.

Usage:
    evalkit run dataset.json --model static --metrics exact,contains
    evalkit run dataset.csv --metrics exact --report html --output report.html
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import click

from .adapters import static_model
from .datasets.loader import load_cases
from .metrics.deterministic import (
    ContainsAll,
    ContainsAny,
    ExactMatch,
    IsJSON,
    LengthRange,
    RegexMatch,
)
from .metrics.statistical import BLEUScore, CostMetric, LatencyMetric, ROUGEScore
from .models import EvalCase, ModelResponse
from .reporters.console import ConsoleReporter
from .reporters.json_reporter import JSONReporter
from .runners.runner import EvalRunner
from .suite import EvalSuite

logger = logging.getLogger(__name__)

# Metric registry
METRIC_MAP = {
    "exact": ExactMatch,
    "contains": ContainsAny,
    "contains_all": ContainsAll,
    "regex": RegexMatch,
    "json": IsJSON,
    "length": LengthRange,
    "bleu": BLEUScore,
    "rouge": ROUGEScore,
    "latency": LatencyMetric,
    "cost": CostMetric,
}


def _build_metrics(metric_names: str) -> list:
    """Parse comma-separated metric names into metric instances."""
    metrics = []
    for name in metric_names.split(","):
        name = name.strip().lower()
        if name in METRIC_MAP:
            metrics.append(METRIC_MAP[name]())
        else:
            click.echo(f"Unknown metric: '{name}'. Available: {', '.join(METRIC_MAP.keys())}")
            sys.exit(1)
    return metrics


@click.group()
@click.version_option(version="0.1.0")
def main():
    """EvalKit — LLM Evaluation Framework."""
    pass


@main.command()
@click.argument("dataset", type=click.Path(exists=True))
@click.option("--metrics", "-m", default="exact", help="Comma-separated metrics (exact,contains,bleu,rouge,json,length,latency,cost)")
@click.option("--name", "-n", default=None, help="Suite name")
@click.option("--report", "-r", default="console", help="Report format: console, json, html")
@click.option("--output", "-o", default=None, help="Output file path")
@click.option("--verbose", "-v", is_flag=True, help="Show per-case details")
@click.option("--concurrency", "-c", default=1, help="Concurrent model calls")
def run(dataset, metrics, name, report, output, verbose, concurrency):
    """Run an evaluation suite on a dataset."""
    # Load dataset
    click.echo(f"Loading dataset: {dataset}")
    cases = load_cases(dataset)
    click.echo(f"Loaded {len(cases)} cases")

    # Build metrics
    metric_instances = _build_metrics(metrics)
    click.echo(f"Metrics: {', '.join(m.name for m in metric_instances)}")

    # Build suite
    suite_name = name or Path(dataset).stem
    suite = EvalSuite(name=suite_name, cases=cases, metrics=metric_instances)

    # Create a simple echo model (for demo without API keys)
    async def echo_model(input_text: str) -> ModelResponse:
        # In CLI mode without --model flag, just echo a placeholder
        return ModelResponse(text="", model="echo", latency_ms=0)

    # Run evaluation
    runner = EvalRunner(concurrency=concurrency)
    result = asyncio.run(runner.run(suite, echo_model, model_name="echo"))

    # Report
    if report == "console":
        ConsoleReporter(verbose=verbose).print(result)
    elif report == "json":
        reporter = JSONReporter()
        if output:
            reporter.save(result, Path(output))
            click.echo(f"Report saved to {output}")
        else:
            click.echo(reporter.to_json(result))
    elif report == "html":
        from .reporters.html_reporter import HTMLReporter
        reporter = HTMLReporter()
        out_path = Path(output or f"evalkit_report_{result.run_id}.html")
        reporter.save(result, out_path)
        click.echo(f"HTML report saved to {out_path}")
    else:
        click.echo(f"Unknown report format: {report}")


@main.command()
def list_metrics():
    """List available metrics."""
    click.echo("Available metrics:")
    click.echo("")
    for name, cls in METRIC_MAP.items():
        click.echo(f"  {name:<15} {cls.__name__}")
    click.echo("")
    click.echo("LLM-as-Judge metrics (require --judge):")
    click.echo("  faithfulness, relevance, hallucination, coherence, toxicity, correctness")


@main.command()
@click.argument("result_file", type=click.Path(exists=True))
@click.option("--format", "-f", "fmt", default="console", help="Output format: console, html")
@click.option("--output", "-o", default=None, help="Output file")
@click.option("--verbose", "-v", is_flag=True)
def report(result_file, fmt, output, verbose):
    """Generate a report from a saved JSON result."""
    data = json.loads(Path(result_file).read_text())

    from .models import EvalSuiteResult, CaseResult, EvalCase, ModelResponse, MetricResult, Verdict

    # Reconstruct result from JSON
    case_results = []
    for cr_data in data.get("case_results", []):
        case = EvalCase(**cr_data["case"])
        response = ModelResponse(**{k: v for k, v in cr_data["response"].items() if k != "raw"})
        metric_results = [
            MetricResult(
                metric_name=mr["metric_name"],
                score=mr["score"],
                verdict=Verdict(mr["verdict"]),
                reason=mr.get("reason", ""),
                threshold=mr.get("threshold", 0.5),
            )
            for mr in cr_data.get("metric_results", [])
        ]
        case_results.append(CaseResult(case=case, response=response, metric_results=metric_results))

    result = EvalSuiteResult(
        suite_name=data["suite_name"],
        model=data["model"],
        run_id=data["run_id"],
        case_results=case_results,
    )

    if fmt == "console":
        ConsoleReporter(verbose=verbose).print(result)
    elif fmt == "html":
        from .reporters.html_reporter import HTMLReporter
        out_path = Path(output or f"evalkit_report_{result.run_id}.html")
        HTMLReporter().save(result, out_path)
        click.echo(f"HTML report saved to {out_path}")


if __name__ == "__main__":
    main()
