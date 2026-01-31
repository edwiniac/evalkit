# 📊 EvalKit

**Production-grade LLM evaluation framework** — test hallucination, factuality, relevance, and more.

Built for AI engineers who need rigorous, reproducible evaluation of language model outputs.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-188%20passed-brightgreen.svg)](#testing)

---

## ✨ Features

- **20+ metrics** — deterministic, statistical, and LLM-as-Judge
- **Multi-model comparison** — benchmark models side-by-side with rankings
- **Dataset loading** — JSON, JSONL, CSV with auto-detection
- **Beautiful reports** — console, JSON, and interactive HTML
- **Async-first** — concurrent evaluation with configurable parallelism
- **Zero-config** — sensible defaults, customize when needed
- **Type-safe** — Pydantic models throughout

## 🚀 Quick Start

### Installation

```bash
pip install evalkit
```

### Python API

```python
import asyncio
from evalkit import EvalSuite, EvalCase, EvalRunner
from evalkit.adapters import static_model
from evalkit.metrics import ExactMatch, ContainsAny, BLEUScore

# Define test cases
suite = EvalSuite(
    name="Geography Quiz",
    cases=[
        EvalCase(input="Capital of France?", expected_output="Paris"),
        EvalCase(input="Capital of Japan?", expected_output="Tokyo"),
    ],
    metrics=[ExactMatch(), ContainsAny(), BLEUScore()],
)

# Define your model (or use an API adapter)
model = static_model({"Capital of France?": "Paris", "Capital of Japan?": "Tokyo"})

# Run evaluation
result = asyncio.run(EvalRunner().run(suite, model))

# Print results
from evalkit.reporters import ConsoleReporter
ConsoleReporter(verbose=True).print(result)
```

### CLI

```bash
# Run evaluation on a dataset
evalkit run dataset.json --metrics exact,contains,bleu

# Generate HTML report
evalkit run dataset.json --metrics exact,bleu --report html --output report.html

# List available metrics
evalkit list-metrics
```

## 📐 Metrics

### Deterministic Metrics
| Metric | Description |
|--------|-------------|
| `ExactMatch` | Exact string match (configurable case sensitivity) |
| `ContainsAny` | Response contains any keyword |
| `ContainsAll` | Response contains all keywords |
| `RegexMatch` | Pattern matching |
| `IsJSON` | Valid JSON with optional key validation |
| `LengthRange` | Response length within bounds |

### Statistical Metrics
| Metric | Description |
|--------|-------------|
| `BLEUScore` | BLEU n-gram overlap |
| `ROUGEScore` | ROUGE recall-oriented scoring |
| `SemanticSimilarity` | Semantic similarity (Jaccard fallback, sentence-transformers optional) |
| `LatencyMetric` | Response time scoring against targets |
| `CostMetric` | API cost scoring against budgets |

### LLM-as-Judge Metrics
| Metric | Description |
|--------|-------------|
| `Faithfulness` | Grounded in provided context (RAG) |
| `AnswerRelevance` | Relevant to the question asked |
| `Hallucination` | Detects fabricated information |
| `Coherence` | Logical flow and consistency |
| `Toxicity` | Harmful or inappropriate content |
| `Correctness` | Factual accuracy against expected output |

## 📂 Dataset Formats

### JSON
```json
[
    {"input": "What is 2+2?", "expected_output": "4"},
    {"input": "Capital of France?", "expected_output": "Paris", "context": "..."}
]
```

### JSONL
```jsonl
{"input": "What is 2+2?", "expected_output": "4"}
{"input": "Capital of France?", "expected_output": "Paris"}
```

### CSV
```csv
input,expected_output,context
What is 2+2?,4,
Capital of France?,Paris,France is a country in Europe
```

Alternative keys supported: `question`/`prompt` → `input`, `answer`/`expected` → `expected_output`

## 🔄 Model Comparison

```python
results = await runner.run_comparison(
    suite,
    models={
        "gpt-4": gpt4_adapter,
        "claude-3": claude_adapter,
        "llama-3": llama_adapter,
    }
)

# Results ranked by pass rate
from evalkit.reporters import JSONReporter
JSONReporter().save_comparison(results, "comparison.json")
```

## 📊 Reports

### Console
Rich terminal output with color-coded pass/fail, metric summaries, and per-case details.

### JSON
Machine-readable results for CI/CD integration:
```python
JSONReporter().save(result, "results.json")
```

### HTML
Interactive reports with charts and drill-down:
```python
HTMLReporter().save(result, "report.html")
```

## 🏗️ Architecture

```
evalkit/
├── models.py          # Core data models (EvalCase, ModelResponse, etc.)
├── suite.py           # EvalSuite definition with fluent API
├── adapters.py        # Model adapters (static, OpenAI, Anthropic)
├── cli.py             # Click-based CLI
├── metrics/
│   ├── base.py        # Abstract EvalMetric interface
│   ├── deterministic.py   # Exact, contains, regex, JSON, length
│   ├── statistical.py     # BLEU, ROUGE, similarity, latency, cost
│   ├── llm_judge.py       # LLM-as-Judge framework
│   └── judge_prompts.py   # Structured prompts for judge metrics
├── runners/
│   └── runner.py      # Async evaluation engine
├── datasets/
│   └── loader.py      # Multi-format dataset loading
└── reporters/
    ├── console.py     # Rich terminal reporter
    ├── json_reporter.py   # JSON serialization
    └── html_reporter.py   # Interactive HTML reports
```

## 🧪 Testing

```bash
# Run all tests
pytest

# With coverage
pytest --cov=evalkit --cov-report=term-missing

# Unit tests only
pytest tests/unit/

# Integration tests
pytest tests/integration/
```

**188 tests** covering all modules with unit and integration tests.

## 🗺️ Roadmap

- [x] Sprint 1: Core models, deterministic metrics, runner
- [x] Sprint 2: LLM-as-Judge, reporting, adapters
- [x] Sprint 3: Statistical metrics, dataset loading
- [x] Sprint 4: CLI, HTML reporter
- [x] Sprint 5: Integration tests, examples, polish
- [ ] Sprint 6: CI/CD integration, GitHub Actions workflow
- [ ] Sprint 7: Embedding-based similarity, custom judge prompts
- [ ] Sprint 8: Dashboard UI, result diffing

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

## 👤 Author

**Edwin Isac** — AI Engineer  
[GitHub](https://github.com/edwinisac) · [Email](mailto:edwinisac007@gmail.com)
