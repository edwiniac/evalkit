# EvalKit

**Production-grade LLM evaluation framework** — test hallucination, factuality, relevance, and more.

Stop vibes-checking your LLMs. EvalKit gives you reproducible, quantitative evaluation with built-in metrics, model comparison, and detailed reporting.

---

## ✨ Features

- **📏 Deterministic Metrics** — ExactMatch, ContainsAny/All, RegexMatch, IsJSON, LengthRange
- **🧠 LLM-as-Judge** — Faithfulness, Hallucination, Relevance, Coherence, Toxicity
- **📊 Statistical Metrics** — BLEU, ROUGE, Semantic Similarity
- **⚡ System Metrics** — Latency, Token Count, Cost Estimation
- **🔄 Model Comparison** — Side-by-side evaluation across providers
- **📋 Rich Reporting** — Console, JSON, HTML, CSV
- **🔌 Pluggable** — Custom metrics, any model backend
- **🧪 Test Suite API** — Define, run, and track evaluations like unit tests

## 🚀 Quick Start

```python
from evalkit.models import EvalCase
from evalkit.metrics import ExactMatch, ContainsAny
from evalkit.suite import EvalSuite
from evalkit.runners import EvalRunner
from evalkit.reporters import ConsoleReporter

# Define test cases
suite = EvalSuite(
    name="Geography Quiz",
    cases=[
        EvalCase(input="Capital of France?", expected_output="Paris"),
        EvalCase(input="Capital of Japan?", expected_output="Tokyo"),
    ],
    metrics=[ExactMatch(), ContainsAny()],
)

# Run against your model
runner = EvalRunner()
result = await runner.run(suite, your_model_fn, model_name="gpt-4")

# Report results
ConsoleReporter(verbose=True).print(result)
```

## 📊 Test Coverage

```
90+ tests passing | Sprint-based development

Sprint 1: Foundation (models, deterministic metrics, runner, reporters)
Sprint 2: LLM-as-Judge metrics
Sprint 3: Statistical metrics & model comparison
Sprint 4: CLI & HTML reports
Sprint 5: Polish & examples
```

## 🏗 Architecture

```
┌─────────────────────────────────────────────────┐
│              CLI / Python API                    │
├─────────────────────────────────────────────────┤
│              Eval Runner                         │
├──────────┬──────────┬──────────┬────────────────┤
│ Metrics  │ Judges   │ Datasets │ Model Adapters │
├──────────┴──────────┴──────────┴────────────────┤
│           Reporter Layer                         │
└─────────────────────────────────────────────────┘
```

## 🛠 Development

```bash
git clone https://github.com/edwiniac/evalkit.git
cd evalkit
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"
pytest tests/ -v
```

## 📝 License

MIT — Built by Edwin Isac
