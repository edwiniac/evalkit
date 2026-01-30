# EvalKit — Architecture

## Vision

A production-grade LLM evaluation framework that goes beyond "vibe checks."
Test hallucination, factuality, relevance, toxicity, and custom metrics —
across any model, with reproducible results.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    CLI / Python API                      │
├─────────────────────────────────────────────────────────┤
│                    Eval Runner                           │
│           (orchestrates evaluation runs)                 │
├──────────┬──────────┬──────────┬────────────────────────┤
│ Metrics  │ Judges   │ Datasets │   Model Adapters       │
│ Engine   │          │          │                        │
├──────────┴──────────┴──────────┴────────────────────────┤
│                   Reporter Layer                         │
│           (JSON, HTML, Console, CSV)                    │
└─────────────────────────────────────────────────────────┘
```

## Core Concepts

### EvalCase
A single evaluation test case:
```python
EvalCase(
    input="What is the capital of France?",
    expected_output="Paris",
    context="France is a country in Europe...",  # optional RAG context
    metadata={"category": "geography"},
)
```

### EvalMetric
A scoring function that evaluates a model response:
```python
class Faithfulness(EvalMetric):
    """Checks if response is grounded in the provided context."""
    async def score(self, case, response) -> MetricResult
```

### EvalSuite
A collection of test cases + metrics:
```python
suite = EvalSuite(
    name="RAG Quality",
    cases=[...],
    metrics=[Faithfulness(), Relevance(), Toxicity()],
)
```

### EvalRun
Executes a suite against one or more models:
```python
run = EvalRun(suite, models=["gpt-4", "claude-3"])
results = await run.execute()
```

## Metrics

### Built-in Metrics
| Metric | Type | Description |
|--------|------|-------------|
| **Faithfulness** | LLM-judge | Is the response grounded in context? |
| **AnswerRelevance** | LLM-judge | Does the response address the question? |
| **Hallucination** | LLM-judge | Does the response contain fabricated info? |
| **Toxicity** | LLM-judge | Is the response harmful or inappropriate? |
| **Coherence** | LLM-judge | Is the response well-structured? |
| **ExactMatch** | Deterministic | Does output exactly match expected? |
| **ContainsAny** | Deterministic | Does output contain expected keywords? |
| **BLEUScore** | Statistical | BLEU score against reference |
| **SemanticSimilarity** | Embedding | Cosine similarity of embeddings |
| **Latency** | System | Response time in ms |
| **CostEstimate** | System | Estimated token cost |

### Custom Metrics
```python
@evalkit.metric("my_metric")
async def check_json_valid(case, response):
    try:
        json.loads(response.text)
        return 1.0
    except:
        return 0.0
```

## Model Adapters

Pluggable backends:
- **OpenAI** (GPT-4, GPT-3.5)
- **Anthropic** (Claude 3/3.5)
- **Ollama** (local models)
- **Custom** (any callable)

## Reporters

- **Console** — Rich terminal tables
- **JSON** — Machine-readable results
- **HTML** — Interactive report with charts
- **CSV** — Spreadsheet export
- **Comparison** — Side-by-side model comparison
