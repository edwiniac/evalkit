# EvalKit — Development Roadmap

## Sprint Plan

### Sprint 1: Foundation
- Project structure, models, config
- EvalCase, EvalMetric ABC, MetricResult, EvalSuite
- Deterministic metrics: ExactMatch, ContainsAny, RegexMatch
- Basic EvalRunner (single model, sync)
- Console reporter
- Tests for all

### Sprint 2: LLM-as-Judge Metrics
- Judge adapter interface (LLM that scores other LLMs)
- Faithfulness, AnswerRelevance, Hallucination
- Coherence, Toxicity
- Model adapters: OpenAI, Anthropic, Ollama
- Prompt templates for each judge metric

### Sprint 3: Advanced Metrics & Comparison
- Statistical: BLEU, ROUGE, semantic similarity
- System: Latency, CostEstimate, TokenCount
- Multi-model comparison runner
- Dataset loading (JSON, CSV, JSONL)
- Custom metric decorator

### Sprint 4: Reporters & CLI
- HTML report with charts (Jinja2)
- JSON/CSV export
- Comparison report (model vs model)
- CLI interface (click)
- Config file support (YAML)

### Sprint 5: Polish & Examples
- Example eval suites (RAG, QA, summarization)
- README with badges
- Integration tests
- PyPI-ready packaging
