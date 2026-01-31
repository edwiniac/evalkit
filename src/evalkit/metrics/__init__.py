"""
Metrics module — scoring functions for LLM evaluation.
"""

from .base import EvalMetric
from .deterministic import (
    ContainsAll,
    ContainsAny,
    ExactMatch,
    IsJSON,
    LengthRange,
    RegexMatch,
)
from .llm_judge import (
    AnswerRelevance,
    Coherence,
    Correctness,
    Faithfulness,
    Hallucination,
    LLMJudgeMetric,
    Toxicity,
)
from .statistical import (
    BLEUScore,
    CostMetric,
    LatencyMetric,
    ROUGEScore,
    SemanticSimilarity,
)

__all__ = [
    "EvalMetric",
    # Deterministic
    "ExactMatch",
    "ContainsAny",
    "ContainsAll",
    "RegexMatch",
    "IsJSON",
    "LengthRange",
    # LLM-as-Judge
    "LLMJudgeMetric",
    "Faithfulness",
    "AnswerRelevance",
    "Hallucination",
    "Coherence",
    "Toxicity",
    "Correctness",
    # Statistical
    "BLEUScore",
    "ROUGEScore",
    "SemanticSimilarity",
    "LatencyMetric",
    "CostMetric",
]
