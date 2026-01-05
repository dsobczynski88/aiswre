"""
Ollama-based Test Case Review Agent

LangGraph-based test case reviewer using local Ollama models with multi-port support
for parallel execution. Each graph instance can run on a separate Ollama port.
"""

from .core import (
    TestCaseReviewState,
    TestCaseInput,
    StructureEvaluation,
    ObjectiveEvaluation,
    ReviewSummary
)

from .pipeline import (
    OllamaTestCaseReviewerRunnable,
    get_ollama_reviewer_runnable,
    run_batch_ollama_test_case_review
)

__all__ = [
    'TestCaseReviewState',
    'TestCaseInput',
    'StructureEvaluation',
    'ObjectiveEvaluation',
    'ReviewSummary',
    'OllamaTestCaseReviewerRunnable',
    'get_ollama_reviewer_runnable',
    'run_batch_ollama_test_case_review'
]
