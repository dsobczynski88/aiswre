"""
RTM Review Agent (Ollama/Local version)

LangGraph-based requirement traceability matrix reviewer for medical device software.
Evaluates how well test case suites verify requirements using local Ollama models.
"""

from .core import (
    Requirement,
    RTMEntry,
    RTMReviewLink,
    RTMState,
    # Response models
    FunctionalCoverageResponse,
    InputOutputCoverageResponse,
    BoundaryCoverageResponse,
    NegativeTestCoverageResponse,
    RiskCoverageResponse,
    TraceabilityCompletenessResponse,
    AcceptanceCriteriaCoverageResponse,
    TestSufficiencyResponse,
    GapAnalysisResponse,
    AggregatorResponse
)

from .pipeline import (
    RTMLocalReviewerRunnable,
    get_rtm_local_reviewer_runnable,
    run_batch_rtm_local_review,
    detect_ollama_ports
)

__all__ = [
    # Core data models
    "Requirement",
    "RTMEntry",
    "RTMReviewLink",
    "RTMState",

    # Response models
    "FunctionalCoverageResponse",
    "InputOutputCoverageResponse",
    "BoundaryCoverageResponse",
    "NegativeTestCoverageResponse",
    "RiskCoverageResponse",
    "TraceabilityCompletenessResponse",
    "AcceptanceCriteriaCoverageResponse",
    "TestSufficiencyResponse",
    "GapAnalysisResponse",
    "AggregatorResponse",

    # Pipeline
    "RTMLocalReviewerRunnable",
    "get_rtm_local_reviewer_runnable",
    "run_batch_rtm_local_review",
    "detect_ollama_ports"
]
