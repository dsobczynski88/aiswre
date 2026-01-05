"""
MedTech Test Case Review Agent (Ollama/Local version)

LangGraph-based test case reviewer for medical device software testing using local Ollama models.
Evaluates test cases against FDA/IEC 62304 best practices with full privacy and no API costs.
"""

from .core import (
    TestCaseState,
    Requirement,
    TestCase,
    MedtechTraceLink,
    # Response models
    UnambiguityResponse,
    IndependenceResponse,
    PreconditionsResponse,
    PostconditionsResponse,
    TechniqueApplicationResponse,
    NegativeTestingResponse,
    BoundaryChecksResponse,
    RiskVerificationResponse,
    TraceabilityResponse,
    SafetyClassRigorResponse,
    ObjectiveEvidenceResponse
)

from .pipeline import (
    MedtechLocalTestCaseReviewerRunnable,
    get_medtech_local_reviewer_runnable,
    run_batch_medtech_local_test_case_review,
    detect_ollama_ports
)

__all__ = [
    'TestCaseState',
    'Requirement',
    'TestCase',
    'MedtechTraceLink',
    'MedtechLocalTestCaseReviewerRunnable',
    'get_medtech_local_reviewer_runnable',
    'run_batch_medtech_local_test_case_review',
    'detect_ollama_ports'
]
