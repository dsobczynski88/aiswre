"""
Core data models for RTM (Requirement Traceability Matrix) review agent (Ollama/Local version).

Defines Pydantic models for requirements, test case summaries, and evaluation responses
focused on assessing verification coverage in requirement traceability matrices.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, TypedDict, Annotated
import operator


class RTMState(TypedDict, total=False):
    """State for LangGraph RTM review pipeline."""
    requirement: 'Requirement'
    rtm_entry: 'RTMEntry'
    # Raw evaluator responses (dicts with "type" and "data" keys)
    raw_evaluator_responses: Annotated[List[dict], operator.add]
    # Converted review links (proper RTMReviewLink objects)
    rtm_links: List['RTMReviewLink']
    final_result: Optional['RTMReviewLink']  # Final aggregated result


class Requirement(BaseModel):
    """Software requirement model."""
    req_id: str
    text: str
    risk_id: Optional[str] = None  # For risk-based testing
    safety_class: Optional[str] = None  # A, B, or C per IEC 62304
    acceptance_criteria: Optional[str] = None  # Acceptance criteria if defined


class RTMEntry(BaseModel):
    """
    RTM entry containing a requirement and summary of test cases traced to it.

    This represents one row in a requirement traceability matrix.
    """
    req_id: str
    test_case_summary: str  # Description of all test cases traced to this requirement
    test_case_count: Optional[int] = None  # Number of test cases if known
    test_ids: Optional[str] = None  # Comma-separated list of test IDs if available


class RTMReviewLink(BaseModel):
    """RTM review link with verification coverage scores."""
    req_id: str

    # Coverage Evaluators (0-1 scores)
    functional_coverage_score: float = 0.0
    input_output_coverage_score: float = 0.0
    boundary_coverage_score: float = 0.0
    negative_test_coverage_score: float = 0.0

    # Risk & Traceability (0-1 scores)
    risk_coverage_score: float = 0.0
    traceability_completeness_score: float = 0.0
    acceptance_criteria_coverage_score: float = 0.0

    # Sufficiency & Gaps (0-1 scores)
    test_sufficiency_score: float = 0.0
    gap_analysis_score: float = 0.0

    # Metadata
    rationale: str = ""
    link_type: str = "RTMLocal"
    reviewer_status: str = "Pending"
    issues: List[str] = Field(default_factory=list)

    # Aggregated analysis (populated by aggregator node)
    review_summary: str = ""
    verification_gaps: str = ""
    recommendations: str = ""


# ============================================================================
# LLM Response Models (one per evaluator node)
# ============================================================================

class FunctionalCoverageResponse(BaseModel):
    """Response for functional coverage evaluation."""
    functional_coverage_score: float = Field(..., ge=0.0, le=1.0, description="0-1 score for functional requirement coverage")
    all_functions_tested: bool = Field(..., description="Are all functional aspects of the requirement verified?")
    missing_functions: List[str] = Field(default_factory=list, description="Functional aspects not covered by test cases")
    rationale: str


class InputOutputCoverageResponse(BaseModel):
    """Response for input/output coverage evaluation."""
    input_output_coverage_score: float = Field(..., ge=0.0, le=1.0, description="0-1 score for input/output coverage")
    all_inputs_tested: bool = Field(..., description="Are all input conditions tested?")
    all_outputs_verified: bool = Field(..., description="Are all expected outputs verified?")
    missing_inputs: List[str] = Field(default_factory=list, description="Input conditions not tested")
    missing_outputs: List[str] = Field(default_factory=list, description="Expected outputs not verified")
    rationale: str


class BoundaryCoverageResponse(BaseModel):
    """Response for boundary condition coverage evaluation."""
    boundary_coverage_score: float = Field(..., ge=0.0, le=1.0, description="0-1 score for boundary testing coverage")
    has_boundary_tests: bool = Field(..., description="Are boundary conditions tested (min, max, edges)?")
    missing_boundaries: List[str] = Field(default_factory=list, description="Boundary conditions not tested")
    rationale: str


class NegativeTestCoverageResponse(BaseModel):
    """Response for negative/error case coverage evaluation."""
    negative_test_coverage_score: float = Field(..., ge=0.0, le=1.0, description="0-1 score for negative test coverage")
    has_negative_tests: bool = Field(..., description="Are error cases, invalid inputs, and exceptions tested?")
    missing_negative_cases: List[str] = Field(default_factory=list, description="Error/exception scenarios not tested")
    rationale: str


class RiskCoverageResponse(BaseModel):
    """Response for risk mitigation coverage evaluation."""
    risk_coverage_score: float = Field(..., ge=0.0, le=1.0, description="0-1 score for risk control verification")
    is_linked_to_risk: bool = Field(..., description="Is requirement linked to a Risk ID?")
    risk_controls_verified: bool = Field(False, description="Are risk controls verified by test cases?")
    verification_gaps: List[str] = Field(default_factory=list, description="Gaps in risk mitigation verification")
    rationale: str


class TraceabilityCompletenessResponse(BaseModel):
    """Response for traceability completeness evaluation."""
    traceability_completeness_score: float = Field(..., ge=0.0, le=1.0, description="0-1 score for traceability completeness")
    has_clear_mapping: bool = Field(..., description="Is the req-to-test mapping clear and complete?")
    test_ids_identified: List[str] = Field(default_factory=list, description="Test IDs identified in summary")
    traceability_issues: List[str] = Field(default_factory=list, description="Issues with traceability mapping")
    rationale: str


class AcceptanceCriteriaCoverageResponse(BaseModel):
    """Response for acceptance criteria coverage evaluation."""
    acceptance_criteria_coverage_score: float = Field(..., ge=0.0, le=1.0, description="0-1 score for acceptance criteria coverage")
    has_acceptance_criteria: bool = Field(..., description="Are acceptance criteria defined?")
    all_criteria_verified: bool = Field(False, description="Are all acceptance criteria verified by tests?")
    missing_criteria: List[str] = Field(default_factory=list, description="Acceptance criteria not verified")
    rationale: str


class TestSufficiencyResponse(BaseModel):
    """Response for test sufficiency evaluation."""
    test_sufficiency_score: float = Field(..., ge=0.0, le=1.0, description="0-1 score for test quantity/quality sufficiency")
    has_sufficient_tests: bool = Field(..., description="Are there enough tests to fully verify the requirement?")
    test_count_assessment: str = Field("", description="Assessment of test case count")
    sufficiency_concerns: List[str] = Field(default_factory=list, description="Concerns about test sufficiency")
    rationale: str


class GapAnalysisResponse(BaseModel):
    """Response for verification gap analysis."""
    gap_analysis_score: float = Field(..., ge=0.0, le=1.0, description="0-1 score (inverse of gap severity)")
    critical_gaps: List[str] = Field(default_factory=list, description="Critical verification gaps identified")
    moderate_gaps: List[str] = Field(default_factory=list, description="Moderate verification gaps")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations to close gaps")
    rationale: str


class AggregatorResponse(BaseModel):
    """Response for aggregating all evaluator outputs into final RTM review."""
    review_summary: str = Field(..., description="Comprehensive summary of RTM verification coverage")
    verification_gaps: str = Field(..., description="Detailed description of verification gaps found")
    recommendations: str = Field(..., description="Specific, actionable recommendations to improve verification coverage")
