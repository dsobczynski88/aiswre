"""
Core data models for MedTech test case review agent (Ollama/Local version).

Defines Pydantic models for test cases, requirements, and evaluation responses
aligned with FDA/IEC 62304 testing best practices, adapted for local LLM execution.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, TypedDict, Annotated
import operator


class TestCaseState(TypedDict, total=False):
    """State for LangGraph medtech test case review pipeline."""
    requirement: Optional['Requirement']
    test: 'TestCase'
    # Raw evaluator responses (dicts with "type" and "data" keys)
    raw_evaluator_responses: Annotated[List[dict], operator.add]
    # Converted trace links (proper MedtechTraceLink objects)
    medtech_links: List['MedtechTraceLink']
    final_result: Optional['MedtechTraceLink']  # Final aggregated result


class Requirement(BaseModel):
    """Software requirement model."""
    req_id: Optional[str] = None
    text: str
    risk_id: Optional[str] = None  # For risk-based testing
    safety_class: Optional[str] = None  # A, B, or C per IEC 62304


class TestCase(BaseModel):
    """Test case model with medtech-specific fields."""
    test_id: str
    description: str
    preconditions: Optional[str] = None
    steps: Optional[str] = None
    expected_result: Optional[str] = None
    postconditions: Optional[str] = None
    test_type: Optional[str] = None  # e.g., "Unit", "Integration", "System"
    technique: Optional[str] = None  # e.g., "EP", "BVA", "Decision Table"


class MedtechTraceLink(BaseModel):
    """Trace link with medtech evaluation scores."""
    req_id: str
    test_id: str

    # General Integrity & Structure scores (0-1)
    unambiguity_score: float = 0.0
    independence_score: float = 0.0
    preconditions_score: float = 0.0
    postconditions_score: float = 0.0

    # Coverage & Technique scores (0-1)
    technique_application_score: float = 0.0
    negative_testing_score: float = 0.0
    boundary_checks_score: float = 0.0
    risk_verification_score: float = 0.0

    # Traceability & Compliance scores (0-1)
    traceability_score: float = 0.0
    safety_class_rigor_score: float = 0.0
    objective_evidence_score: float = 0.0

    # Metadata
    rationale: str = ""
    link_type: str = "MedtechLocal"
    reviewer_status: str = "Pending"
    issues: List[str] = Field(default_factory=list)

    # Aggregated analysis (populated by aggregator node)
    review_summary: str = ""
    test_case_improvements: str = ""


# ============================================================================
# LLM Response Models (one per evaluator node)
# ============================================================================

# General Integrity & Structure
class UnambiguityResponse(BaseModel):
    """Response for unambiguity evaluation."""
    unambiguity_score: float = Field(..., ge=0.0, le=1.0, description="0-1 score for instruction clarity")
    is_unambiguous: bool = Field(..., description="Can someone with no prior knowledge execute this?")
    ambiguous_phrases: List[str] = Field(default_factory=list, description="List of unclear phrases")
    rationale: str


class IndependenceResponse(BaseModel):
    """Response for test independence evaluation."""
    independence_score: float = Field(..., ge=0.0, le=1.0, description="0-1 score for test self-containment")
    is_independent: bool = Field(..., description="Is the test self-contained?")
    dependencies: List[str] = Field(default_factory=list, description="External dependencies found")
    cleanup_issues: List[str] = Field(default_factory=list, description="Data cleanup concerns")
    rationale: str


class PreconditionsResponse(BaseModel):
    """Response for preconditions evaluation."""
    preconditions_score: float = Field(..., ge=0.0, le=1.0, description="0-1 score for precondition clarity")
    are_explicit: bool = Field(..., description="Are initial states explicitly defined?")
    missing_preconditions: List[str] = Field(default_factory=list, description="Missing state definitions (Power, Network, DB, etc.)")
    rationale: str


class PostconditionsResponse(BaseModel):
    """Response for postconditions evaluation."""
    postconditions_score: float = Field(..., ge=0.0, le=1.0, description="0-1 score for postcondition clarity")
    returns_to_safe_state: bool = Field(..., description="Does test return system to safe/neutral state?")
    cleanup_steps: List[str] = Field(default_factory=list, description="Cleanup steps identified")
    rationale: str


# Coverage & Technique
class TechniqueApplicationResponse(BaseModel):
    """Response for test technique application evaluation."""
    technique_application_score: float = Field(..., ge=0.0, le=1.0, description="0-1 score for technique appropriateness")
    technique_used: Optional[str] = Field(None, description="Technique identified (EP, BVA, Decision Table, etc.)")
    is_appropriate: bool = Field(..., description="Is the technique appropriate for this requirement?")
    recommendations: str = Field("", description="Suggested techniques or improvements")
    rationale: str


class NegativeTestingResponse(BaseModel):
    """Response for negative testing evaluation."""
    negative_testing_score: float = Field(..., ge=0.0, le=1.0, description="0-1 score for negative test coverage")
    has_negative_tests: bool = Field(..., description="Are there tests for invalid inputs, timeouts, errors?")
    missing_negative_cases: List[str] = Field(default_factory=list, description="Missing negative test scenarios")
    rationale: str


class BoundaryChecksResponse(BaseModel):
    """Response for boundary value analysis evaluation."""
    boundary_checks_score: float = Field(..., ge=0.0, le=1.0, description="0-1 score for boundary testing")
    has_boundary_tests: bool = Field(..., description="Are edges (Min, Min-1, Max, Max+1) verified?")
    missing_boundaries: List[str] = Field(default_factory=list, description="Missing boundary conditions")
    rationale: str


class RiskVerificationResponse(BaseModel):
    """Response for risk-based testing evaluation."""
    risk_verification_score: float = Field(..., ge=0.0, le=1.0, description="0-1 score for risk control verification")
    is_linked_to_risk: bool = Field(..., description="Is test linked to a Risk ID?")
    verifies_control_effectiveness: bool = Field(False, description="Does it verify control effectiveness (not just flag)?")
    verification_gaps: List[str] = Field(default_factory=list, description="Gaps in risk mitigation verification")
    rationale: str


# Traceability & Compliance
class TraceabilityResponse(BaseModel):
    """Response for requirement traceability evaluation."""
    traceability_score: float = Field(..., ge=0.0, le=1.0, description="0-1 score for traceability correctness")
    has_correct_link: bool = Field(..., description="Is there a correct link to Requirement ID or Risk ID?")
    linked_ids: List[str] = Field(default_factory=list, description="IDs found in test case")
    rationale: str


class SafetyClassRigorResponse(BaseModel):
    """Response for safety class rigor evaluation."""
    safety_class_rigor_score: float = Field(..., ge=0.0, le=1.0, description="0-1 score for safety class compliance")
    safety_class: Optional[str] = Field(None, description="Safety class (A, B, C) if identified")
    has_unit_test: bool = Field(False, description="For Class C, is there an associated unit test?")
    has_mcdc_coverage: bool = Field(False, description="For Class C, is there MC/DC coverage?")
    compliance_issues: List[str] = Field(default_factory=list, description="Safety class compliance gaps")
    rationale: str


class ObjectiveEvidenceResponse(BaseModel):
    """Response for objective evidence evaluation."""
    objective_evidence_score: float = Field(..., ge=0.0, le=1.0, description="0-1 score for evidence objectivity")
    has_specific_values: bool = Field(..., description="Does expected result define specific values?")
    subjective_statements: List[str] = Field(default_factory=list, description="Subjective/vague statements found")
    improvement_suggestions: str = Field("", description="How to make evidence more objective")
    rationale: str


class AggregatorResponse(BaseModel):
    """Response for aggregating all evaluator outputs into final review."""
    review_summary: str = Field(..., description="Comprehensive paragraph summarizing the review based on all evaluator findings")
    test_case_improvements: str = Field(..., description="Specific, actionable improvements recommended for the test case")
