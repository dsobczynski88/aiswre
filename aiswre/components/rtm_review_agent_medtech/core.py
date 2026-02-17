"""
Core data models for RTM review agent.

Defines Pydantic models for test cases, requirements, and evaluation responses
aligned with FDA/IEC 62304 testing best practices.
"""

from pydantic import BaseModel, Field
import operator
from typing import Optional, List, TypedDict, Annotated


class Requirement(BaseModel):
    """Software requirement model."""
    req_id: Optional[str] = None
    text: str
    #risk_id: Optional[str] = None  # For risk-based testing
    #safety_class: Optional[str] = None  # A, B, or C per IEC 62304


class EdgeCaseAnalysis(BaseModel):
    potential_edge_cases: List[str]
    risk_of_escaped_defect: str
    recommended_mitigation: str


class DecomposedSpec(BaseModel):
    spec_id: str
    type: str
    description: str
    verification_method: str
    acceptance_criteria: str
    rationale: str
    edge_case_analysis: EdgeCaseAnalysis


class DecomposedRequirement(BaseModel):
    requirement: Requirement
    decomposed_specifications: List[DecomposedSpec]


class TestCase(BaseModel):
    """Test case model with medtech-specific fields."""
    test_id: str
    description: str
    setup: Optional[str] = None
    steps: Optional[str] = None
    expectedResults: Optional[str] = None
    #postconditions: Optional[str] = None
    #test_type: Optional[str] = None  # e.g., "Unit", "Integration", "System"
    #technique: Optional[str] = None  # e.g., "EP", "BVA", "Decision Table"


class SummarizedTestCase(BaseModel):
    test_id: str
    objective: str
    type: str
    steps: List[str]
    expectedResults: List[str]


class TestSuite(BaseModel):
    requirement: Requirement
    test_cases: List[TestCase]
    summary: List[SummarizedTestCase]


class CoverageEvaluator(BaseModel):
    """Response for functional coverage evaluation."""
    covered: bool = Field(..., description="Are all functional aspects of the requirement verified?")
    missing: List[str] = Field(default_factory=list, description="Functional aspects not covered by test suite")
    rationale: str = Field(..., description="Thought process behind the determination of what was covered and what was missing")
    #functional_coverage_score: float = Field(..., ge=0.0, le=1.0, description="0-1 score for functional requirement coverage")


class ReviewComment(BaseModel):
    comment: str
    rationale: str
    question: str
    topic: str


class RTMReviewState(TypedDict, total=False):
    requirement: Requirement
    test_cases: List[TestCase]
    decomposed_requirement: DecomposedRequirement
    test_suite: TestSuite
    coverage_responses: Annotated[List['CoverageEvaluator'], operator.add]
    aggregated_review: Annotated[List['ReviewComment'], operator.add]


