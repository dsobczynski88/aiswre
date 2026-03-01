"""
Core data models for RTM review agent.

Defines Pydantic models for test cases, requirements, and evaluation responses
aligned with FDA/IEC 62304 testing best practices.
"""

from pydantic import BaseModel, Field
import operator
from typing import Optional, List, TypedDict, Annotated, Literal


class EdgeCaseAnalysis(BaseModel):
    potential_edge_cases: List[str]
    risk_of_escaped_defect: str
    recommended_mitigation: str


class Requirement(BaseModel):
    """Software requirement model."""
    req_id: Optional[str] = None
    text: str
    #risk_id: Optional[str] = None  # For risk-based testing
    #safety_class: Optional[str] = None  # A, B, or C per IEC 62304

class DecomposedSpec(BaseModel):
    spec_id: str
    type: str
    description: str
    acceptance_criteria: str
    rationale: str

class DecomposedRequirement(BaseModel):
    requirement: Requirement
    decomposed_specifications: List[DecomposedSpec]

class TestCase(BaseModel):
    test_id: str
    description: str
    setup: Optional[str] = None
    steps: Optional[str] = None
    expectedResults: Optional[str] = None
    #postconditions: Optional[str] = None
    #test_type: Optional[str] = None  # e.g., "Unit", "Integration", "System"
    #technique: Optional[str] = None  # e.g., "EP", "BVA", "Decision Table"

class SummarizedTestCase(BaseModel):
    test_case_id: str
    objective: str
    verifies: str
    protocol: List[str]
    acceptance_criteria: List[str]

class TestSuite(BaseModel):
    requirement: Requirement
    test_cases: List[TestCase]
    summary: List[SummarizedTestCase]

#class MissingBoundary(BaseModel):
#    summarized_test_case: SummarizedTestCase
#    gap_description: str
#    escaped_defect_risk: Literal["High", "Medium", "Low"]
#    rationale: str
    
#class CoveredBoundary(BaseModel):
#    spec_id: str
#    edge_case_summary: str
#    mapped_test_case_id: str
#    coverage_rationale: str

class EvaluatedSpec(BaseModel):
    """Per-spec coverage verdict from an evaluator node."""
    spec_id: str = Field(..., description="The spec_id from the DecomposedSpec")
    covered_exists: bool = Field(..., description="True if coverage exists in at least one test case of input TestSuite otherwise False")
    covered_by_test_cases: List[str] = Field(..., description="A list of test case IDs from TestSuite['summary'] that effectively cover the test. In the event no test cases are covered, this should return as an empty list.")
    rationale: str = Field(..., description="Thought process behind the determination of whether the existing test cases within TestSuite cover or fail to cover the described DecomposedSpec")

class CoverageEvaluator(BaseModel):
    """Container returned by each evaluator node — one EvaluatedEdgeSpec per decomposed spec."""
    evaluations: List[EvaluatedSpec]

class ReviewComment(BaseModel):
    comment: str
    rationale: str
    question: str
    topic: str

class AITestSuite(BaseModel):
    spec_id: str = Field(..., description="The spec_id from the DecomposedSpec")
    current_test_suite: List[SummarizedTestCase]
    generated_tests: List[SummarizedTestCase]
    ai_test_suite: List[SummarizedTestCase] = Field(..., description="The final test suite consisting of the original TestSuite and the newly generated tests")
    rationale: str = Field(..., description="The reasoning as to why this test was generated given the input requirement and current test suite")

class RTMReviewState(TypedDict, total=False):
    requirement: Requirement
    test_cases: List[TestCase]
    decomposed_requirement: DecomposedRequirement
    test_suite: TestSuite
    ai_test_suite: AITestSuite
    coverage_responses: Annotated[List[CoverageEvaluator], operator.add]
    #aggregated_review: Annotated[List['ReviewComment'], operator.add]