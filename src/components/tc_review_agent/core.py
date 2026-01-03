from pydantic import BaseModel
from typing import Optional, List, TypedDict, Annotated
import operator

class TestCaseState(TypedDict, total=False):
    """State for LangGraph test case review pipeline."""
    requirement: Optional['Requirement']
    test: 'TestCase'
    # Use Annotated with operator.add to allow concurrent updates from parallel nodes
    trace_links: Annotated[List['TraceLink'], operator.add]
    final_result: Optional['TraceLink']  # Final aggregated result

class Requirement(BaseModel):
    req_id: Optional[str] = None
    text: str

class TestCase(BaseModel):
    test_id: str
    description: str

class TraceLink(BaseModel):
    req_id: str
    test_id: str
    confidence_score: float = 0.0
    adequacy_score: float = 0.0
    clarity_score: float = 0.0
    rationale: str = ""
    link_type: str = "Suggested"
    reviewer_status: str = "Pending"

class CoverageReport(BaseModel):
    requirement_id: str
    linked_tests: List[TraceLink]
    summary: str

# Response models for LLM JSON
class TraceabilityResponse(BaseModel):
    confidence_score: float
    rationale: str

class AdequacyResponse(BaseModel):
    adequacy_score: float
    missing_conditions: Optional[List[str]] = []
    rationale: str

class ClarityResponse(BaseModel):
    clarity_score: float
    rewrite_suggestions: Optional[str] = ""
    rationale: str