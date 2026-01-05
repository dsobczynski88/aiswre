"""
Core data models for Ollama-based test case review agent.

Defines Pydantic models and TypedDict state for LangGraph workflow.
"""

from pydantic import BaseModel, Field
from typing import TypedDict, Annotated, List, Optional, Literal
from langgraph.graph import add_messages


class TestCaseReviewState(TypedDict, total=False):
    """
    State for LangGraph test case review pipeline.

    Uses add_messages for accumulating chat history from parallel nodes.
    """
    messages: Annotated[List, add_messages]  # Chat history aggregated via add_messages
    test_case_id: str
    test_case_text: str
    structure: str  # Raw JSON string from eval_structure
    objective: str  # Raw JSON string from eval_objective
    review_summary: str  # Raw JSON string from get_review_summary
    final_result: Optional['ReviewResult']  # Parsed final result


class TestCaseInput(BaseModel):
    """Input test case model."""
    test_id: str
    test_case_text: str
    title: Optional[str] = None
    preconditions: Optional[str] = None
    steps: Optional[str] = None
    expected_result: Optional[str] = None


class StructureEvaluation(BaseModel):
    """
    Response model for structure evaluation node.

    Evaluates test case structure quality against acceptance criteria.
    """
    assessment_verdict: Literal["complete", "partial", "inadequate"] = Field(
        ...,
        description="Overall verdict on structure quality"
    )
    assessment_rationale: str = Field(
        ...,
        description="Explanation of why the verdict was chosen"
    )
    identified_gaps: List[str] = Field(
        default_factory=list,
        description="Gaps found in test case structure"
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Actionable recommendations for improvement"
    )
    test_case_improvements: List[str] = Field(
        default_factory=list,
        description="Specific improvements to apply"
    )


class ObjectiveEvaluation(BaseModel):
    """
    Response model for objective/completeness evaluation node.

    Evaluates test case completeness and alignment with objectives.
    """
    assessment_verdict: Literal["complete", "partial", "inadequate"] = Field(
        ...,
        description="Overall verdict on completeness"
    )
    assessment_rationale: str = Field(
        ...,
        description="Explanation of why the verdict was chosen"
    )
    identified_gaps: List[str] = Field(
        default_factory=list,
        description="Gaps found in test case completeness"
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Actionable recommendations for improvement"
    )
    test_case_improvements: List[str] = Field(
        default_factory=list,
        description="Specific improvements to apply"
    )


class ReviewSummary(BaseModel):
    """Response model for review summary aggregation."""
    testcase_review_summary: str = Field(
        ...,
        description="Comprehensive summary of test case review findings"
    )


class ReviewResult(BaseModel):
    """
    Final aggregated review result.

    Combines structure and objective evaluations with summary.
    """
    test_id: str

    # Structure evaluation
    structure_verdict: str
    structure_gaps: List[str] = Field(default_factory=list)
    structure_recommendations: List[str] = Field(default_factory=list)

    # Objective evaluation
    objective_verdict: str
    objective_gaps: List[str] = Field(default_factory=list)
    objective_recommendations: List[str] = Field(default_factory=list)

    # Summary
    review_summary: str

    # Overall score (derived from verdicts)
    overall_score: float = Field(
        default=0.0,
        description="Composite score: complete=1.0, partial=0.5, inadequate=0.0"
    )

    # Metadata
    link_type: str = "OllamaReview"

    @classmethod
    def from_evaluations(
        cls,
        test_id: str,
        structure_eval: StructureEvaluation,
        objective_eval: ObjectiveEvaluation,
        summary: ReviewSummary
    ) -> 'ReviewResult':
        """Create ReviewResult from component evaluations."""

        # Calculate overall score
        verdict_scores = {"complete": 1.0, "partial": 0.5, "inadequate": 0.0}
        structure_score = verdict_scores.get(structure_eval.assessment_verdict, 0.0)
        objective_score = verdict_scores.get(objective_eval.assessment_verdict, 0.0)
        overall_score = (structure_score + objective_score) / 2.0

        return cls(
            test_id=test_id,
            structure_verdict=structure_eval.assessment_verdict,
            structure_gaps=structure_eval.identified_gaps,
            structure_recommendations=structure_eval.recommendations,
            objective_verdict=objective_eval.assessment_verdict,
            objective_gaps=objective_eval.identified_gaps,
            objective_recommendations=objective_eval.recommendations,
            review_summary=summary.testcase_review_summary,
            overall_score=overall_score
        )
