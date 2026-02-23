"""
Node implementations for Ollama test case review agent.

Provides result aggregation and parsing functionality.
"""

import json
import logging
from typing import Dict, Any
from .core import (
    StructureEvaluation,
    ObjectiveEvaluation,
    ReviewSummary,
    ReviewResult
)


class ResultAggregatorNode:
    """
    Aggregates evaluation results from parallel nodes into final ReviewResult.

    Parses JSON strings from structure, objective, and summary nodes.
    """

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse evaluation results and create final ReviewResult.

        Args:
            state: Current graph state containing structure, objective, review_summary

        Returns:
            Updated state with final_result
        """
        test_id = state.get("test_case_id", "unknown")
        structure_json = state.get("structure", "{}")
        objective_json = state.get("objective", "{}")
        summary_json = state.get("review_summary", "{}")

        # Parse JSON strings into Pydantic models
        try:
            structure_eval = StructureEvaluation.model_validate_json(structure_json)
        except Exception as e:
            logging.error(f"Failed to parse structure evaluation: {e}")
            structure_eval = StructureEvaluation(
                assessment_verdict="inadequate",
                assessment_rationale="Parsing failed",
                identified_gaps=["JSON parsing error"],
                recommendations=[],
                test_case_improvements=[]
            )

        try:
            objective_eval = ObjectiveEvaluation.model_validate_json(objective_json)
        except Exception as e:
            logging.error(f"Failed to parse objective evaluation: {e}")
            objective_eval = ObjectiveEvaluation(
                assessment_verdict="inadequate",
                assessment_rationale="Parsing failed",
                identified_gaps=["JSON parsing error"],
                recommendations=[],
                test_case_improvements=[]
            )

        try:
            summary = ReviewSummary.model_validate_json(summary_json)
        except Exception as e:
            logging.error(f"Failed to parse review summary: {e}")
            summary = ReviewSummary(
                testcase_review_summary="Summary generation failed"
            )

        # Create final result
        final_result = ReviewResult.from_evaluations(
            test_id=test_id,
            structure_eval=structure_eval,
            objective_eval=objective_eval,
            summary=summary
        )

        logging.debug(f"Aggregated review result for {test_id}: score={final_result.overall_score:.2f}")

        return {"final_result": final_result}


def create_aggregator_node() -> ResultAggregatorNode:
    """Factory function to create result aggregator node."""
    return ResultAggregatorNode()
