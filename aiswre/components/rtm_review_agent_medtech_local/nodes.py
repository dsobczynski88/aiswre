"""
Node implementations for RTM review agent (Ollama/Local version).

Provides helper functions to convert evaluator responses to RTMReviewLink
and an Ollama-powered aggregator node for requirement verification coverage analysis.
"""

import json
import logging
from typing import Any, Dict, Optional, List

try:
    from langchain_ollama import ChatOllama
except ImportError:
    from langchain_community.chat_models import ChatOllama

from .core import (
    Requirement, RTMEntry, RTMReviewLink,
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


def convert_evaluator_responses_to_rtm_links(
    state: dict
) -> List[RTMReviewLink]:
    """
    Convert raw evaluator response dictionaries to RTMReviewLink objects.

    Args:
        state: Current graph state with requirement, RTM entry, and raw evaluator responses

    Returns:
        List of RTMReviewLink objects with appropriate scores and issues
    """
    requirement = state.get('requirement')
    rtm_entry = state.get('rtm_entry')
    raw_links = state.get('raw_evaluator_responses', [])

    review_links = []

    for raw_link in raw_links:
        link_type = raw_link.get("type")
        resp = raw_link.get("data")

        rl = RTMReviewLink(
            req_id=requirement.req_id,
            link_type=link_type,
            rationale=getattr(resp, "rationale", "")
        )

        # Map response fields to review link based on type
        if isinstance(resp, FunctionalCoverageResponse):
            rl.functional_coverage_score = resp.functional_coverage_score
            if not resp.all_functions_tested:
                rl.issues.extend(resp.missing_functions)

        elif isinstance(resp, InputOutputCoverageResponse):
            rl.input_output_coverage_score = resp.input_output_coverage_score
            if not resp.all_inputs_tested:
                rl.issues.extend([f"Missing input: {i}" for i in resp.missing_inputs])
            if not resp.all_outputs_verified:
                rl.issues.extend([f"Missing output: {o}" for o in resp.missing_outputs])

        elif isinstance(resp, BoundaryCoverageResponse):
            rl.boundary_coverage_score = resp.boundary_coverage_score
            if not resp.has_boundary_tests:
                rl.issues.extend(resp.missing_boundaries)

        elif isinstance(resp, NegativeTestCoverageResponse):
            rl.negative_test_coverage_score = resp.negative_test_coverage_score
            if not resp.has_negative_tests:
                rl.issues.extend(resp.missing_negative_cases)

        elif isinstance(resp, RiskCoverageResponse):
            rl.risk_coverage_score = resp.risk_coverage_score
            if resp.is_linked_to_risk and not resp.risk_controls_verified:
                rl.issues.extend(resp.verification_gaps)

        elif isinstance(resp, TraceabilityCompletenessResponse):
            rl.traceability_completeness_score = resp.traceability_completeness_score
            if not resp.has_clear_mapping:
                rl.issues.extend(resp.traceability_issues)

        elif isinstance(resp, AcceptanceCriteriaCoverageResponse):
            rl.acceptance_criteria_coverage_score = resp.acceptance_criteria_coverage_score
            if resp.has_acceptance_criteria and not resp.all_criteria_verified:
                rl.issues.extend(resp.missing_criteria)

        elif isinstance(resp, TestSufficiencyResponse):
            rl.test_sufficiency_score = resp.test_sufficiency_score
            if not resp.has_sufficient_tests:
                rl.issues.extend(resp.sufficiency_concerns)

        elif isinstance(resp, GapAnalysisResponse):
            rl.gap_analysis_score = resp.gap_analysis_score
            rl.issues.extend([f"CRITICAL: {g}" for g in resp.critical_gaps])
            rl.issues.extend([f"MODERATE: {g}" for g in resp.moderate_gaps])

        review_links.append(rl)

    return review_links


class RTMLocalAggregatorNode:
    """
    Aggregates RTM review links from parallel evaluators using Ollama LLM analysis.

    Combines 9 evaluation scores using weighted averaging, then uses a local Ollama model
    to analyze all evaluator outputs and generate a comprehensive review summary,
    verification gaps, and recommendations.
    """

    def __init__(
        self,
        model: str = "llama3.1",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.0,
        weights: Optional[dict] = None
    ):
        """
        Initialize the aggregator node.

        Args:
            model: Ollama model name
            base_url: Ollama server URL
            temperature: LLM temperature
            weights: Optional dict of score weights for aggregation
        """
        self.llm = ChatOllama(model=model, base_url=base_url, temperature=temperature)

        # Default weights (emphasizing critical coverage areas)
        self.weights = weights or {
            # Coverage Evaluators (50% total)
            "functional_coverage_score": 0.15,
            "input_output_coverage_score": 0.15,
            "boundary_coverage_score": 0.10,
            "negative_test_coverage_score": 0.10,

            # Risk & Traceability (30% total)
            "risk_coverage_score": 0.10,
            "traceability_completeness_score": 0.10,
            "acceptance_criteria_coverage_score": 0.10,

            # Sufficiency & Gaps (20% total)
            "test_sufficiency_score": 0.10,
            "gap_analysis_score": 0.10,
        }

        # Create structured output LLM for aggregation
        self.structured_llm = self.llm.with_structured_output(AggregatorResponse)

    def __call__(self, state: dict) -> dict:
        """Aggregate RTM review links from state and return final result."""
        links = state.get('rtm_links', [])

        if not links:
            raise ValueError("No RTM review links to aggregate")

        # Weighted sum / normalization
        total_w = sum(self.weights.values())

        def weighted_avg(field):
            return round(
                sum(getattr(t, field, 0.0) * self.weights.get(field, 0.0) for t in links) / total_w if total_w > 0 else 0.0,
                3
            )

        # Collect all issues
        all_issues = []
        for link in links:
            all_issues.extend(link.issues)

        # Build evaluator summaries for LLM
        evaluator_summaries = []

        for link in links:
            summary = f"**{link.link_type} Evaluator**\n"

            # Extract the relevant score
            score_fields = [
                "functional_coverage_score", "input_output_coverage_score",
                "boundary_coverage_score", "negative_test_coverage_score",
                "risk_coverage_score", "traceability_completeness_score",
                "acceptance_criteria_coverage_score", "test_sufficiency_score",
                "gap_analysis_score"
            ]

            for field in score_fields:
                val = getattr(link, field, 0.0)
                if val > 0:
                    summary += f"  Score: {val:.2f}\n"
                    break

            summary += f"  Rationale: {link.rationale}\n"
            if link.issues:
                summary += f"  Issues: {', '.join(link.issues[:3])}"  # First 3 issues
                if len(link.issues) > 3:
                    summary += f" (+ {len(link.issues) - 3} more)"
                summary += "\n"

            evaluator_summaries.append(summary)

        # Build prompt for LLM aggregation
        aggregation_prompt = self._build_aggregation_prompt(evaluator_summaries, state)

        # Call LLM to generate summary, gaps, and recommendations
        try:
            from langchain_core.messages import SystemMessage, HumanMessage
            messages = [
                SystemMessage(content=aggregation_prompt["system"]),
                HumanMessage(content=aggregation_prompt["user"])
            ]
            agg_response = self.structured_llm.invoke(messages)
            review_summary = agg_response.review_summary
            verification_gaps = agg_response.verification_gaps
            recommendations = agg_response.recommendations
        except Exception as e:
            logging.error(f"Ollama aggregation failed: {e}")
            review_summary = "Aggregation summary unavailable due to LLM error."
            verification_gaps = "Unable to analyze verification gaps."
            recommendations = "Unable to generate recommendations."

        # Create aggregated link
        combined = RTMReviewLink(
            req_id=links[0].req_id,

            # Coverage Evaluators
            functional_coverage_score=weighted_avg("functional_coverage_score"),
            input_output_coverage_score=weighted_avg("input_output_coverage_score"),
            boundary_coverage_score=weighted_avg("boundary_coverage_score"),
            negative_test_coverage_score=weighted_avg("negative_test_coverage_score"),

            # Risk & Traceability
            risk_coverage_score=weighted_avg("risk_coverage_score"),
            traceability_completeness_score=weighted_avg("traceability_completeness_score"),
            acceptance_criteria_coverage_score=weighted_avg("acceptance_criteria_coverage_score"),

            # Sufficiency & Gaps
            test_sufficiency_score=weighted_avg("test_sufficiency_score"),
            gap_analysis_score=weighted_avg("gap_analysis_score"),

            rationale=" | ".join(t.rationale for t in links if t.rationale),
            link_type="RTMLocalAggregated",
            issues=all_issues,

            # LLM-generated fields
            review_summary=review_summary,
            verification_gaps=verification_gaps,
            recommendations=recommendations
        )

        logging.debug(f"Aggregated RTM local link: {combined}")

        return {"final_result": combined}

    def _build_aggregation_prompt(self, evaluator_summaries: List[str], state: dict) -> dict:
        """
        Build the prompt for LLM aggregation.

        Args:
            evaluator_summaries: List of formatted evaluator output summaries
            state: Current state containing requirement and RTM entry info

        Returns:
            Dict with 'system' and 'user' prompt content
        """
        requirement = state.get('requirement')
        rtm_entry = state.get('rtm_entry')

        num_evaluators = len(evaluator_summaries)
        answers = "\n\n".join(evaluator_summaries)

        system_prompt = f"""Act as a medical device software verification expert and FDA/IEC 62304 compliance specialist.

Your goal is to analyze verification coverage findings from {num_evaluators} specialized evaluators
and determine how well the test cases verify the requirement.

You will receive evaluation results from {num_evaluators} specialized evaluators covering:
- Coverage (Functional, Input/Output, Boundary, Negative Testing)
- Risk & Traceability (Risk Coverage, Traceability Completeness, Acceptance Criteria)
- Sufficiency & Gaps (Test Sufficiency, Gap Analysis)

Your job is to:
1. Synthesize all findings into a coherent verification coverage summary
2. Identify specific verification gaps (what is NOT verified)
3. Provide actionable recommendations to close gaps and improve coverage

Focus on FDA/IEC 62304 verification requirements and medical device software testing best practices."""

        user_prompt = f"""Here are the evaluation results from {num_evaluators} different evaluator agents:

{answers}

**Requirement Information:**
- Requirement ID: {requirement.req_id}
- Requirement Text: {requirement.text}
- Risk ID: {requirement.risk_id if requirement.risk_id else 'N/A'}
- Safety Class: {requirement.safety_class if requirement.safety_class else 'N/A'}

**Test Case Summary:**
{rtm_entry.test_case_summary}

**Test Case Count:** {rtm_entry.test_case_count if rtm_entry.test_case_count else 'Unknown'}

Based on all the evaluator outputs above, provide:
1. A comprehensive verification coverage summary (paragraph format)
2. Specific verification gaps (what aspects of the requirement are NOT adequately verified)
3. Actionable recommendations to improve verification coverage and close gaps"""

        return {"system": system_prompt, "user": user_prompt}


def create_aggregator_node(
    model: str = "llama3.1",
    base_url: str = "http://localhost:11434",
    temperature: float = 0.0,
    weights: Optional[dict] = None
) -> RTMLocalAggregatorNode:
    """Factory function to create Ollama-powered RTM aggregator node."""
    return RTMLocalAggregatorNode(
        model=model,
        base_url=base_url,
        temperature=temperature,
        weights=weights
    )
