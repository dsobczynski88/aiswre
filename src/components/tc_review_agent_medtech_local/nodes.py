"""
Node implementations for medtech test case review agent (Ollama/Local version).

Provides helper functions to convert evaluator responses to MedtechTraceLink
and an Ollama-powered aggregator node.
"""

import json
import logging
from typing import Any, Dict, Optional, List

try:
    from langchain_ollama import ChatOllama
except ImportError:
    from langchain_community.chat_models import ChatOllama

from .core import (
    Requirement, TestCase, MedtechTraceLink,
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
    ObjectiveEvidenceResponse,
    AggregatorResponse
)


def convert_evaluator_responses_to_tracelinks(
    state: dict
) -> List[MedtechTraceLink]:
    """
    Convert raw evaluator response dictionaries to MedtechTraceLink objects.

    Args:
        state: Current graph state with test, requirement, and raw evaluator responses

    Returns:
        List of MedtechTraceLink objects with appropriate scores and issues
    """
    requirement = state.get('requirement')
    test = state.get('test')
    raw_links = state.get('raw_evaluator_responses', [])

    trace_links = []

    for raw_link in raw_links:
        link_type = raw_link.get("type")
        resp = raw_link.get("data")

        tl = MedtechTraceLink(
            req_id=requirement.req_id if requirement else "unknown",
            test_id=test.test_id,
            link_type=link_type,
            rationale=getattr(resp, "rationale", "")
        )

        # Map response fields to trace link based on type
        if isinstance(resp, UnambiguityResponse):
            tl.unambiguity_score = resp.unambiguity_score
            if not resp.is_unambiguous:
                tl.issues.extend(resp.ambiguous_phrases)

        elif isinstance(resp, IndependenceResponse):
            tl.independence_score = resp.independence_score
            if not resp.is_independent:
                tl.issues.extend(resp.dependencies + resp.cleanup_issues)

        elif isinstance(resp, PreconditionsResponse):
            tl.preconditions_score = resp.preconditions_score
            if not resp.are_explicit:
                tl.issues.extend(resp.missing_preconditions)

        elif isinstance(resp, PostconditionsResponse):
            tl.postconditions_score = resp.postconditions_score
            if not resp.returns_to_safe_state:
                tl.issues.append("Does not return to safe state")

        elif isinstance(resp, TechniqueApplicationResponse):
            tl.technique_application_score = resp.technique_application_score
            if not resp.is_appropriate:
                tl.issues.append(f"Technique application issue: {resp.recommendations}")

        elif isinstance(resp, NegativeTestingResponse):
            tl.negative_testing_score = resp.negative_testing_score
            if not resp.has_negative_tests:
                tl.issues.extend(resp.missing_negative_cases)

        elif isinstance(resp, BoundaryChecksResponse):
            tl.boundary_checks_score = resp.boundary_checks_score
            if not resp.has_boundary_tests:
                tl.issues.extend(resp.missing_boundaries)

        elif isinstance(resp, RiskVerificationResponse):
            tl.risk_verification_score = resp.risk_verification_score
            if resp.is_linked_to_risk and not resp.verifies_control_effectiveness:
                tl.issues.extend(resp.verification_gaps)

        elif isinstance(resp, TraceabilityResponse):
            tl.traceability_score = resp.traceability_score
            if not resp.has_correct_link:
                tl.issues.append("Missing or incorrect traceability link")

        elif isinstance(resp, SafetyClassRigorResponse):
            tl.safety_class_rigor_score = resp.safety_class_rigor_score
            tl.issues.extend(resp.compliance_issues)

        elif isinstance(resp, ObjectiveEvidenceResponse):
            tl.objective_evidence_score = resp.objective_evidence_score
            if not resp.has_specific_values:
                tl.issues.extend(resp.subjective_statements)

        trace_links.append(tl)

    return trace_links


class MedtechLocalAggregatorNode:
    """
    Aggregates medtech trace links from parallel evaluators using Ollama LLM analysis.

    Combines 11 evaluation scores using weighted averaging, then uses a local Ollama model
    to analyze all evaluator outputs and generate a comprehensive review summary
    and improvement recommendations.
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

        # Default weights (can be customized based on regulatory priorities)
        self.weights = weights or {
            # General Integrity & Structure (25% total)
            "unambiguity_score": 0.10,
            "independence_score": 0.05,
            "preconditions_score": 0.05,
            "postconditions_score": 0.05,

            # Coverage & Technique (40% total)
            "technique_application_score": 0.10,
            "negative_testing_score": 0.10,
            "boundary_checks_score": 0.10,
            "risk_verification_score": 0.10,

            # Traceability & Compliance (35% total)
            "traceability_score": 0.15,
            "safety_class_rigor_score": 0.10,
            "objective_evidence_score": 0.10,
        }

        # Create structured output LLM for aggregation
        self.structured_llm = self.llm.with_structured_output(AggregatorResponse)

    def __call__(self, state: dict) -> dict:
        """Aggregate medtech trace links from state and return final result."""
        links = state.get('medtech_links', [])

        if not links:
            raise ValueError("No medtech trace links to aggregate")

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
                "unambiguity_score", "independence_score", "preconditions_score",
                "postconditions_score", "technique_application_score", "negative_testing_score",
                "boundary_checks_score", "risk_verification_score", "traceability_score",
                "safety_class_rigor_score", "objective_evidence_score"
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

        # Call LLM to generate summary and improvements
        try:
            from langchain_core.messages import SystemMessage, HumanMessage
            messages = [
                SystemMessage(content=aggregation_prompt["system"]),
                HumanMessage(content=aggregation_prompt["user"])
            ]
            agg_response = self.structured_llm.invoke(messages)
            review_summary = agg_response.review_summary
            test_case_improvements = agg_response.test_case_improvements
        except Exception as e:
            logging.error(f"Ollama aggregation failed: {e}")
            review_summary = "Aggregation summary unavailable due to LLM error."
            test_case_improvements = "Unable to generate improvement recommendations."

        # Create aggregated link
        combined = MedtechTraceLink(
            req_id=links[0].req_id,
            test_id=links[0].test_id,

            # General Integrity & Structure
            unambiguity_score=weighted_avg("unambiguity_score"),
            independence_score=weighted_avg("independence_score"),
            preconditions_score=weighted_avg("preconditions_score"),
            postconditions_score=weighted_avg("postconditions_score"),

            # Coverage & Technique
            technique_application_score=weighted_avg("technique_application_score"),
            negative_testing_score=weighted_avg("negative_testing_score"),
            boundary_checks_score=weighted_avg("boundary_checks_score"),
            risk_verification_score=weighted_avg("risk_verification_score"),

            # Traceability & Compliance
            traceability_score=weighted_avg("traceability_score"),
            safety_class_rigor_score=weighted_avg("safety_class_rigor_score"),
            objective_evidence_score=weighted_avg("objective_evidence_score"),

            rationale=" | ".join(t.rationale for t in links if t.rationale),
            link_type="MedtechLocalAggregated",
            issues=all_issues,

            # LLM-generated fields
            review_summary=review_summary,
            test_case_improvements=test_case_improvements
        )

        logging.debug(f"Aggregated medtech local link: {combined}")

        return {"final_result": combined}

    def _build_aggregation_prompt(self, evaluator_summaries: List[str], state: dict) -> dict:
        """
        Build the prompt for LLM aggregation.

        Args:
            evaluator_summaries: List of formatted evaluator output summaries
            state: Current state containing test case and requirement info

        Returns:
            Dict with 'system' and 'user' prompt content
        """
        test = state.get('test')
        requirement = state.get('requirement')

        num_evaluators = len(evaluator_summaries)
        answers = "\n\n".join(evaluator_summaries)

        system_prompt = f"""Act as a medical device software test quality expert and FDA/IEC 62304 compliance specialist.

Your goal is to analyze the issues and rationales previously output by {num_evaluators} different evaluator agents.
Then determine the right way for the test case to move forward.

It's critical that you provide a helpful, comprehensive analysis about this test case and any necessary improvements to be made.

You will receive evaluation results from {num_evaluators} specialized evaluators covering:
- General Integrity & Structure (Unambiguity, Independence, Preconditions, Postconditions)
- Coverage & Technique (Technique Application, Negative Testing, Boundary Checks, Risk Verification)
- Traceability & Compliance (Traceability, Safety Class Rigor, Objective Evidence)

Your job is to:
1. Synthesize all evaluator findings into a coherent review summary
2. Identify the most critical issues that need addressing
3. Provide specific, actionable improvements for the test case

Focus on FDA/IEC 62304 compliance requirements and best practices for medical device software testing."""

        user_prompt = f"""Here are the evaluation results from {num_evaluators} different evaluator agents:

{answers}

**Test Case Information:**
- Test ID: {test.test_id if test else 'N/A'}
- Description: {test.description if test else 'N/A'}

"""

        if requirement:
            user_prompt += f"""**Requirement Information:**
- Requirement ID: {requirement.req_id if requirement.req_id else 'N/A'}
- Requirement Text: {requirement.text}
- Risk ID: {requirement.risk_id if requirement.risk_id else 'N/A'}
- Safety Class: {requirement.safety_class if requirement.safety_class else 'N/A'}

"""

        user_prompt += """Based on all the evaluator outputs above, provide:
1. A comprehensive review summary (paragraph format) that synthesizes the key findings
2. Specific, actionable improvements to enhance this test case's quality and compliance"""

        return {"system": system_prompt, "user": user_prompt}


def create_aggregator_node(
    model: str = "llama3.1",
    base_url: str = "http://localhost:11434",
    temperature: float = 0.0,
    weights: Optional[dict] = None
) -> MedtechLocalAggregatorNode:
    """Factory function to create Ollama-powered aggregator node."""
    return MedtechLocalAggregatorNode(
        model=model,
        base_url=base_url,
        temperature=temperature,
        weights=weights
    )
