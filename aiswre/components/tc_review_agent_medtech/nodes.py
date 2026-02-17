"""
Node implementations for medtech test case review agent.

Provides BaseEvaluatorNode for LLM-powered evaluation and AggregatorNode
for combining multiple evaluation scores.
"""

import json
import logging
from typing import Any, Dict, Type, TypeVar, Optional, List
from langchain_openai import ChatOpenAI
from pydantic import ValidationError
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

R = TypeVar("R", bound=Any)


class MedtechEvaluatorNode:
    """
    Generic evaluator node for medtech test case review.

    Uses LangChain's structured output to return Pydantic models directly.
    Each instance is configured with a specific response model and prompt.
    """

    def __init__(
        self,
        llm: ChatOpenAI,
        response_model: Type[R],
        system_prompt: str,
        *,
        link_type: str
    ):
        self.llm = llm
        self.response_model = response_model
        self.system_prompt = system_prompt
        self.link_type = link_type

        # Create a structured output LLM that returns the response model directly
        self.structured_llm = llm.with_structured_output(response_model)

    async def __call__(self, state: dict) -> dict:
        """Process state and return updated state with new medtech trace link."""
        requirement = state.get('requirement')
        test = state.get('test')

        if not test:
            raise ValueError("TestCase is required in state")

        # Build payload for LLM
        payload = self._build_payload(requirement, test)

        # Call LLM with structured output
        try:
            from langchain_core.messages import SystemMessage, HumanMessage
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=json.dumps(payload))
            ]
            # Use structured output to get Pydantic model directly
            parsed = await self.structured_llm.ainvoke(messages)
            trace_link = self._to_medtech_tracelink(parsed, requirement, test)
        except Exception as e:
            logging.error(f"LLM invocation or parsing failed in {self.link_type}: {e}")
            trace_link = self._fallback_link(requirement, test)

        # Return only the new trace link (not accumulated list)
        # LangGraph will use operator.add to merge with other parallel node outputs
        return {"medtech_links": [trace_link]}

    def _build_payload(self, requirement: Optional[Requirement], test: TestCase) -> dict:
        """Build the payload dict sent to the LLM."""
        payload = {
            "test_id": test.test_id,
            "test_description": test.description,
        }

        # Add optional test fields
        if test.preconditions:
            payload["preconditions"] = test.preconditions
        if test.steps:
            payload["steps"] = test.steps
        if test.expected_result:
            payload["expected_result"] = test.expected_result
        if test.postconditions:
            payload["postconditions"] = test.postconditions
        if test.test_type:
            payload["test_type"] = test.test_type
        if test.technique:
            payload["technique"] = test.technique

        # Add requirement info
        if requirement:
            payload["requirement_id"] = requirement.req_id
            payload["requirement_text"] = requirement.text
            if requirement.risk_id:
                payload["risk_id"] = requirement.risk_id
            if requirement.safety_class:
                payload["safety_class"] = requirement.safety_class
        else:
            payload["requirement"] = "No requirement provided"

        return payload

    def _fallback_link(self, requirement: Optional[Requirement], test: TestCase) -> MedtechTraceLink:
        """Create fallback trace link on LLM error."""
        return MedtechTraceLink(
            req_id=requirement.req_id if requirement else "unknown",
            test_id=test.test_id,
            rationale=f"LLM error in {self.link_type} evaluator",
            link_type=self.link_type,
            issues=[f"LLM invocation failed for {self.link_type}"]
        )

    def _to_medtech_tracelink(
        self,
        resp: R,
        requirement: Optional[Requirement],
        test: TestCase
    ) -> MedtechTraceLink:
        """Convert LLM response to MedtechTraceLink."""
        data = resp.dict()
        tl = MedtechTraceLink(
            req_id=requirement.req_id if requirement else "unknown",
            test_id=test.test_id,
            link_type=self.link_type,
            rationale=data.get("rationale", "")
        )

        # Map score fields based on response type
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

        return tl


class MedtechAggregatorNode:
    """
    Aggregates medtech trace links from parallel evaluators using LLM analysis.

    Combines 11 evaluation scores using weighted averaging, then uses an LLM
    to analyze all evaluator outputs and generate a comprehensive review summary
    and improvement recommendations.
    """

    def __init__(self, llm: ChatOpenAI, weights: Optional[dict] = None):
        """
        Initialize the aggregator node.

        Args:
            llm: ChatOpenAI client for LLM calls
            weights: Optional dict of score weights for aggregation
        """
        self.llm = llm

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
        self.structured_llm = llm.with_structured_output(AggregatorResponse)

    async def __call__(self, state: dict) -> dict:
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

        # Map link types to evaluator names and extract findings
        evaluator_mapping = {
            "Unambiguity": "Unambiguity Evaluator",
            "Independence": "Independence Evaluator",
            "Preconditions": "Preconditions Evaluator",
            "Postconditions": "Postconditions Evaluator",
            "TechniqueApplication": "Technique Application Evaluator",
            "NegativeTesting": "Negative Testing Evaluator",
            "BoundaryChecks": "Boundary Checks Evaluator",
            "RiskVerification": "Risk Verification Evaluator",
            "Traceability": "Traceability Evaluator",
            "SafetyClassRigor": "Safety Class Rigor Evaluator",
            "ObjectiveEvidence": "Objective Evidence Evaluator"
        }

        for link in links:
            evaluator_name = evaluator_mapping.get(link.link_type, link.link_type)

            # Extract score for this evaluator
            score_field = None
            if "unambiguity_score" in link.__dict__ and link.unambiguity_score > 0:
                score_field = ("unambiguity_score", link.unambiguity_score)
            elif "independence_score" in link.__dict__ and link.independence_score > 0:
                score_field = ("independence_score", link.independence_score)
            elif "preconditions_score" in link.__dict__ and link.preconditions_score > 0:
                score_field = ("preconditions_score", link.preconditions_score)
            elif "postconditions_score" in link.__dict__ and link.postconditions_score > 0:
                score_field = ("postconditions_score", link.postconditions_score)
            elif "technique_application_score" in link.__dict__ and link.technique_application_score > 0:
                score_field = ("technique_application_score", link.technique_application_score)
            elif "negative_testing_score" in link.__dict__ and link.negative_testing_score > 0:
                score_field = ("negative_testing_score", link.negative_testing_score)
            elif "boundary_checks_score" in link.__dict__ and link.boundary_checks_score > 0:
                score_field = ("boundary_checks_score", link.boundary_checks_score)
            elif "risk_verification_score" in link.__dict__ and link.risk_verification_score > 0:
                score_field = ("risk_verification_score", link.risk_verification_score)
            elif "traceability_score" in link.__dict__ and link.traceability_score > 0:
                score_field = ("traceability_score", link.traceability_score)
            elif "safety_class_rigor_score" in link.__dict__ and link.safety_class_rigor_score > 0:
                score_field = ("safety_class_rigor_score", link.safety_class_rigor_score)
            elif "objective_evidence_score" in link.__dict__ and link.objective_evidence_score > 0:
                score_field = ("objective_evidence_score", link.objective_evidence_score)

            summary = f"**{evaluator_name}**\n"
            if score_field:
                summary += f"  Score: {score_field[1]:.2f}\n"
            summary += f"  Rationale: {link.rationale}\n"
            if link.issues:
                summary += f"  Issues: {', '.join(link.issues)}\n"

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
            agg_response = await self.structured_llm.ainvoke(messages)
            review_summary = agg_response.review_summary
            test_case_improvements = agg_response.test_case_improvements
        except Exception as e:
            logging.error(f"LLM aggregation failed: {e}")
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
            link_type="MedtechAggregated",
            issues=all_issues,

            # LLM-generated fields
            review_summary=review_summary,
            test_case_improvements=test_case_improvements
        )

        logging.debug(f"Aggregated medtech link: {combined}")

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
