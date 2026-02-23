"""
Medtech evaluator factory functions.

Creates specialized evaluator nodes for FDA/IEC 62304 test case review.
Each factory returns a BaseEvaluatorNode configured with domain-specific prompts.
"""

from langchain_openai import ChatOpenAI
from .nodes import MedtechEvaluatorNode
from .core import (
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
    ObjectiveEvidenceResponse
)


# ============================================================================
# General Integrity & Structure Evaluators
# ============================================================================

def make_unambiguity_evaluator(llm: ChatOpenAI) -> MedtechEvaluatorNode:
    """
    Evaluates if test instructions are clear enough that a person with no prior
    knowledge could execute them and get the exact same result.
    """
    prompt = (
        "You are a medical device QA specialist evaluating test case clarity.\n\n"
        "TASK: Assess if the test instructions are unambiguous and executable by someone "
        "with no prior knowledge of the system.\n\n"
        "CRITERIA:\n"
        "- Are all steps explicitly defined?\n"
        "- Are there undefined terms or references?\n"
        "- Could two different people execute this identically?\n\n"
        "Return ONLY valid JSON (no markdown, no code blocks) with this structure:\n"
        '{"unambiguity_score": <float 0-1>, "is_unambiguous": <bool>, '
        '"ambiguous_phrases": [<list>], "rationale": "<explanation>"}'
    )
    return MedtechEvaluatorNode(
        llm=llm,
        response_model=UnambiguityResponse,
        system_prompt=prompt,
        link_type="Unambiguity"
    )


def make_independence_evaluator(llm: ChatOpenAI) -> MedtechEvaluatorNode:
    """
    Evaluates if the test is self-contained and cleans up its own data.
    """
    prompt = (
        "You are a test automation expert evaluating test independence.\n\n"
        "TASK: Assess if the test is self-contained and manages its own lifecycle.\n\n"
        "CRITERIA:\n"
        "- Does the test depend on other tests or external state?\n"
        "- Does it create and clean up its own test data?\n"
        "- Can it run in isolation without side effects?\n\n"
        "Return ONLY valid JSON (no markdown, no code blocks) with this structure:\n"
        '{"independence_score": <float 0-1>, "is_independent": <bool>, '
        '"dependencies": [<list>], "cleanup_issues": [<list>], "rationale": "<explanation>"}'
    )
    return MedtechEvaluatorNode(
        llm=llm,
        response_model=IndependenceResponse,
        system_prompt=prompt,
        link_type="Independence"
    )


def make_preconditions_evaluator(llm: ChatOpenAI) -> MedtechEvaluatorNode:
    """
    Evaluates if initial states (Power, Network, Database) are explicitly defined.
    """
    prompt = (
        "You are a verification engineer evaluating test preconditions.\n\n"
        "TASK: Assess if the test explicitly defines all initial states required.\n\n"
        "CRITERIA:\n"
        "- Are system states (Power, Network, Database, Configuration) explicitly defined?\n"
        "- Are environmental conditions specified?\n"
        "- Is the starting state reproducible?\n\n"
        "Return ONLY valid JSON (no markdown, no code blocks) with this structure:\n"
        '{"preconditions_score": <float 0-1>, "are_explicit": <bool>, '
        '"missing_preconditions": [<list>], "rationale": "<explanation>"}'
    )
    return MedtechEvaluatorNode(
        llm=llm,
        response_model=PreconditionsResponse,
        system_prompt=prompt,
        link_type="Preconditions"
    )


def make_postconditions_evaluator(llm: ChatOpenAI) -> MedtechEvaluatorNode:
    """
    Evaluates if the test returns the system to a safe/neutral state.
    """
    prompt = (
        "You are a safety-critical systems QA evaluating test postconditions.\n\n"
        "TASK: Assess if the test properly returns the system to a safe/neutral state.\n\n"
        "CRITERIA:\n"
        "- Are cleanup steps defined?\n"
        "- Does the test leave the system in a known safe state?\n"
        "- Are resources properly released?\n\n"
        "Return ONLY valid JSON (no markdown, no code blocks) with this structure:\n"
        '{"postconditions_score": <float 0-1>, "returns_to_safe_state": <bool>, '
        '"cleanup_steps": [<list>], "rationale": "<explanation>"}'
    )
    return MedtechEvaluatorNode(
        llm=llm,
        response_model=PostconditionsResponse,
        system_prompt=prompt,
        link_type="Postconditions"
    )


# ============================================================================
# Coverage & Technique Evaluators
# ============================================================================

def make_technique_application_evaluator(llm: ChatOpenAI) -> MedtechEvaluatorNode:
    """
    Evaluates if the test utilizes EP, BVA, or Decision Tables appropriately.
    """
    prompt = (
        "You are a test design expert evaluating technique application.\n\n"
        "TASK: Assess if the test applies recognized testing techniques appropriately.\n\n"
        "TECHNIQUES TO IDENTIFY:\n"
        "- Equivalence Partitioning (EP)\n"
        "- Boundary Value Analysis (BVA)\n"
        "- Decision Tables\n"
        "- State Transition Testing\n"
        "- Use Case Testing\n\n"
        "CRITERIA:\n"
        "- Is a technique identifiable?\n"
        "- Is it appropriate for the requirement?\n"
        "- Is it applied correctly?\n\n"
        "Return ONLY valid JSON (no markdown, no code blocks) with this structure:\n"
        '{"technique_application_score": <float 0-1>, "technique_used": "<technique or null>", '
        '"is_appropriate": <bool>, "recommendations": "<suggestions>", "rationale": "<explanation>"}'
    )
    return MedtechEvaluatorNode(
        llm=llm,
        response_model=TechniqueApplicationResponse,
        system_prompt=prompt,
        link_type="TechniqueApplication"
    )


def make_negative_testing_evaluator(llm: ChatOpenAI) -> MedtechEvaluatorNode:
    """
    Evaluates if there are tests for Invalid Inputs, Timeouts, and Error States.
    """
    prompt = (
        "You are a robustness testing expert evaluating negative test coverage.\n\n"
        "TASK: Assess if the test addresses failure modes and invalid conditions.\n\n"
        "CRITERIA:\n"
        "- Are invalid inputs tested?\n"
        "- Are timeout conditions handled?\n"
        "- Are error states and exceptions tested?\n"
        "- Are out-of-range values tested?\n\n"
        "Return ONLY valid JSON (no markdown, no code blocks) with this structure:\n"
        '{"negative_testing_score": <float 0-1>, "has_negative_tests": <bool>, '
        '"missing_negative_cases": [<list>], "rationale": "<explanation>"}'
    )
    return MedtechEvaluatorNode(
        llm=llm,
        response_model=NegativeTestingResponse,
        system_prompt=prompt,
        link_type="NegativeTesting"
    )


def make_boundary_checks_evaluator(llm: ChatOpenAI) -> MedtechEvaluatorNode:
    """
    Evaluates if the edges (Min, Min-1, Max, Max+1) are explicitly verified.
    """
    prompt = (
        "You are a boundary value analysis expert evaluating boundary testing.\n\n"
        "TASK: Assess if the test explicitly verifies boundary conditions.\n\n"
        "CRITERIA:\n"
        "- Are minimum and maximum values tested?\n"
        "- Are just-inside and just-outside boundaries tested (Min-1, Max+1)?\n"
        "- Are special values tested (0, null, empty, etc.)?\n\n"
        "Return ONLY valid JSON (no markdown, no code blocks) with this structure:\n"
        '{"boundary_checks_score": <float 0-1>, "has_boundary_tests": <bool>, '
        '"missing_boundaries": [<list>], "rationale": "<explanation>"}'
    )
    return MedtechEvaluatorNode(
        llm=llm,
        response_model=BoundaryChecksResponse,
        system_prompt=prompt,
        link_type="BoundaryChecks"
    )


def make_risk_verification_evaluator(llm: ChatOpenAI) -> MedtechEvaluatorNode:
    """
    Evaluates if the test verifies the effectiveness of risk controls.
    """
    prompt = (
        "You are a medical device risk management expert evaluating risk-based testing.\n\n"
        "TASK: Assess if the test properly verifies risk control effectiveness.\n\n"
        "CRITERIA:\n"
        "- Is the test linked to a Risk ID?\n"
        "- Does it verify that the risk control is EFFECTIVE (not just that it exists)?\n"
        "- Example: Verifying an alarm is audible, not just that the alarm flag is set\n"
        "- Does it test the actual mitigation, not just the implementation?\n\n"
        "Return ONLY valid JSON (no markdown, no code blocks) with this structure:\n"
        '{"risk_verification_score": <float 0-1>, "is_linked_to_risk": <bool>, '
        '"verifies_control_effectiveness": <bool>, "verification_gaps": [<list>], "rationale": "<explanation>"}'
    )
    return MedtechEvaluatorNode(
        llm=llm,
        response_model=RiskVerificationResponse,
        system_prompt=prompt,
        link_type="RiskVerification"
    )


# ============================================================================
# Traceability & Compliance Evaluators
# ============================================================================

def make_traceability_evaluator(llm: ChatOpenAI) -> MedtechEvaluatorNode:
    """
    Evaluates if there is a correct link to a Requirement ID or Risk ID.
    """
    prompt = (
        "You are a traceability analyst evaluating requirement/risk linkage.\n\n"
        "TASK: Assess if the test has correct and complete traceability links.\n\n"
        "CRITERIA:\n"
        "- Is there a link to a Requirement ID?\n"
        "- Is there a link to a Risk ID (if applicable)?\n"
        "- Are the links correct and verifiable?\n"
        "- Is the coverage complete (test addresses the full requirement)?\n\n"
        "Return ONLY valid JSON (no markdown, no code blocks) with this structure:\n"
        '{"traceability_score": <float 0-1>, "has_correct_link": <bool>, '
        '"linked_ids": [<list>], "rationale": "<explanation>"}'
    )
    return MedtechEvaluatorNode(
        llm=llm,
        response_model=TraceabilityResponse,
        system_prompt=prompt,
        link_type="Traceability"
    )


def make_safety_class_rigor_evaluator(llm: ChatOpenAI) -> MedtechEvaluatorNode:
    """
    Evaluates if Class C units have associated Unit Tests with MC/DC coverage.
    """
    prompt = (
        "You are an IEC 62304 compliance expert evaluating safety class testing rigor.\n\n"
        "TASK: Assess if the test meets safety class requirements per IEC 62304.\n\n"
        "SAFETY CLASS REQUIREMENTS:\n"
        "- Class A: Basic testing acceptable\n"
        "- Class B: More rigorous testing, boundary analysis\n"
        "- Class C: MUST have unit tests with MC/DC (Modified Condition/Decision Coverage)\n\n"
        "CRITERIA:\n"
        "- Is the safety class identified?\n"
        "- For Class C: Is there an associated unit test?\n"
        "- For Class C: Is MC/DC coverage documented?\n\n"
        "Return ONLY valid JSON (no markdown, no code blocks) with this structure:\n"
        '{"safety_class_rigor_score": <float 0-1>, "safety_class": "<A/B/C or null>", '
        '"has_unit_test": <bool>, "has_mcdc_coverage": <bool>, "compliance_issues": [<list>], "rationale": "<explanation>"}'
    )
    return MedtechEvaluatorNode(
        llm=llm,
        response_model=SafetyClassRigorResponse,
        system_prompt=prompt,
        link_type="SafetyClassRigor"
    )


def make_objective_evidence_evaluator(llm: ChatOpenAI) -> MedtechEvaluatorNode:
    """
    Evaluates if expected results define specific values rather than subjective statements.
    """
    prompt = (
        "You are a regulatory compliance QA evaluating test evidence objectivity.\n\n"
        "TASK: Assess if the test provides objective, measurable evidence.\n\n"
        "CRITERIA:\n"
        "- Are expected results defined with specific values (e.g., '5V +/- 0.1V')?\n"
        "- Avoid subjective statements (e.g., 'Output is correct', 'System works properly')\n"
        "- Are acceptance criteria quantifiable and verifiable?\n"
        "- Can pass/fail be determined objectively?\n\n"
        "GOOD: 'Temperature displays 37.0°C +/- 0.5°C'\n"
        "BAD: 'Temperature is correct'\n\n"
        "Return ONLY valid JSON (no markdown, no code blocks) with this structure:\n"
        '{"objective_evidence_score": <float 0-1>, "has_specific_values": <bool>, '
        '"subjective_statements": [<list>], "improvement_suggestions": "<suggestions>", "rationale": "<explanation>"}'
    )
    return MedtechEvaluatorNode(
        llm=llm,
        response_model=ObjectiveEvidenceResponse,
        system_prompt=prompt,
        link_type="ObjectiveEvidence"
    )
