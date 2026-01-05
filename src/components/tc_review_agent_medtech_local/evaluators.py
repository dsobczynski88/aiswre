"""
Medtech evaluator factory functions for Ollama-based test case review.

Creates specialized evaluator nodes for FDA/IEC 62304 test case review using local Ollama models.
Each factory returns a node configured with domain-specific prompts and response models.
"""

try:
    from langchain_ollama import ChatOllama
except ImportError:
    from langchain_community.chat_models import ChatOllama

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

def make_unambiguity_evaluator(
    model: str = "llama3.1",
    base_url: str = "http://localhost:11434",
    temperature: float = 0.0
):
    """
    Evaluates if test instructions are clear enough that a person with no prior
    knowledge could execute them and get the exact same result.
    """
    llm = ChatOllama(model=model, base_url=base_url, temperature=temperature)
    structured_llm = llm.with_structured_output(UnambiguityResponse)

    SYSTEM_PROMPT = """You are a medical device QA specialist evaluating test case clarity.

TASK: Assess if the test instructions are unambiguous and executable by someone with no prior knowledge of the system.

CRITERIA:
- Are all steps explicitly defined?
- Are there undefined terms or references?
- Could two different people execute this identically?

Return ONLY valid JSON (no markdown, no code blocks) with this structure:
{"unambiguity_score": <float 0-1>, "is_unambiguous": <bool>, "ambiguous_phrases": [<list>], "rationale": "<explanation>"}"""

    def evaluator_node(state: dict) -> dict:
        from langchain_core.messages import SystemMessage, HumanMessage
        import json

        requirement = state.get('requirement')
        test = state.get('test')

        payload = {"test_id": test.test_id, "test_description": test.description}
        if test.preconditions:
            payload["preconditions"] = test.preconditions
        if test.steps:
            payload["steps"] = test.steps
        if test.expected_result:
            payload["expected_result"] = test.expected_result
        if requirement:
            payload["requirement_text"] = requirement.text

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=json.dumps(payload))
        ]

        result = structured_llm.invoke(messages)
        return {"raw_evaluator_responses": [{"type": "Unambiguity", "data": result}]}

    return evaluator_node


def make_independence_evaluator(
    model: str = "llama3.1",
    base_url: str = "http://localhost:11434",
    temperature: float = 0.0
):
    """Evaluates if the test is self-contained and cleans up its own data."""
    llm = ChatOllama(model=model, base_url=base_url, temperature=temperature)
    structured_llm = llm.with_structured_output(IndependenceResponse)

    SYSTEM_PROMPT = """You are a test automation expert evaluating test independence.

TASK: Assess if the test is self-contained and manages its own lifecycle.

CRITERIA:
- Does the test depend on other tests or external state?
- Does it create and clean up its own test data?
- Can it run in isolation without side effects?

Return ONLY valid JSON (no markdown, no code blocks) with this structure:
{"independence_score": <float 0-1>, "is_independent": <bool>, "dependencies": [<list>], "cleanup_issues": [<list>], "rationale": "<explanation>"}"""

    def evaluator_node(state: dict) -> dict:
        from langchain_core.messages import SystemMessage, HumanMessage
        import json

        requirement = state.get('requirement')
        test = state.get('test')

        payload = {"test_id": test.test_id, "test_description": test.description}
        if test.preconditions:
            payload["preconditions"] = test.preconditions
        if test.steps:
            payload["steps"] = test.steps
        if test.postconditions:
            payload["postconditions"] = test.postconditions

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=json.dumps(payload))
        ]

        result = structured_llm.invoke(messages)
        return {"raw_evaluator_responses": [{"type": "Independence", "data": result}]}

    return evaluator_node


def make_preconditions_evaluator(
    model: str = "llama3.1",
    base_url: str = "http://localhost:11434",
    temperature: float = 0.0
):
    """Evaluates if initial states (Power, Network, Database) are explicitly defined."""
    llm = ChatOllama(model=model, base_url=base_url, temperature=temperature)
    structured_llm = llm.with_structured_output(PreconditionsResponse)

    SYSTEM_PROMPT = """You are a verification engineer evaluating test preconditions.

TASK: Assess if the test explicitly defines all initial states required.

CRITERIA:
- Are system states (Power, Network, Database, Configuration) explicitly defined?
- Are environmental conditions specified?
- Is the starting state reproducible?

Return ONLY valid JSON (no markdown, no code blocks) with this structure:
{"preconditions_score": <float 0-1>, "are_explicit": <bool>, "missing_preconditions": [<list>], "rationale": "<explanation>"}"""

    def evaluator_node(state: dict) -> dict:
        from langchain_core.messages import SystemMessage, HumanMessage
        import json

        test = state.get('test')

        payload = {"test_id": test.test_id, "test_description": test.description}
        if test.preconditions:
            payload["preconditions"] = test.preconditions

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=json.dumps(payload))
        ]

        result = structured_llm.invoke(messages)
        return {"raw_evaluator_responses": [{"type": "Preconditions", "data": result}]}

    return evaluator_node


def make_postconditions_evaluator(
    model: str = "llama3.1",
    base_url: str = "http://localhost:11434",
    temperature: float = 0.0
):
    """Evaluates if the test returns the system to a safe/neutral state."""
    llm = ChatOllama(model=model, base_url=base_url, temperature=temperature)
    structured_llm = llm.with_structured_output(PostconditionsResponse)

    SYSTEM_PROMPT = """You are a safety-critical systems QA evaluating test postconditions.

TASK: Assess if the test properly returns the system to a safe/neutral state.

CRITERIA:
- Are cleanup steps defined?
- Does the test leave the system in a known safe state?
- Are resources properly released?

Return ONLY valid JSON (no markdown, no code blocks) with this structure:
{"postconditions_score": <float 0-1>, "returns_to_safe_state": <bool>, "cleanup_steps": [<list>], "rationale": "<explanation>"}"""

    def evaluator_node(state: dict) -> dict:
        from langchain_core.messages import SystemMessage, HumanMessage
        import json

        test = state.get('test')

        payload = {"test_id": test.test_id, "test_description": test.description}
        if test.postconditions:
            payload["postconditions"] = test.postconditions

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=json.dumps(payload))
        ]

        result = structured_llm.invoke(messages)
        return {"raw_evaluator_responses": [{"type": "Postconditions", "data": result}]}

    return evaluator_node


# ============================================================================
# Coverage & Technique Evaluators
# ============================================================================

def make_technique_application_evaluator(
    model: str = "llama3.1",
    base_url: str = "http://localhost:11434",
    temperature: float = 0.0
):
    """Evaluates if the test utilizes EP, BVA, or Decision Tables appropriately."""
    llm = ChatOllama(model=model, base_url=base_url, temperature=temperature)
    structured_llm = llm.with_structured_output(TechniqueApplicationResponse)

    SYSTEM_PROMPT = """You are a test design expert evaluating technique application.

TASK: Assess if the test applies recognized testing techniques appropriately.

TECHNIQUES TO IDENTIFY:
- Equivalence Partitioning (EP)
- Boundary Value Analysis (BVA)
- Decision Tables
- State Transition Testing
- Use Case Testing

CRITERIA:
- Is a technique identifiable?
- Is it appropriate for the requirement?
- Is it applied correctly?

Return ONLY valid JSON (no markdown, no code blocks) with this structure:
{"technique_application_score": <float 0-1>, "technique_used": "<technique or null>", "is_appropriate": <bool>, "recommendations": "<suggestions>", "rationale": "<explanation>"}"""

    def evaluator_node(state: dict) -> dict:
        from langchain_core.messages import SystemMessage, HumanMessage
        import json

        requirement = state.get('requirement')
        test = state.get('test')

        payload = {"test_id": test.test_id, "test_description": test.description}
        if test.technique:
            payload["technique"] = test.technique
        if test.steps:
            payload["steps"] = test.steps
        if requirement:
            payload["requirement_text"] = requirement.text

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=json.dumps(payload))
        ]

        result = structured_llm.invoke(messages)
        return {"raw_evaluator_responses": [{"type": "TechniqueApplication", "data": result}]}

    return evaluator_node


def make_negative_testing_evaluator(
    model: str = "llama3.1",
    base_url: str = "http://localhost:11434",
    temperature: float = 0.0
):
    """Evaluates if there are tests for Invalid Inputs, Timeouts, and Error States."""
    llm = ChatOllama(model=model, base_url=base_url, temperature=temperature)
    structured_llm = llm.with_structured_output(NegativeTestingResponse)

    SYSTEM_PROMPT = """You are a robustness testing expert evaluating negative test coverage.

TASK: Assess if the test addresses failure modes and invalid conditions.

CRITERIA:
- Are invalid inputs tested?
- Are timeout conditions handled?
- Are error states and exceptions tested?
- Are out-of-range values tested?

Return ONLY valid JSON (no markdown, no code blocks) with this structure:
{"negative_testing_score": <float 0-1>, "has_negative_tests": <bool>, "missing_negative_cases": [<list>], "rationale": "<explanation>"}"""

    def evaluator_node(state: dict) -> dict:
        from langchain_core.messages import SystemMessage, HumanMessage
        import json

        test = state.get('test')

        payload = {"test_id": test.test_id, "test_description": test.description}
        if test.steps:
            payload["steps"] = test.steps
        if test.expected_result:
            payload["expected_result"] = test.expected_result

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=json.dumps(payload))
        ]

        result = structured_llm.invoke(messages)
        return {"raw_evaluator_responses": [{"type": "NegativeTesting", "data": result}]}

    return evaluator_node


def make_boundary_checks_evaluator(
    model: str = "llama3.1",
    base_url: str = "http://localhost:11434",
    temperature: float = 0.0
):
    """Evaluates if the edges (Min, Min-1, Max, Max+1) are explicitly verified."""
    llm = ChatOllama(model=model, base_url=base_url, temperature=temperature)
    structured_llm = llm.with_structured_output(BoundaryChecksResponse)

    SYSTEM_PROMPT = """You are a boundary value analysis expert evaluating boundary testing.

TASK: Assess if the test explicitly verifies boundary conditions.

CRITERIA:
- Are minimum and maximum values tested?
- Are just-inside and just-outside boundaries tested (Min-1, Max+1)?
- Are special values tested (0, null, empty, etc.)?

Return ONLY valid JSON (no markdown, no code blocks) with this structure:
{"boundary_checks_score": <float 0-1>, "has_boundary_tests": <bool>, "missing_boundaries": [<list>], "rationale": "<explanation>"}"""

    def evaluator_node(state: dict) -> dict:
        from langchain_core.messages import SystemMessage, HumanMessage
        import json

        requirement = state.get('requirement')
        test = state.get('test')

        payload = {"test_id": test.test_id, "test_description": test.description}
        if test.steps:
            payload["steps"] = test.steps
        if test.expected_result:
            payload["expected_result"] = test.expected_result
        if requirement:
            payload["requirement_text"] = requirement.text

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=json.dumps(payload))
        ]

        result = structured_llm.invoke(messages)
        return {"raw_evaluator_responses": [{"type": "BoundaryChecks", "data": result}]}

    return evaluator_node


def make_risk_verification_evaluator(
    model: str = "llama3.1",
    base_url: str = "http://localhost:11434",
    temperature: float = 0.0
):
    """Evaluates if the test verifies the effectiveness of risk controls."""
    llm = ChatOllama(model=model, base_url=base_url, temperature=temperature)
    structured_llm = llm.with_structured_output(RiskVerificationResponse)

    SYSTEM_PROMPT = """You are a medical device risk management expert evaluating risk-based testing.

TASK: Assess if the test properly verifies risk control effectiveness.

CRITERIA:
- Is the test linked to a Risk ID?
- Does it verify that the risk control is EFFECTIVE (not just that it exists)?
- Example: Verifying an alarm is audible, not just that the alarm flag is set
- Does it test the actual mitigation, not just the implementation?

Return ONLY valid JSON (no markdown, no code blocks) with this structure:
{"risk_verification_score": <float 0-1>, "is_linked_to_risk": <bool>, "verifies_control_effectiveness": <bool>, "verification_gaps": [<list>], "rationale": "<explanation>"}"""

    def evaluator_node(state: dict) -> dict:
        from langchain_core.messages import SystemMessage, HumanMessage
        import json

        requirement = state.get('requirement')
        test = state.get('test')

        payload = {"test_id": test.test_id, "test_description": test.description}
        if test.steps:
            payload["steps"] = test.steps
        if test.expected_result:
            payload["expected_result"] = test.expected_result
        if requirement:
            payload["requirement_text"] = requirement.text
            if requirement.risk_id:
                payload["risk_id"] = requirement.risk_id

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=json.dumps(payload))
        ]

        result = structured_llm.invoke(messages)
        return {"raw_evaluator_responses": [{"type": "RiskVerification", "data": result}]}

    return evaluator_node


# ============================================================================
# Traceability & Compliance Evaluators
# ============================================================================

def make_traceability_evaluator(
    model: str = "llama3.1",
    base_url: str = "http://localhost:11434",
    temperature: float = 0.0
):
    """Evaluates if there is a correct link to a Requirement ID or Risk ID."""
    llm = ChatOllama(model=model, base_url=base_url, temperature=temperature)
    structured_llm = llm.with_structured_output(TraceabilityResponse)

    SYSTEM_PROMPT = """You are a traceability analyst evaluating requirement/risk linkage.

TASK: Assess if the test has correct and complete traceability links.

CRITERIA:
- Is there a link to a Requirement ID?
- Is there a link to a Risk ID (if applicable)?
- Are the links correct and verifiable?
- Is the coverage complete (test addresses the full requirement)?

Return ONLY valid JSON (no markdown, no code blocks) with this structure:
{"traceability_score": <float 0-1>, "has_correct_link": <bool>, "linked_ids": [<list>], "rationale": "<explanation>"}"""

    def evaluator_node(state: dict) -> dict:
        from langchain_core.messages import SystemMessage, HumanMessage
        import json

        requirement = state.get('requirement')
        test = state.get('test')

        payload = {"test_id": test.test_id, "test_description": test.description}
        if requirement:
            payload["requirement_id"] = requirement.req_id
            payload["requirement_text"] = requirement.text
            if requirement.risk_id:
                payload["risk_id"] = requirement.risk_id

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=json.dumps(payload))
        ]

        result = structured_llm.invoke(messages)
        return {"raw_evaluator_responses": [{"type": "Traceability", "data": result}]}

    return evaluator_node


def make_safety_class_rigor_evaluator(
    model: str = "llama3.1",
    base_url: str = "http://localhost:11434",
    temperature: float = 0.0
):
    """Evaluates if Class C units have associated Unit Tests with MC/DC coverage."""
    llm = ChatOllama(model=model, base_url=base_url, temperature=temperature)
    structured_llm = llm.with_structured_output(SafetyClassRigorResponse)

    SYSTEM_PROMPT = """You are an IEC 62304 compliance expert evaluating safety class testing rigor.

TASK: Assess if the test meets safety class requirements per IEC 62304.

SAFETY CLASS REQUIREMENTS:
- Class A: Basic testing acceptable
- Class B: More rigorous testing, boundary analysis
- Class C: MUST have unit tests with MC/DC (Modified Condition/Decision Coverage)

CRITERIA:
- Is the safety class identified?
- For Class C: Is there an associated unit test?
- For Class C: Is MC/DC coverage documented?

Return ONLY valid JSON (no markdown, no code blocks) with this structure:
{"safety_class_rigor_score": <float 0-1>, "safety_class": "<A/B/C or null>", "has_unit_test": <bool>, "has_mcdc_coverage": <bool>, "compliance_issues": [<list>], "rationale": "<explanation>"}"""

    def evaluator_node(state: dict) -> dict:
        from langchain_core.messages import SystemMessage, HumanMessage
        import json

        requirement = state.get('requirement')
        test = state.get('test')

        payload = {"test_id": test.test_id, "test_description": test.description}
        if test.test_type:
            payload["test_type"] = test.test_type
        if requirement and requirement.safety_class:
            payload["safety_class"] = requirement.safety_class

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=json.dumps(payload))
        ]

        result = structured_llm.invoke(messages)
        return {"raw_evaluator_responses": [{"type": "SafetyClassRigor", "data": result}]}

    return evaluator_node


def make_objective_evidence_evaluator(
    model: str = "llama3.1",
    base_url: str = "http://localhost:11434",
    temperature: float = 0.0
):
    """Evaluates if expected results define specific values rather than subjective statements."""
    llm = ChatOllama(model=model, base_url=base_url, temperature=temperature)
    structured_llm = llm.with_structured_output(ObjectiveEvidenceResponse)

    SYSTEM_PROMPT = """You are a regulatory compliance QA evaluating test evidence objectivity.

TASK: Assess if the test provides objective, measurable evidence.

CRITERIA:
- Are expected results defined with specific values (e.g., '5V +/- 0.1V')?
- Avoid subjective statements (e.g., 'Output is correct', 'System works properly')
- Are acceptance criteria quantifiable and verifiable?
- Can pass/fail be determined objectively?

GOOD: 'Temperature displays 37.0°C +/- 0.5°C'
BAD: 'Temperature is correct'

Return ONLY valid JSON (no markdown, no code blocks) with this structure:
{"objective_evidence_score": <float 0-1>, "has_specific_values": <bool>, "subjective_statements": [<list>], "improvement_suggestions": "<suggestions>", "rationale": "<explanation>"}"""

    def evaluator_node(state: dict) -> dict:
        from langchain_core.messages import SystemMessage, HumanMessage
        import json

        test = state.get('test')

        payload = {"test_id": test.test_id, "test_description": test.description}
        if test.expected_result:
            payload["expected_result"] = test.expected_result

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=json.dumps(payload))
        ]

        result = structured_llm.invoke(messages)
        return {"raw_evaluator_responses": [{"type": "ObjectiveEvidence", "data": result}]}

    return evaluator_node
