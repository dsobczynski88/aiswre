"""
RTM evaluator factory functions for Ollama-based requirement verification coverage review.

Creates specialized evaluator nodes for assessing how well a suite of test cases
verifies a requirement, aligned with FDA/IEC 62304 verification principles.
"""

try:
    from langchain_ollama import ChatOllama
except ImportError:
    from langchain_community.chat_models import ChatOllama

from .core import (
    FunctionalCoverageResponse,
    InputOutputCoverageResponse,
    BoundaryCoverageResponse,
    NegativeTestCoverageResponse,
    RiskCoverageResponse,
    TraceabilityCompletenessResponse,
    AcceptanceCriteriaCoverageResponse,
    TestSufficiencyResponse,
    GapAnalysisResponse
)


# ============================================================================
# Coverage Evaluators
# ============================================================================

def make_functional_coverage_evaluator(
    model: str = "llama3.1",
    base_url: str = "http://localhost:11434",
    temperature: float = 0.0
):
    """
    Evaluates if all functional aspects of the requirement are covered by the test cases.
    """
    llm = ChatOllama(model=model, base_url=base_url, temperature=temperature)
    structured_llm = llm.with_structured_output(FunctionalCoverageResponse)

    SYSTEM_PROMPT = """You are a medical device verification specialist evaluating functional coverage.

TASK: Assess if the test cases cover all functional aspects of the requirement.

CRITERIA:
- Are all stated functions/capabilities verified?
- Are both normal operations and alternate flows tested?
- Does the test suite address the complete functional scope?

Return ONLY valid JSON (no markdown, no code blocks) with this structure:
{"functional_coverage_score": <float 0-1>, "all_functions_tested": <bool>, "missing_functions": [<list>], "rationale": "<explanation>"}"""

    def evaluator_node(state: dict) -> dict:
        from langchain_core.messages import SystemMessage, HumanMessage
        import json

        requirement = state.get('requirement')
        rtm_entry = state.get('rtm_entry')

        payload = {
            "requirement_id": requirement.req_id,
            "requirement_text": requirement.text,
            "test_case_summary": rtm_entry.test_case_summary
        }

        if rtm_entry.test_case_count:
            payload["test_case_count"] = rtm_entry.test_case_count
        if rtm_entry.test_ids:
            payload["test_ids"] = rtm_entry.test_ids

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=json.dumps(payload))
        ]

        result = structured_llm.invoke(messages)
        return {"raw_evaluator_responses": [{"type": "FunctionalCoverage", "data": result}]}

    return evaluator_node


def make_input_output_coverage_evaluator(
    model: str = "llama3.1",
    base_url: str = "http://localhost:11434",
    temperature: float = 0.0
):
    """
    Evaluates if all input conditions and expected outputs are tested.
    """
    llm = ChatOllama(model=model, base_url=base_url, temperature=temperature)
    structured_llm = llm.with_structured_output(InputOutputCoverageResponse)

    SYSTEM_PROMPT = """You are a test coverage analyst evaluating input/output verification.

TASK: Assess if all input conditions and expected outputs are verified by the test cases.

CRITERIA:
- Are all input types/ranges tested?
- Are all valid input combinations covered?
- Are all expected outputs verified?
- Are output conditions (success, error, warnings) tested?

Return ONLY valid JSON (no markdown, no code blocks) with this structure:
{"input_output_coverage_score": <float 0-1>, "all_inputs_tested": <bool>, "all_outputs_verified": <bool>, "missing_inputs": [<list>], "missing_outputs": [<list>], "rationale": "<explanation>"}"""

    def evaluator_node(state: dict) -> dict:
        from langchain_core.messages import SystemMessage, HumanMessage
        import json

        requirement = state.get('requirement')
        rtm_entry = state.get('rtm_entry')

        payload = {
            "requirement_id": requirement.req_id,
            "requirement_text": requirement.text,
            "test_case_summary": rtm_entry.test_case_summary
        }

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=json.dumps(payload))
        ]

        result = structured_llm.invoke(messages)
        return {"raw_evaluator_responses": [{"type": "InputOutputCoverage", "data": result}]}

    return evaluator_node


def make_boundary_coverage_evaluator(
    model: str = "llama3.1",
    base_url: str = "http://localhost:11434",
    temperature: float = 0.0
):
    """
    Evaluates if boundary conditions are tested (min, max, edges).
    """
    llm = ChatOllama(model=model, base_url=base_url, temperature=temperature)
    structured_llm = llm.with_structured_output(BoundaryCoverageResponse)

    SYSTEM_PROMPT = """You are a boundary value analysis expert evaluating boundary coverage.

TASK: Assess if the test cases verify boundary conditions for the requirement.

CRITERIA:
- Are minimum and maximum values tested?
- Are boundary edges tested (just inside/outside limits)?
- Are special values tested (0, null, empty, etc.)?
- Are threshold conditions verified?

Return ONLY valid JSON (no markdown, no code blocks) with this structure:
{"boundary_coverage_score": <float 0-1>, "has_boundary_tests": <bool>, "missing_boundaries": [<list>], "rationale": "<explanation>"}"""

    def evaluator_node(state: dict) -> dict:
        from langchain_core.messages import SystemMessage, HumanMessage
        import json

        requirement = state.get('requirement')
        rtm_entry = state.get('rtm_entry')

        payload = {
            "requirement_id": requirement.req_id,
            "requirement_text": requirement.text,
            "test_case_summary": rtm_entry.test_case_summary
        }

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=json.dumps(payload))
        ]

        result = structured_llm.invoke(messages)
        return {"raw_evaluator_responses": [{"type": "BoundaryCoverage", "data": result}]}

    return evaluator_node


def make_negative_test_coverage_evaluator(
    model: str = "llama3.1",
    base_url: str = "http://localhost:11434",
    temperature: float = 0.0
):
    """
    Evaluates if error cases, invalid inputs, and exceptions are tested.
    """
    llm = ChatOllama(model=model, base_url=base_url, temperature=temperature)
    structured_llm = llm.with_structured_output(NegativeTestCoverageResponse)

    SYSTEM_PROMPT = """You are a robustness testing expert evaluating negative test coverage.

TASK: Assess if the test cases verify error handling and invalid conditions.

CRITERIA:
- Are invalid inputs tested?
- Are error conditions verified?
- Are exception scenarios covered?
- Are failure modes tested?
- Are timeout/abort conditions tested?

Return ONLY valid JSON (no markdown, no code blocks) with this structure:
{"negative_test_coverage_score": <float 0-1>, "has_negative_tests": <bool>, "missing_negative_cases": [<list>], "rationale": "<explanation>"}"""

    def evaluator_node(state: dict) -> dict:
        from langchain_core.messages import SystemMessage, HumanMessage
        import json

        requirement = state.get('requirement')
        rtm_entry = state.get('rtm_entry')

        payload = {
            "requirement_id": requirement.req_id,
            "requirement_text": requirement.text,
            "test_case_summary": rtm_entry.test_case_summary
        }

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=json.dumps(payload))
        ]

        result = structured_llm.invoke(messages)
        return {"raw_evaluator_responses": [{"type": "NegativeTestCoverage", "data": result}]}

    return evaluator_node


# ============================================================================
# Risk & Traceability Evaluators
# ============================================================================

def make_risk_coverage_evaluator(
    model: str = "llama3.1",
    base_url: str = "http://localhost:11434",
    temperature: float = 0.0
):
    """
    Evaluates if risk controls are verified by the test cases.
    """
    llm = ChatOllama(model=model, base_url=base_url, temperature=temperature)
    structured_llm = llm.with_structured_output(RiskCoverageResponse)

    SYSTEM_PROMPT = """You are a medical device risk management expert evaluating risk control verification.

TASK: Assess if the test cases verify risk mitigation controls for this requirement.

CRITERIA:
- Is the requirement linked to a Risk ID?
- Do the test cases verify that risk controls are EFFECTIVE (not just present)?
- Are hazardous scenarios tested?
- Are risk mitigation measures validated?

Return ONLY valid JSON (no markdown, no code blocks) with this structure:
{"risk_coverage_score": <float 0-1>, "is_linked_to_risk": <bool>, "risk_controls_verified": <bool>, "verification_gaps": [<list>], "rationale": "<explanation>"}"""

    def evaluator_node(state: dict) -> dict:
        from langchain_core.messages import SystemMessage, HumanMessage
        import json

        requirement = state.get('requirement')
        rtm_entry = state.get('rtm_entry')

        payload = {
            "requirement_id": requirement.req_id,
            "requirement_text": requirement.text,
            "test_case_summary": rtm_entry.test_case_summary
        }

        if requirement.risk_id:
            payload["risk_id"] = requirement.risk_id
        if requirement.safety_class:
            payload["safety_class"] = requirement.safety_class

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=json.dumps(payload))
        ]

        result = structured_llm.invoke(messages)
        return {"raw_evaluator_responses": [{"type": "RiskCoverage", "data": result}]}

    return evaluator_node


def make_traceability_completeness_evaluator(
    model: str = "llama3.1",
    base_url: str = "http://localhost:11434",
    temperature: float = 0.0
):
    """
    Evaluates if the requirement-to-test mapping is clear and complete.
    """
    llm = ChatOllama(model=model, base_url=base_url, temperature=temperature)
    structured_llm = llm.with_structured_output(TraceabilityCompletenessResponse)

    SYSTEM_PROMPT = """You are a traceability analyst evaluating RTM completeness.

TASK: Assess if the requirement-to-test mapping is clear, complete, and verifiable.

CRITERIA:
- Is the mapping between requirement and test cases clear?
- Are test IDs explicitly identified?
- Is the traceability bidirectional (req→test and test→req)?
- Can the mapping be verified/audited?

Return ONLY valid JSON (no markdown, no code blocks) with this structure:
{"traceability_completeness_score": <float 0-1>, "has_clear_mapping": <bool>, "test_ids_identified": [<list>], "traceability_issues": [<list>], "rationale": "<explanation>"}"""

    def evaluator_node(state: dict) -> dict:
        from langchain_core.messages import SystemMessage, HumanMessage
        import json

        requirement = state.get('requirement')
        rtm_entry = state.get('rtm_entry')

        payload = {
            "requirement_id": requirement.req_id,
            "requirement_text": requirement.text,
            "test_case_summary": rtm_entry.test_case_summary
        }

        if rtm_entry.test_ids:
            payload["test_ids"] = rtm_entry.test_ids

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=json.dumps(payload))
        ]

        result = structured_llm.invoke(messages)
        return {"raw_evaluator_responses": [{"type": "TraceabilityCompleteness", "data": result}]}

    return evaluator_node


def make_acceptance_criteria_coverage_evaluator(
    model: str = "llama3.1",
    base_url: str = "http://localhost:11434",
    temperature: float = 0.0
):
    """
    Evaluates if all acceptance criteria are verified by test cases.
    """
    llm = ChatOllama(model=model, base_url=base_url, temperature=temperature)
    structured_llm = llm.with_structured_output(AcceptanceCriteriaCoverageResponse)

    SYSTEM_PROMPT = """You are a requirements verification specialist evaluating acceptance criteria coverage.

TASK: Assess if all acceptance criteria for the requirement are verified by test cases.

CRITERIA:
- Are acceptance criteria explicitly defined?
- Is each acceptance criterion verified by at least one test?
- Are verification methods appropriate for each criterion?
- Is pass/fail determinable objectively?

Return ONLY valid JSON (no markdown, no code blocks) with this structure:
{"acceptance_criteria_coverage_score": <float 0-1>, "has_acceptance_criteria": <bool>, "all_criteria_verified": <bool>, "missing_criteria": [<list>], "rationale": "<explanation>"}"""

    def evaluator_node(state: dict) -> dict:
        from langchain_core.messages import SystemMessage, HumanMessage
        import json

        requirement = state.get('requirement')
        rtm_entry = state.get('rtm_entry')

        payload = {
            "requirement_id": requirement.req_id,
            "requirement_text": requirement.text,
            "test_case_summary": rtm_entry.test_case_summary
        }

        if requirement.acceptance_criteria:
            payload["acceptance_criteria"] = requirement.acceptance_criteria

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=json.dumps(payload))
        ]

        result = structured_llm.invoke(messages)
        return {"raw_evaluator_responses": [{"type": "AcceptanceCriteriaCoverage", "data": result}]}

    return evaluator_node


# ============================================================================
# Sufficiency & Gap Analysis Evaluators
# ============================================================================

def make_test_sufficiency_evaluator(
    model: str = "llama3.1",
    base_url: str = "http://localhost:11434",
    temperature: float = 0.0
):
    """
    Evaluates if there are enough tests to fully verify the requirement.
    """
    llm = ChatOllama(model=model, base_url=base_url, temperature=temperature)
    structured_llm = llm.with_structured_output(TestSufficiencyResponse)

    SYSTEM_PROMPT = """You are a test planning expert evaluating test sufficiency.

TASK: Assess if the quantity and quality of test cases are sufficient to verify the requirement.

CRITERIA:
- Is the number of tests appropriate for requirement complexity?
- Are there enough tests to cover all aspects?
- Are test techniques appropriate and comprehensive?
- Would additional tests add meaningful verification value?

Return ONLY valid JSON (no markdown, no code blocks) with this structure:
{"test_sufficiency_score": <float 0-1>, "has_sufficient_tests": <bool>, "test_count_assessment": "<assessment>", "sufficiency_concerns": [<list>], "rationale": "<explanation>"}"""

    def evaluator_node(state: dict) -> dict:
        from langchain_core.messages import SystemMessage, HumanMessage
        import json

        requirement = state.get('requirement')
        rtm_entry = state.get('rtm_entry')

        payload = {
            "requirement_id": requirement.req_id,
            "requirement_text": requirement.text,
            "test_case_summary": rtm_entry.test_case_summary
        }

        if rtm_entry.test_case_count:
            payload["test_case_count"] = rtm_entry.test_case_count

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=json.dumps(payload))
        ]

        result = structured_llm.invoke(messages)
        return {"raw_evaluator_responses": [{"type": "TestSufficiency", "data": result}]}

    return evaluator_node


def make_gap_analysis_evaluator(
    model: str = "llama3.1",
    base_url: str = "http://localhost:11434",
    temperature: float = 0.0
):
    """
    Performs gap analysis to identify critical verification gaps.
    """
    llm = ChatOllama(model=model, base_url=base_url, temperature=temperature)
    structured_llm = llm.with_structured_output(GapAnalysisResponse)

    SYSTEM_PROMPT = """You are a verification gap analyst identifying testing deficiencies.

TASK: Identify critical and moderate gaps in requirement verification coverage.

CRITERIA:
- What aspects of the requirement are NOT verified?
- What are the critical gaps that pose compliance/safety risks?
- What are moderate gaps that reduce verification confidence?
- What specific tests should be added?

Return ONLY valid JSON (no markdown, no code blocks) with this structure:
{"gap_analysis_score": <float 0-1>, "critical_gaps": [<list>], "moderate_gaps": [<list>], "recommendations": [<list>], "rationale": "<explanation>"}

Note: gap_analysis_score is INVERSE of gap severity (1.0 = no gaps, 0.0 = many critical gaps)"""

    def evaluator_node(state: dict) -> dict:
        from langchain_core.messages import SystemMessage, HumanMessage
        import json

        requirement = state.get('requirement')
        rtm_entry = state.get('rtm_entry')

        payload = {
            "requirement_id": requirement.req_id,
            "requirement_text": requirement.text,
            "test_case_summary": rtm_entry.test_case_summary
        }

        if requirement.safety_class:
            payload["safety_class"] = requirement.safety_class

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=json.dumps(payload))
        ]

        result = structured_llm.invoke(messages)
        return {"raw_evaluator_responses": [{"type": "GapAnalysis", "data": result}]}

    return evaluator_node
