"""
Node implementations for RTM review agent.
"""
import json
from typing import Optional, List, TypedDict, Annotated, Dict
from langchain_core.messages import SystemMessage, HumanMessage
from .core import (
    RTMReviewState,
    Requirement,
    DecomposedRequirement,
    TestCase,
    TestSuite,
    CoverageEvaluator,
    ReviewComment
)

class DecomposerNode:

    def __init__(self, llm, response_model, system_prompt):
        self.llm = llm
        self.response_model = response_model
        self.structured_llm = llm.with_structured_output(response_model)
        self.system_prompt = system_prompt

    @staticmethod
    def _build_payload(requirement: Requirement) -> dict:
        """Build the payload dict sent to the LLM."""
        payload = {
            "requirement_id": requirement.req_id,
            "requirement": requirement.text,
        }
        return payload

    def __call__(self, state: Dict) -> Dict:
        requirement = state.get("requirement")
        # Build payload for LLM
        payload = self._build_payload(requirement)

        try:
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=json.dumps(payload))
            ]
            # Use structured output to get Pydantic model directly
            parsed = self.structured_llm.invoke(messages)
        except Exception as e:
            print(e)
            parsed = None

        return {"decomposed_requirement": parsed}

class SummaryNode:

    def __init__(self, llm, response_model, system_prompt):
        self.llm = llm
        self.response_model = response_model
        self.structured_llm = llm.with_structured_output(response_model)
        self.system_prompt = system_prompt

    @staticmethod
    def _build_payload(test_cases: List[TestCase]) -> list:
        """Build the payload list sent to the LLM."""
        payload = [
            {
                "test_id": tc.test_id,
                "description": tc.description,
                "setup": tc.setup,
                "steps": tc.steps,
                "expectedResults": tc.expectedResults
            }
            for tc in test_cases
        ]
        return payload

    def __call__(self, state: Dict) -> Dict:
        test_cases = state.get("test_cases")
        # Build payload for LLM
        payload = self._build_payload(test_cases)

        try:
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=json.dumps(payload))
            ]
            # Use structured output to get Pydantic model directly
            parsed = self.structured_llm.invoke(messages)
        except Exception as e:
            print(e)
            parsed = None

        return {"test_suite": parsed}


class BaseEvaluatorNode:

    def __init__(self, llm, response_model, system_prompt):
        self.llm=llm
        self.response_model=response_model
        self.structured_llm = llm.with_structured_output(response_model)
        self.system_prompt=system_prompt

    
    @staticmethod
    def _build_payload(
        requirement: Requirement, 
        decomposed_requirement: DecomposedRequirement, 
        test_suite: TestSuite
        ) -> dict:
        
        payload = {
            "original_requirement": requirement.model_dump(),
            "decomposed_requirement": decomposed_requirement.model_dump(),
            "test_suite": test_suite.model_dump(),
        }
        return payload
    
    async def __call__(self, state: Dict) -> Dict:
        original_requirement = state.get("requirement")
        decomposed_requirement = state.get("decomposed_requirement")
        test_suite = state.get("test_suite")
        # Build payload for LLM
        payload = self._build_payload(original_requirement, decomposed_requirement, test_suite)

        try:
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=json.dumps(payload))
            ]
            # Use structured output to get Pydantic model directly
            parsed = await self.structured_llm.ainvoke(messages)
        except Exception as e:
            print(e)
            
        # Return parsed instance in a list — LangGraph uses operator.add to merge
        return {"coverage_responses": [parsed]}
        

def make_decomposer_node(llm) -> DecomposerNode:
    system_prompt = """
    ### Role
    Act as a Senior Medical Device Systems Engineer and Requirements Analyst specializing in IEC 62304 and ISO 14971 Risk Management. Your expertise is in systematic requirement decomposition and hazard analysis for safety-critical software systems. 

    ### Context
    You are an intelligent node in a verification pipeline. Your task is to transform a high-level, often ambiguous medical device requirement into granular, atomic specifications. Beyond simple decomposition, you must identify potential failure modes and edge cases that could lead to escaped defects—scenarios where the system might technically meet a basic requirement but fail under stress, concurrency, or exceptional conditions. 

    ### Instructions
    1. **Decompose**: Break the [Requirement Statement] into "Summary," "User," and "Sub-function" goals. 
    2. **Standardize**: Ensure every decomposed specification meets ISO 29148 characteristics (Unambiguous, Singular, Verifiable). 
    3. **Stress-Test**: For every atomic specification, identify 3-4 specific edge cases (e.g., race conditions, resource exhaustion, or UI/state synchronization issues) that could cause a defect to "escape" standard testing. 
    4. **Mitigate**: Suggest a technical mitigation for each identified risk. 

    ### Steps
    1. **Singularity Analysis**: Isolate unique Actor-Verb-Object relationships to ensure each spec is atomic. 
    2. **Flow Mapping**: Define the "Happy Path" and identify "Alternative" and "Exception" flows. 
    3. **NFR Extraction**: Quantify implicit performance, safety, and security constraints (e.g., specific response times in milliseconds, encryption standards). 
    4. **Boundary & Concurrency Analysis**: Identify edge cases such as rapid-fire inputs, partial authentication states, or latency during critical UI transitions.
    5. **JSON Synthesis**: Map all findings into the strict Pydantic-compatible schema provided.

    ### Narrowing (Constraints)
    - **Quantifiable Metrics Only**: Avoid subjective terms like "fast" or "secure"; use exact bounds (e.g., "< 500ms", "AES-256"). 
    - **Medical Specificity**: Focus on risks relevant to patient safety and data integrity (e.g., session mismanagement, stale data display). 
    - **Strict Atomicity**: If a specification contains "and" or "or," it must be split into two separate entries. 
    - **Output Format**: Return ONLY valid JSON. No conversational preamble.

    ### JSON Schema Requirement
    {
    "requirement_id": "string",
    "original_statement": "string",
    "decomposed_specifications": [
        {
        "spec_id": "string",
        "type": "functional | performance | safety | security",
        "description": "string (singular, atomic)",
        "verification_method": "test | demo | analysis | inspection",
        "acceptance_criteria": "string (measurable)",
        "rationale": "string",
        "edge_case_analysis": {
            "potential_edge_cases": ["string"],
            "risk_of_escaped_defect": "low | medium | high",
            "recommended_mitigation": "string"
        }
        }
    ]
    }
    """
    return DecomposerNode(
        llm=llm,
        response_model=DecomposedRequirement,
        system_prompt=system_prompt,
    )

def make_summarizer_node(llm) -> SummaryNode:
    system_prompt = """
    <role>
    Act as a Senior QA Automation Architect specializing in requirement traceability and boundary value analysis. Your goal is to function as a high-precision "Summarizer Node" within a multi-agent testing pipeline.
    </role>

    <context>
    You are positioned between a "Decomposer Node" (which identifies functional requirements and edge cases) and a "Boundary Evaluator Node" (which maps tests to those edge cases). 
    To ensure the Boundary Evaluator can accurately perform its job, you must ingest raw test data and transform it into a summarized JSON format that explicitly highlights the logic required to satisfy the "edge_case_analysis" provided by the Decomposer.
    Below is the reference schema from the Decomposer Node to guide your understanding of the requirement landscape:
    {
        "requirement_id": "string",
        "original_statement": "string",
        "decomposed_specifications": [
            {
            "spec_id": "string",
            "type": "functional | performance | safety | security",
            "description": "string (singular, atomic)",
            "verification_method": "test | demo | analysis | inspection",
            "acceptance_criteria": "string (measurable)",
            "rationale": "string",
            "edge_case_analysis": {
                "potential_edge_cases": ["string"],
                "risk_of_escaped_defect": "low | medium | high",
                "recommended_mitigation": "string"
            }
            }
        ]
    }
    Below is the reference pydantic model of a single test case that needs to be summarized. The user will provide a list of the data class TestCase:  

    class TestCase(BaseModel):
        test_id: str
        description: str
        setup: Optional[str] = None
        steps: Optional[str] = None
        expectedResults: Optional[str] = None
    </context>

    <instructions>
    Summarize the raw test cases provided by the user. Follow these logical steps for each test case:
    1. Retain the exact "Test Case ID".
    2. Synthesize the "Objective" via the description and identifying the key inputs and outputs of the function.
    3. Define what the test "Verifies" by mapping the test's intent to specific functional aspects of the requirement.
    4. Distill the "Protocol" into a concise summary of the execution steps.
    5. Extract "Acceptance Criteria" from the Expected Results. If a test case contains multiple distinct validation points, return them as a list within the string.
    </instructions>

    <narrowing>
    - Output MUST be valid JSON.
    - Do not include any conversational filler or "here is the output" text.
    - Ensure the "verifies" field uses technical language compatible with boundary testing (e.g., "validates upper bound," "tests null persistence").
    - If the raw test case is lengthy, ensure the "acceptance criteria" captures every distinct outcome as a separate entry in the list.
    </narrowing>

    <format_template>
    {
    "summarized_test_cases": [
        {
        "test_case_id": "string",
        "objective": "string",
        "verifies": "string",
        "protocol": "string",
        "acceptance_criteria": ["string", "string"]
        }
    ]
    }
    </format_template>
    """
    return SummaryNode(
        llm=llm,
        response_model=TestSuite,
        system_prompt=system_prompt,
    )

def make_boundary_coverage_evaluator(llm) -> BaseEvaluatorNode:
    system_prompt="""
    # ROLE
    Act as a Senior Software Verification & Validation (V&V) Engineer specializing in Medical Device Software (IEC 62304 / IEC 82304 / ISO 14971). Your expertise lies in identifying high-consequence "escaped defects" where software boundaries and edge cases fail to meet intended requirements.

    # CONTEXT
    You are the **Boundary Evaluator** node in an automated test-generation pipeline. Your goal is to perform a gap analysis between the "Decomposed Specs" (identifying theoretical risks) and the "Summarized Test Suite" (representing the current verification state). You must triage these gaps based on the likelihood of an "escaped defect" a failure that bypasses testing and reaches the production (e.g., clinical) environment.

    # INPUT DATA DESCRIPTION 
    1. <Requirement Statement>: The requirement statement (text form)
    2. <List['DecomposedSpec']>: Focus on `edge_case_analysis`
    3. <TestSuite>: Focus on `protocol` and `acceptance_criteria`
    4. <Project Context/Best Practices>: Optional project-specific information

    # TASK: BOUNDARY ANALYSIS & TRIAGE
    1. **Map Coverage**: Compare each `potential_edge_case` from the [Decomposed Specs] against the existing [Summarized Test Suite]. Identify where a test protocol explicitly exercises the described boundary.
    2. **Identify Gaps**: Highlight edge cases that are not covered.
    3. **Escaped Defect Risk Assessment**: For every missing aspect, evaluate the risk.
        - **High Risk**: Scenarios involving race conditions, resource exhaustion (e.g., memory/storage full), or invalid state transitions that could lead to patient harm or device failure.
        - **Low Risk**: Theoretical edge cases with negligible clinical impact or extremely low probability in production.
    4. **Tool Grounding**: Use available search/document tools (e.g., [Project Context/Best Practices] to verify if specific edge cases (e.g., specific CDN failures or storage persistence issues) are known "escaped defect" patterns in this project's domain.

    # IN-CONTEXT LEARNING EXAMPLES
    - **Example 1 (Escaped Defect)**: 
    - *Requirement*: System must save user logs. 
    - *Current Test*: Verifies logs save when disk is empty. 
    - *Missing Boundary*: Disk Full/Write Error. 
    - *Escaped Defect Rationale*: In a clinical setting, a full disk led to a system hang during a critical procedure because the error handling was never verified.
    - **Example 2 (Low Relevance)**:
    - *Requirement*: UI renders in < 2 seconds.
    - *Missing Boundary*: Rendering during a solar flare event.
    - *Rationale*: While technically a boundary, it is excluded from priority due to near-zero production probability.

    # OUTPUT FORMAT (Pydantic-Compatible JSON)
    Return only the JSON object:
    {
    "covered_boundaries": [
        {
        "spec_id": "string",
        "edge_case_summary": "string",
        "mapped_test_case_id": "string",
        "coverage_rationale": "Explanation of how the test protocol satisfies the edge case logic."
        }
    ],
    "missing_boundaries": [
        {
        "summarized_test_case": {
            "objective": "string",
            "verifies": "string",
            "protocol": "string",
            "acceptance_criteria": "string"
        },
        "gap_description": "Why the current suite fails to address this.",
        "escaped_defect_risk": "High | Medium | Low",
        "rationale": "Cite specific clinical or system risks. Explain if this is a high-risk escaped defect candidate."
        }
    ]
    }
    """
    return BaseEvaluatorNode(
        llm=llm,
        response_model=CoverageEvaluator,
        system_prompt=system_prompt,
    )

def make_functional_coverage_evaluator(llm) -> BaseEvaluatorNode:
    system_prompt = """
    You are a medical device verification specialist evaluating functional coverage.
    TASK: Assess if the test cases cover all functional aspects of the requirement.
    CRITERIA:
    - Are all stated functions/capabilities verified?
    - Are both normal operations and alternate flows tested?
    - Does the test suite address the complete functional scope?

    Return ONLY valid JSON (no markdown, no code blocks) with this structure:
    {
        "covered": str = <Functional elements adequately covered by the test suite>, 
        "missing": str = <Functional elements not clearly covered by test suite>, 
        "rationale": str = <Thought process behind the determination of what was covered and what was missing>, 
    }
    """
    return BaseEvaluatorNode(
        llm=llm,
        response_model=CoverageEvaluator,
        system_prompt=system_prompt,
    )

def make_input_output_coverage_evaluator(llm) -> BaseEvaluatorNode:
    pass

def make_negative_test_coverage_evaluator(llm) -> BaseEvaluatorNode:
    pass

def make_assembler_node(llm) -> Dict: 
    pass

def make_aggregator_node(llm) -> Dict:
    pass
