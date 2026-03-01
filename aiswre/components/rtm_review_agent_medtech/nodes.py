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
    ReviewComment,
    AITestSuite
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


class TestGeneratorNode:

    def __init__(self, llm, response_model, system_prompt):
        self.llm=llm
        self.response_model=response_model
        self.structured_llm = llm.with_structured_output(response_model)
        self.system_prompt=system_prompt

    @staticmethod
    def _build_payload(
        decomposed_requirement: DecomposedRequirement, 
        test_suite: TestSuite
        ) -> dict:
        
        payload = {
            "decomposed_requirement": decomposed_requirement.model_dump(),
            "test_suite": test_suite.model_dump(),
        }
        return payload
    
    async def __call__(self, state: Dict) -> Dict:
        decomposed_requirement = state.get("decomposed_requirement")
        test_suite = state.get("test_suite")
        # Build payload for LLM
        payload = self._build_payload(decomposed_requirement, test_suite)

        try:
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=json.dumps(payload))
            ]
            # Use structured output to get Pydantic model directly
            parsed = await self.structured_llm.ainvoke(messages)
        except Exception as e:
            print(e)
            
        return {"ai_test_suite": parsed}


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
        ai_test_suite: AITestSuite
        ) -> dict:

        payload = {
            "original_requirement": requirement.model_dump(),
            "decomposed_requirement": decomposed_requirement.model_dump(),
            "ai_test_suite": ai_test_suite.model_dump(),
        }
        return payload

    async def __call__(self, state: Dict) -> Dict:
        original_requirement = state.get("requirement")
        decomposed_requirement = state.get("decomposed_requirement")
        ai_test_suite = state.get("ai_test_suite")
        # Build payload for LLM
        payload = self._build_payload(original_requirement, decomposed_requirement, ai_test_suite)

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
    You are a specialized "Requirement Decomposer" in a verification pipeline. Your mission is to transform high-level (potentially ambiguous) requirements into atomic, technical "Sub-function" goals that comprehensively cover happy-path functional requirement expectations and any implicit sub-functional goals that are reasonably inferred from the input [Requirement Statement] 

    ### Instructions
    1. **Decompose to Sub-Function Goals**: Break the [Requirement Statement] into discrete, atomic steps and technical rules.
    2. **Apply SPIDR Splitting**: Isolate atomic steps by slicing the requirement based on **Paths** (alternate workflows), **Interfaces** (various devices/OS), **Data** (subset vs. full sets), and **Rules** (complex business constraints).
    3. **Standardize**: Ensure every decomposed specification meets ISO 29148 characteristics (Unambiguous, Singular, Verifiable).
    4. **Conduct State-Behavior Analysis**: 
        * Define "Exception Flows" to manage error conditions and network timeouts.
        * Identify "Guard Conditions" and "Actions" required to maintain a safe state during invalid transitions.
    5. **Quantify NFR Boundaries**: Transform the [Requirement Statement] into exact, measurable metrics for performance and usability.
    
    ### Steps
    1. **Singularity Analysis**: Isolate unique Actor-Verb-Object relationships to ensure each spec is atomic. 
    2. **Flow Mapping**: Define the "Happy Path" and identify "Alternative" and "Exception" flows. 
    3. **Logic Permutations**: Use a mathematical approach to identify $2^n$ combinations of input conditions that could lead to "don't care" or impossible states.
    4. **Technical Specificity**: Use implementation-free language that defines "what" the system must do to remain safe under stress, using hard bounds (e.g., "< 50ms latency", "AES-256").
    
    ### Narrowing (Constraints)
    - **Quantifiable Metrics Only**: Avoid subjective terms like "fast" or "secure"; use exact bounds (e.g., "< 500ms", "AES-256"). 
    - **Medical Specificity**: Focus on risks relevant to patient safety, system reliability and data integrity (e.g., session mismanagement, stale data display). 
    - **Strict Atomicity**: If a specification contains "and" or "or," it must be split into two separate entries. 
    - **Denote Assumptions**: If exact bounds are proposed which are not part of the original requirement, denote this within bracketed text ("[]") 
    - **Output Format**: Return ONLY valid JSON. No conversational preamble.

    ### JSON Schema Requirement
    {
    "requirement_id": "string",
    "original_statement": "string",
    "decomposed_specifications": [
        {
        "spec_id": "string",
        "type": "functional | performance | safety | security",
        "description": "string (description of the requirement sub-function statement which is singular and atomic)",
        "acceptance_criteria": "string (measurable/quantifiable evidence that would demonstrate this requirement sub-function is effectively tested)",
        "rationale": "string (describes why this spec was identified based on the input [Requirement Statement])",
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
    Act as a Senior QA Automation Architect specializing in requirement traceability. Your goal is to function as a high-precision "Summarizer Node" within a multi-agent testing pipeline.
    </role>

    <context>
    You are positioned between a "Decomposer Node" (which systematically decomposes requirements) and an "Evaluator Node" (which maps decomposed requirement specs to test cases to perform coverage analysis). 
    To ensure the Evaluator can accurately perform its job, you must ingest raw test data and transform it into the below JSON format.
    Below is the reference schemas from the Decomposer Node to guide your understanding of the requirement landscape:
    
    class Requirement(BaseModel):
        req_id: Optional[str] = None
        text: str

    class DecomposedSpec(BaseModel):
        spec_id: str
        type: str
        description: str
        acceptance_criteria: str
        rationale: str
    
    class DecomposedRequirement(BaseModel):
        requirement: Requirement
        decomposed_specifications: List[DecomposedSpec]
    
    Below is the reference pydantic model of a single test case that needs to be summarized. The user will provide a list of the data class TestCase:  

    class TestCase(BaseModel):
        test_id: str
        description: str (A high-level objective statement for the test case)
        setup: str (A description of the pre-requisite steps to run the test case)
        steps: str (Step-by-step actions for the described test case)
        expectedResults: str (Step-by-step expected results for the described test case)
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
    class SummarizedTestCase(BaseModel):
        test_case_id: str
        objective: str
        verifies: str
        protocol: List[str]
        acceptance_criteria: List[str]

    class TestSuite(BaseModel):
        requirement: Requirement
        test_cases: List[TestCase]
        summary: List[SummarizedTestCase]
    
    This node shall return the following data class structure TestSuite:
    
    class TestSuite(BaseModel):
        requirement: Requirement
        test_cases: List[TestCase]
        summary: List[SummarizedTestCase]
    </format_template>
    """
    return SummaryNode(
        llm=llm,
        response_model=TestSuite,
        system_prompt=system_prompt,
    )

def make_generator_node(llm) -> TestGeneratorNode:
    system_prompt="""
    # ROLE
    You are a Lead Medical Device Software Verification Engineer specializing in IEC 62304 (Software Lifecycle) and ISO 14971 (Risk Management). Your expertise is "Adversarial Testing"—specifically finding "escaped defects" that standard functional tests miss, such as race conditions, memory corruption, and boundary-logic failures.

    # INPUT DATA
    You will be provided with the following data:
    1. **DecomposedRequirement**: A requirement object containing a list of atomic specifications (`decomposed_specifications`).
    2. **TestSuite**: The current suite of existing tests (`summary`) already mapped to this requirement.

    # TASK (TAG Framework)
    Analyze the gap between the `decomposed_specifications` and the `TestSuite`. Generate a single `AITestSuite` object that identifies high-risk scenarios (Negative, Boundary, and Stress tests) designed to catch defects that would otherwise escape to production.

    # EXECUTION STEPS (COGNITIVE SCAFFOLDING)
    1. **Semantic Parsing**: Break down each `DecomposedSpec` to identify hidden variables, state dependencies, and timing constraints.
    2. **Gap Analysis**: Compare the existing `TestSuite.summary` against the specifications. Determine what is NOT being tested (e.g., are there tests for "just outside" the boundary? are there tests for "invalid state" interruptions?).
    3. **Adversarial Brainstorming**: Specifically target:
        - **Boundary Value Analysis (BVA)**: Values at Min-1, Max+1, or precisely on the limit.
        - **Temporal/Race Conditions**: Interrupting a process while this specification is active.
        - **Resource Constraints**: How the logic behaves during low battery or memory pressure.
    4. **Union & Synthesis**: Combine the existing `current_test_suite` with your new `generated_tests` to create the final `ai_test_suite`.
    5. **Rationale Formulation**: Write a technical justification explaining why these new tests are necessary for medical safety and why the original suite was insufficient for catching these specific escaped defects.

    # EXPECTED OUTPUT (STRICT PYDANTIC SCHEMA)
    Return exactly ONE JSON object matching the `AITestSuite` class. Do not return a list. Do not include conversational filler.
    The generated tests key shall contain the list of all generated tests from this prompt.

    {
        "spec_id": "The primary spec_id or requirement ID being addressed",
        "current_test_suite": [/* List of original SummarizedTestCase objects from input */],
        "generated_tests": [
            {
                "test_case_id": "TC-ADV-XXXX",
                "objective": "Identify [Specific Escaped Defect Type]",
                "verifies": "The specific logic/boundary being challenged",
                "protocol": ["Step 1...", "Step 2..."],
                "acceptance_criteria": ["Rigorous safety-critical result"]
            }
        ],
        "ai_test_suite": [/* Full merged list: current_test_suite + generated_tests */],
        "rationale": "Deep technical reasoning on the identified gaps and how the generated tests prevent escaped defects."
    }
    """
    return TestGeneratorNode(
        llm=llm,
        response_model=AITestSuite,
        system_prompt=system_prompt,
    )

def make_coverage_evaluator(llm) -> BaseEvaluatorNode:
    system_prompt="""
    # ROLE
    Act as a Senior Software Verification & Validation (V&V) Engineer specializing in Medical Device Software (IEC 62304 / IEC 82304 / ISO 14971). Your expertise lies in identifying high-consequence "escaped defects" where software test suites fail to meet intended requirements.

    # CONTEXT
    You are the **Evaluator** node in an automated test-generation pipeline. Your goal is to perform a gap analysis between the "Decomposed Specs" (identifying theoretical risks) and the "AI Test Suite" (representing the current verification state). You must triage these gaps based on the likelihood of an "escaped defect" a failure that bypasses testing and reaches the production (e.g., clinical) environment.

    # INPUT DATA DESCRIPTION 
    1. <Requirement Statement>: The requirement statement (text form)
    2. <List['DecomposedEdgeSpec']>: Focus on `edge_case_analysis`
    3. <AITestSuite>: Focus on `protocol` and `acceptance_criteria`
    4. <Project Context/Best Practices>: Optional project-specific information

    # TASK: BOUNDARY ANALYSIS & TRIAGE
    1. **Map Coverage**: Compare each DecompsedSpec description from the DecomposedRequirement against each SummarizedTestCase objective from AITestSuite. Identify if any SummarizedTestCase objective (from AITestSuite class) verifies the DecomposedSpec description.
    2. **Identify Gaps**: Highlight DecomposedSpecs that are not covered by any of the SummarizedTestCase objectives or expected to be poorly covered due to low similarity match.
    3. **Escaped Defect Risk Assessment**: For every missing aspect, evaluate the risk.
        - **High Risk**: Scenarios involving race conditions, resource exhaustion (e.g., memory/storage full), or invalid state transitions that could lead to patient harm or device failure.
        - **Low Risk**: Theoretical edge cases with negligible clinical impact or extremely low probability in production.
    4. **Tool Grounding**: Use available search/document tools (e.g., [Project Context/Best Practices] to verify if specific edge cases (e.g., specific CDN failures or storage persistence issues) are known "escaped defect" patterns in this project's domain and to evaluate whether it is meaningful edge case to test.

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
    Return a list of the following pydantic class (each element in list corresponds to each EdgeCaseSpec):
    class EvaluatedSpec(BaseModel):
        spec_id: str = Field(..., description="The spec_id from the DecomposedEdgeSpec")
        covered_exists: bool = Field(..., description="True if coverage exists in at least one test case of input TestSuite otherwise False")
        covered_by_test_cases: List[str] = Field(..., description="A list of test case IDs from TestSuite['summary'] that effectively cover the test. In the event no test cases are covered, this should return as an empty list.")
        rationale: str = Field(..., description="Thought process behind the determination of whether the existing test cases within TestSuite cover or fail to cover the described EdgeCaseSpec")
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
