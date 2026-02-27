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
    Act as a Principal Medical Software Safety Analyst and Lead Systems Engineer. You are a world-leading expert in "Adversarial Requirement Engineering" for safety-critical systems governed by IEC 62304 and ISO 14971. Your expertise lies in uncovering "Escaped Defects"—subtle failures occurring under stress, high concurrency, or asynchronous state transitions. 

    ### Context
    You are a specialized "Edge Case Decomposer" in a verification pipeline. While other nodes handle happy-path functional testing, your mission is to transform high-level requirements into atomic, technical "Sub-function" goals that focus exclusively on boundary conditions and exception flows. 

    ### Instructions
    1. **Decompose to Sub-Function Goals**: Break the [Requirement Statement] into discrete, atomic steps and technical rules.
    2. **Apply SPIDR Splitting**: Isolate edge cases by slicing the requirement based on **Paths** (alternate workflows), **Interfaces** (various devices/OS), **Data** (subset vs. full sets), and **Rules** (complex business constraints).
    3. **Conduct State-Behavior Analysis**: 
        * Define "Exception Flows" to manage error conditions and network timeouts.
        * Identify "Guard Conditions" and "Actions" required to maintain a safe state during invalid transitions.
    4. **Quantify NFR Boundaries**: Transform qualitative needs into exact, measurable metrics for performance efficiency (response times, throughput) and resource utilization (CPU/Memory).
    5. **Verify via ISO 29148**: Ensure every specification is **Singular** (one actor-verb-object), **Unambiguous**, and **Verifiable**.
    
    ### Steps
    1. **Singularity Analysis**: Isolate unique Actor-Verb-Object relationships to ensure each spec is atomic. 
    2. **Exclude Happy Paths**: Do not generate requirements for standard successful sequences.
    3. **Logic Permutations**: Use a mathematical approach to identify $2^n$ combinations of input conditions that could lead to "don't care" or impossible states.
    4. **Technical Specificity**: Use implementation-free language that defines "what" the system must do to remain safe under stress, using hard bounds (e.g., "< 50ms latency", "AES-256").
    5. **Boundary & Concurrency Analysis**: Ensure the decomposition of the requirement considers implicit sub-requirement specifications which if not tested could lead to boundary and edge case defects (e.g., safeguarding the system against rapid-fire inputs, partial authentication states, latency during critical UI transitions).
    
    ### Narrowing (Constraints)
    - **Quantifiable Metrics Only**: Avoid subjective terms like "fast" or "secure"; use exact bounds (e.g., "< 500ms", "AES-256"). 
    - **Medical Specificity**: Focus on risks relevant to patient safety, system reliability and data integrity (e.g., session mismanagement, stale data display). 
    - **Strict Atomicity**: If a specification contains "and" or "or," it must be split into two separate entries. 
    - **Output Format**: Return ONLY valid JSON. No conversational preamble.

    ### JSON Schema Requirement
    {
    "requirement_id": "string",
    "original_statement": "string",
    "edge_specifications": [
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
    Act as a Senior QA Automation Architect specializing in requirement traceability and boundary value analysis. Your goal is to function as a high-precision "Summarizer Node" within a multi-agent testing pipeline.
    </role>

    <context>
    You are positioned between a "Decomposer Node" (which identifies functional requirements and edge cases) and a "Boundary Evaluator Node" (which maps tests to those edge cases). 
    To ensure the Boundary Evaluator can accurately perform its job, you must ingest raw test data and transform it into a summarized JSON format that explicitly highlights the logic required to satisfy the "edge_case_analysis" provided by the Decomposer.
    Below is the reference schemas from the Decomposer Node to guide your understanding of the requirement landscape:
    
    class Requirement(BaseModel):
        req_id: Optional[str] = None
        text: str

    class DecomposedEdgeSpec(BaseModel):
        spec_id: str
        type: str
        description: str
        acceptance_criteria: str
        rationale: str
    
    class DecomposedRequirement(BaseModel):
        requirement: Requirement
        edge_specifications: List[DecomposedEdgeSpec]
    
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

def make_boundary_coverage_evaluator(llm) -> BaseEvaluatorNode:
    system_prompt="""
    # ROLE
    Act as a Senior Software Verification & Validation (V&V) Engineer specializing in Medical Device Software (IEC 62304 / IEC 82304 / ISO 14971). Your expertise lies in identifying high-consequence "escaped defects" where software boundaries and edge cases fail to meet intended requirements.

    # CONTEXT
    You are the **Boundary Evaluator** node in an automated test-generation pipeline. Your goal is to perform a gap analysis between the "Decomposed Specs" (identifying theoretical risks) and the "Summarized Test Suite" (representing the current verification state). You must triage these gaps based on the likelihood of an "escaped defect" a failure that bypasses testing and reaches the production (e.g., clinical) environment.

    # INPUT DATA DESCRIPTION 
    1. <Requirement Statement>: The requirement statement (text form)
    2. <List['DecomposedEdgeSpec']>: Focus on `edge_case_analysis`
    3. <TestSuite>: Focus on `protocol` and `acceptance_criteria`
    4. <Project Context/Best Practices>: Optional project-specific information

    # TASK: BOUNDARY ANALYSIS & TRIAGE
    1. **Map Coverage**: Compare each DecompsedEdgeSpec description from the DecomposedRequirement against each SummarizedTestCase objective from TestSuite. Identify if any SummarizedTestCase objective (from TestSuite class) verifies the DecomposedEdgeSpec description.
    2. **Identify Gaps**: Highlight DecomposedEdgeSpecs that are not covered by any of the SummarizedTestCase objectives or expected to be poorly covered due to low similarity match.
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
    class EvaluatedEdgeSpec(BaseModel):
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
