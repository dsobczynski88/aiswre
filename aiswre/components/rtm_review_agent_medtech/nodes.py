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
        decomposed_requirement: DecomposedRequirement, test_suite: TestSuite) -> dict:
        pass
    
    async def __call__(self, state: Dict) -> Dict:
        requirement = state.get("decomposed_requirement")
        test_suite = state.get("test_suite")
        # Build payload for LLM
        payload = self._build_payload(requirement, test_suite)

        try:
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=json.dumps(payload))
            ]
            # Use structured output to get Pydantic model directly
            parsed = await self.structured_llm.ainvoke(messages)
        except Exception as e:
            print(e)
            
        # Return only the new trace link (not accumulated list)
        # LangGraph will use operator.add to merge with other parallel node outputs
        return {"coverage_responses": self.response_model}
        

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

def make_boundary_coverage_evaluator(llm) -> BaseEvaluatorNode:
    pass

def make_negative_test_coverage_evaluator(llm) -> BaseEvaluatorNode:
    pass

def make_decomposer_node(llm) -> DecomposerNode:
    system_prompt = """
    Act as a verification expert and break down the requirement into testable elements...
    """
    return DecomposerNode(
        llm=llm,
        response_model=DecomposedRequirement,
        system_prompt=system_prompt,
    )

def make_summarizer_node(llm) -> SummaryNode:
    system_prompt = """
    You are a test case analysis expert. Summarize each test case into its core objective,
    type, key steps, and expected results. Focus on what is being verified rather than
    raw procedural detail.

    Return structured JSON for each test case with:
    - test_id: the test case identifier
    - objective: a concise statement of what is being verified
    - type: the category of test (e.g., boundary, nominal, negative, performance)
    - steps: list of key verification steps
    - expectedResults: list of expected outcomes
    """
    return SummaryNode(
        llm=llm,
        response_model=TestSuite,
        system_prompt=system_prompt,
    )

def make_assembler_node(llm) -> Dict: 
    pass

def make_aggregator_node(llm) -> Dict:
    pass