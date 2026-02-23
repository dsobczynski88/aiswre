import asyncio
from typing import List, Optional, Dict, Any
from langchain_core.runnables import Runnable
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from aiswre.prj_logger import ProjectLogger
from aiswre.components.processors import df_to_prompt_items
#from .core import Requirement, TestCase, MedtechTraceLink, TestCaseState
from .nodes import (
    make_functional_coverage_evaluator,
    make_input_output_coverage_evaluator,
    make_boundary_coverage_evaluator,
    make_negative_test_coverage_evaluator,
    make_decomposer_node, 
    make_summarizer_node, 
    make_assembler_node, 
    make_aggregator_node

)
from .core import (
    RTMReviewState,
    Requirement,
    DecomposedRequirement,
    TestCase,
    TestSuite,
    CoverageEvaluator,
    ReviewComment
)

class RTMReviewerRunnable:
    """
    LangGraph-based RTM reviewer using OpenAI and Anthropic LLMs.

    Evaluates requirement verification coverage against 4 criteria:
    1. Functional
    2. Input/Output
    3. Boundary
    4. Negative Testing
    
    Initially, a requirement and all traced test cases is supplied. The requirement
    is decomposed to testable blocks using the decomposer node. The test cases are summarized
    via the summary node to focus on what is achieved rather than provide the full raw steps. 
    The intent is to get an overall view of what the test case expects to accomplish and how it intends to 
    do so.

    The decomposed requirement and summarized test cases are assembled using the assemble node. This
    node doesn't require an LLM it is simply an organization step to collect the generated inputs.

    The assembled context is then passed (in parallel) to each of the four (4) evaluator nodes. Each
    evaluator node will update the state with an assessment based on its domain expertise. 

    The assessments across the coverage evaluators is then aggregated at the aggregator node. The intent
    of the aggregator node is to reason using the initial inputs and the assessments 
    from the coverage evaluators to provide a refined, actionable recommendation on any additional steps
    needed to update the test suite. 
    """

    def __init__(self, client: ChatOpenAI):

        self.client = client
        self.graph = RTMReviewerRunnable.build_graph(self.client)
    
    
    @staticmethod
    def build_graph(client: ChatOpenAI) -> Runnable:
        """
        Build the LangGraph state machine with 11 parallel evaluators.

        Graph structure:
            START
              ↓
            ┌─────────────────────────────────────────────┐
            │  DECOMPOSER, SUMMARIZER        (parallel)   │
            └─────────────────────────────────────────────┘
              ↓
            ASSEMBLER
              ↓
            ┌─────────────────────────────────────────────┐
            │  Coverage Evaluation           (parallel)   │
            │  - Functional                               │
            │  - I/O                                      │
            │  - Boundary                                 │
            │  - Negative                                 │
            └─────────────────────────────────────────────┘
              ↓
            AGGREGATOR
              ↓
            END
        """
        sg = StateGraph(RTMReviewState)

        ## Define the nodes

        # Decomposer and Summarizer
        decomposer = make_decomposer_node(client)
        summarizer = make_summarizer_node(client)
        
        # Assembler
        assembler = make_assembler_node()

        # Coverage evaluator nodes
        functional = make_functional_coverage_evaluator(client)
        input_output = make_input_output_coverage_evaluator(client)
        boundary = make_boundary_coverage_evaluator(client)
        negative = make_negative_test_coverage_evaluator(client)

        # Aggregator
        aggregator = make_aggregator_node(client)

        ## Register nodes to graph
        sg.add_node("decomposer", decomposer)
        sg.add_node("summarizer", summarizer)
        sg.add_node("assembler", assembler)

        sg.add_node("functional", functional)
        sg.add_node("input_output", input_output)
        sg.add_node("boundary", boundary)
        sg.add_node("negative", negative)

        sg.add_node("aggregator", aggregator)

        # Kick off summarizer in parallel with decomposer
        sg.add_edge(START, "decomposer")
        sg.add_edge(START, "summarizer")

        # Both decomposer and summarizer feed the assembler
        sg.add_edge("decomposer", "assembler")
        sg.add_edge("summarizer", "assembler")

        # Fan-out to coverage evaluators (parallel)
        for evaluator in ("functional", "input_output", "boundary", "negative"):
            sg.add_edge("assembler", evaluator)
            # Each evaluator feeds the aggregator (fan-in)
            sg.add_edge(evaluator, "aggregator")

        # Aggregator concludes the review
        sg.add_edge("aggregator", END)

        return sg.compile()
    
    @staticmethod
    def build_simple_graph(client: ChatOpenAI) -> Runnable:
        """
        Build a simple decomposer -> summarizer -> boundary evaluator graph

        Graph structure:
            START
              ↓
            ┌─────────────────────────────────┐
            │DECOMPOSER, SUMMARIZER (parallel)│
            └─────────────────────────────────┘
              ↓
            ┌─────────────────────────────────┐
            │BOUNDARY EVALUATOR               │
            └─────────────────────────────────┘
            END
        """
        sg = StateGraph(RTMReviewState)

        decomposer = make_decomposer_node(client)
        summarizer = make_summarizer_node(client)
        boundary = make_boundary_coverage_evaluator(client)

        sg.add_node("decomposer", decomposer)
        sg.add_node("summarizer", summarizer)
        sg.add_node("boundary", boundary)

        sg.add_edge(START, "decomposer")
        sg.add_edge(START, "summarizer")

        sg.add_edge("decomposer", "boundary")
        sg.add_edge("summarizer", "boundary")

        sg.add_edge("boundary", END)

        return sg.compile()