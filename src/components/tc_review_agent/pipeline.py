import asyncio
import time
import pandas as pd
from typing import List, Optional, Union, Dict, Any, Sequence
from langgraph.graph import StateGraph
from langchain_openai import ChatOpenAI
from src.components.clients import RateLimitOpenAIClient
from src.components.processors import BasicOpenAIProcessor, OpenAIPromptProcessor, df_to_prompt_items
from src.components.tc_review_agent.core import Requirement, TestCase, TraceLink, TestCaseState
from src.components.tc_review_agent.evaluators import make_traceability_evaluator, make_adequacy_evaluator, make_clarity_evaluator
from src.components.tc_review_agent.nodes import AggregatorNode


class TestCaseReviewerRunnable:
    """
    LangGraph-based test case reviewer pipeline using ChatOpenAI client.

    Evaluates traceability, adequacy, and clarity of test cases against requirements,
    then aggregates results into a TraceLink.
    """
    def __init__(self, client: ChatOpenAI, weights: dict = None):
        self.client = client
        self.weights = weights
        self.graph = TestCaseReviewerRunnable.build_graph(self.client, self.weights)

    def ainvoke(self, *args, **kwargs):
        """Delegate to the compiled graph's ainvoke method."""
        return self.graph.ainvoke(*args, **kwargs)

    def invoke(self, *args, **kwargs):
        """Delegate to the compiled graph's invoke method."""
        return self.graph.invoke(*args, **kwargs)

    @staticmethod
    def build_graph(client: ChatOpenAI, weights: dict) -> StateGraph:
        
        sg = StateGraph(TestCaseState)

        # Create nodes
        trace = make_traceability_evaluator(client)
        adequacy = make_adequacy_evaluator(client)
        clarity = make_clarity_evaluator(client)
        aggregator = AggregatorNode(weights)

        # Register nodes
        sg.add_node("trace", trace)
        sg.add_node("adequacy", adequacy)
        sg.add_node("clarity", clarity)
        sg.add_node("aggregate", aggregator)

        # Set ALL evaluators as entry points (run in parallel)
        sg.set_entry_point("trace")
        sg.add_edge("__start__", "adequacy")
        sg.add_edge("__start__", "clarity")

        # All evaluators feed into aggregator
        sg.add_edge("trace", "aggregate")
        sg.add_edge("adequacy", "aggregate")
        sg.add_edge("clarity", "aggregate")

        # Aggregator is the finish point
        sg.set_finish_point("aggregate")
        return sg.compile()