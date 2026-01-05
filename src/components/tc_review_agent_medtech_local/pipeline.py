"""
MedTech Test Case Review Pipeline (Ollama/Local version)

LangGraph-based workflow for evaluating medical device software test cases
against FDA/IEC 62304 best practices using local Ollama models.
"""

import asyncio
import logging
from typing import List, Optional, Dict, Any
from langgraph.graph import StateGraph, START, END

from .core import Requirement, TestCase, MedtechTraceLink, TestCaseState
from .evaluators import (
    make_unambiguity_evaluator,
    make_independence_evaluator,
    make_preconditions_evaluator,
    make_postconditions_evaluator,
    make_technique_application_evaluator,
    make_negative_testing_evaluator,
    make_boundary_checks_evaluator,
    make_risk_verification_evaluator,
    make_traceability_evaluator,
    make_safety_class_rigor_evaluator,
    make_objective_evidence_evaluator
)
from .nodes import create_aggregator_node, convert_evaluator_responses_to_tracelinks


class MedtechLocalTestCaseReviewerRunnable:
    """
    LangGraph-based medtech test case reviewer using Ollama models.

    Evaluates test cases against 11 FDA/IEC 62304 criteria organized into 3 categories:
    1. General Integrity & Structure (4 evaluators)
    2. Coverage & Technique (4 evaluators)
    3. Traceability & Compliance (3 evaluators)

    All evaluators run in parallel, then results are aggregated using Ollama LLM.
    """

    def __init__(
        self,
        model: str = "llama3.1",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.0,
        weights: Optional[dict] = None
    ):
        """
        Initialize the medtech Ollama reviewer pipeline.

        Args:
            model: Ollama model name (e.g., "llama3.1", "mistral", "qwen2.5")
            base_url: Ollama server URL (for multi-port: http://localhost:11434, :11435, etc.)
            temperature: LLM temperature
            weights: Optional dict of score weights for aggregation
        """
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.weights = weights
        self.graph = self.build_graph()

    def build_graph(self) -> Any:
        """
        Build the LangGraph with 11 parallel evaluators.

        Returns:
            Compiled LangGraph workflow
        """
        # Create all 11 evaluator nodes
        eval_unambiguity = make_unambiguity_evaluator(self.model, self.base_url, self.temperature)
        eval_independence = make_independence_evaluator(self.model, self.base_url, self.temperature)
        eval_preconditions = make_preconditions_evaluator(self.model, self.base_url, self.temperature)
        eval_postconditions = make_postconditions_evaluator(self.model, self.base_url, self.temperature)

        eval_technique = make_technique_application_evaluator(self.model, self.base_url, self.temperature)
        eval_negative = make_negative_testing_evaluator(self.model, self.base_url, self.temperature)
        eval_boundary = make_boundary_checks_evaluator(self.model, self.base_url, self.temperature)
        eval_risk = make_risk_verification_evaluator(self.model, self.base_url, self.temperature)

        eval_traceability = make_traceability_evaluator(self.model, self.base_url, self.temperature)
        eval_safety = make_safety_class_rigor_evaluator(self.model, self.base_url, self.temperature)
        eval_evidence = make_objective_evidence_evaluator(self.model, self.base_url, self.temperature)

        # Create conversion middleware node
        def convert_node(state: dict) -> dict:
            """Convert raw evaluator responses to MedtechTraceLink objects."""
            raw_links = state.get('medtech_links', [])
            if raw_links and isinstance(raw_links[0], dict):
                # Convert raw responses to proper trace links
                trace_links = convert_evaluator_responses_to_tracelinks(state, raw_links)
                return {"medtech_links": trace_links}
            return {"medtech_links": raw_links}

        # Create aggregator
        aggregator = create_aggregator_node(
            model=self.model,
            base_url=self.base_url,
            temperature=self.temperature,
            weights=self.weights
        )

        # Build graph
        graph = StateGraph(TestCaseState)

        # Add evaluator nodes
        graph.add_node("unambiguity", eval_unambiguity)
        graph.add_node("independence", eval_independence)
        graph.add_node("preconditions", eval_preconditions)
        graph.add_node("postconditions", eval_postconditions)
        graph.add_node("technique_application", eval_technique)
        graph.add_node("negative_testing", eval_negative)
        graph.add_node("boundary_checks", eval_boundary)
        graph.add_node("risk_verification", eval_risk)
        graph.add_node("traceability", eval_traceability)
        graph.add_node("safety_class_rigor", eval_safety)
        graph.add_node("objective_evidence", eval_evidence)

        # Add conversion and aggregation nodes
        graph.add_node("convert", convert_node)
        graph.add_node("aggregate", aggregator)

        # Parallel execution: all 11 evaluators start from START
        graph.add_edge(START, "unambiguity")
        graph.add_edge(START, "independence")
        graph.add_edge(START, "preconditions")
        graph.add_edge(START, "postconditions")
        graph.add_edge(START, "technique_application")
        graph.add_edge(START, "negative_testing")
        graph.add_edge(START, "boundary_checks")
        graph.add_edge(START, "risk_verification")
        graph.add_edge(START, "traceability")
        graph.add_edge(START, "safety_class_rigor")
        graph.add_edge(START, "objective_evidence")

        # All evaluators feed into conversion node
        graph.add_edge("unambiguity", "convert")
        graph.add_edge("independence", "convert")
        graph.add_edge("preconditions", "convert")
        graph.add_edge("postconditions", "convert")
        graph.add_edge("technique_application", "convert")
        graph.add_edge("negative_testing", "convert")
        graph.add_edge("boundary_checks", "convert")
        graph.add_edge("risk_verification", "convert")
        graph.add_edge("traceability", "convert")
        graph.add_edge("safety_class_rigor", "convert")
        graph.add_edge("objective_evidence", "convert")

        # Conversion feeds into aggregator
        graph.add_edge("convert", "aggregate")

        # Aggregator is the end
        graph.add_edge("aggregate", END)

        return graph.compile()

    async def ainvoke(self, state: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Async invocation of the graph."""
        return await self.graph.ainvoke(state, **kwargs)

    def invoke(self, state: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Sync invocation of the graph."""
        return self.graph.invoke(state, **kwargs)


def get_medtech_local_reviewer_runnable(
    model: str = "llama3.1",
    base_url: str = "http://localhost:11434",
    temperature: float = 0.0,
    weights: Optional[dict] = None
) -> MedtechLocalTestCaseReviewerRunnable:
    """
    Factory function to create a medtech local test case reviewer.

    Args:
        model: Ollama model name
        base_url: Ollama server URL (for multi-port support)
        temperature: LLM temperature
        weights: Optional score weights for aggregation

    Returns:
        MedtechLocalTestCaseReviewerRunnable instance
    """
    return MedtechLocalTestCaseReviewerRunnable(
        model=model,
        base_url=base_url,
        temperature=temperature,
        weights=weights
    )


async def run_batch_medtech_local_test_case_review(
    test_cases: List[TestCase],
    requirements: Optional[List[Requirement]] = None,
    model: str = "llama3.1",
    base_urls: Optional[List[str]] = None,
    temperature: float = 0.0,
    weights: Optional[dict] = None,
    max_concurrent: int = 3
) -> List[MedtechTraceLink]:
    """
    Run batch review of test cases against medtech checklist using Ollama with multi-port support.

    Args:
        test_cases: List of TestCase objects
        requirements: Optional list of Requirement objects (matched by index)
        model: Ollama model name
        base_urls: List of Ollama URLs for multi-port execution
                   (e.g., ["http://localhost:11434", "http://localhost:11435"])
                   If None, uses single instance at default port
        temperature: LLM temperature
        weights: Optional score weights for aggregation
        max_concurrent: Maximum concurrent reviews

    Returns:
        List of MedtechTraceLink results
    """
    if base_urls is None:
        base_urls = ["http://localhost:11434"]

    results = []

    # Create semaphore to limit concurrency
    semaphore = asyncio.Semaphore(max_concurrent)

    async def review_single_test_case(
        test: TestCase,
        req: Optional[Requirement],
        port_index: int
    ) -> MedtechTraceLink:
        """Review a single test case using assigned Ollama port."""
        async with semaphore:
            # Select Ollama port in round-robin fashion
            base_url = base_urls[port_index % len(base_urls)]

            logging.info(f"Reviewing {test.test_id} on {base_url}")

            # Create reviewer for this port
            reviewer = get_medtech_local_reviewer_runnable(
                model=model,
                base_url=base_url,
                temperature=temperature,
                weights=weights
            )

            # Build initial state
            state = {
                "test": test,
                "requirement": req,
                "medtech_links": [],
                "final_result": None
            }

            try:
                # Run async review
                result = await reviewer.ainvoke(state)
                final = result.get("final_result")

                if final:
                    logging.info(
                        f"Completed {test.test_id} on {base_url}: "
                        f"avg_score={sum([final.unambiguity_score, final.independence_score, final.preconditions_score, final.postconditions_score, final.technique_application_score, final.negative_testing_score, final.boundary_checks_score, final.risk_verification_score, final.traceability_score, final.safety_class_rigor_score, final.objective_evidence_score]) / 11:.2f}"
                    )
                    return final
                else:
                    logging.warning(f"No final result for {test.test_id}")
                    # Return fallback result
                    return MedtechTraceLink(
                        req_id=req.req_id if req else "unknown",
                        test_id=test.test_id,
                        rationale="Review failed to produce result",
                        link_type="Error",
                        issues=["No result returned"]
                    )

            except Exception as e:
                logging.error(f"Failed to review {test.test_id} on {base_url}: {e}")
                # Return error result
                return MedtechTraceLink(
                    req_id=req.req_id if req else "unknown",
                    test_id=test.test_id,
                    rationale=f"Review failed: {str(e)}",
                    link_type="Error",
                    issues=[str(e)]
                )

    # Create tasks for all test cases
    tasks = []
    for idx, test in enumerate(test_cases):
        req = requirements[idx] if requirements and idx < len(requirements) else None
        tasks.append(review_single_test_case(test, req, idx))

    # Execute all reviews concurrently
    results = await asyncio.gather(*tasks)

    return results
