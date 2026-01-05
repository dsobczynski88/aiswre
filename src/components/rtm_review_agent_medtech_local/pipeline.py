"""
RTM Review Pipeline (Ollama/Local version)

LangGraph-based workflow for evaluating requirement verification coverage
in requirement traceability matrices using local Ollama models.
"""

import asyncio
import logging
from typing import List, Optional, Dict, Any
from langgraph.graph import StateGraph, START, END

from .core import Requirement, RTMEntry, RTMReviewLink, RTMState
from .evaluators import (
    make_functional_coverage_evaluator,
    make_input_output_coverage_evaluator,
    make_boundary_coverage_evaluator,
    make_negative_test_coverage_evaluator,
    make_risk_coverage_evaluator,
    make_traceability_completeness_evaluator,
    make_acceptance_criteria_coverage_evaluator,
    make_test_sufficiency_evaluator,
    make_gap_analysis_evaluator
)
from .nodes import create_aggregator_node, convert_evaluator_responses_to_rtm_links


def detect_ollama_ports(
    base_port: int = 11434,
    max_ports: int = 10,
    host: str = "localhost"
) -> List[str]:
    """
    Detect active Ollama instances by checking which ports respond to API calls.

    Args:
        base_port: Starting port to check (default: 11434)
        max_ports: Maximum number of ports to check (default: 10)
        host: Hostname to check (default: localhost)

    Returns:
        List of URLs for active Ollama instances (e.g., ["http://localhost:11434", ...])
    """
    import urllib.request
    import urllib.error

    active_ports = []

    for i in range(max_ports):
        port = base_port + i
        url = f"http://{host}:{port}/api/version"

        try:
            req = urllib.request.Request(url, method='GET')
            with urllib.request.urlopen(req, timeout=1) as response:
                if response.status == 200:
                    base_url = f"http://{host}:{port}"
                    active_ports.append(base_url)
                    logging.info(f"Found active Ollama instance at {base_url}")
        except (urllib.error.URLError, OSError, TimeoutError):
            pass

    if not active_ports:
        logging.warning(
            f"No active Ollama instances found on {host}:{base_port}-{base_port + max_ports - 1}. "
            f"Make sure Ollama is running."
        )
        return [f"http://{host}:{base_port}"]

    return active_ports


class RTMLocalReviewerRunnable:
    """
    LangGraph-based RTM reviewer using Ollama models.

    Evaluates requirement verification coverage against 9 criteria organized into 3 categories:
    1. Coverage (4 evaluators): Functional, Input/Output, Boundary, Negative Testing
    2. Risk & Traceability (3 evaluators): Risk, Traceability, Acceptance Criteria
    3. Sufficiency & Gaps (2 evaluators): Test Sufficiency, Gap Analysis

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
        Initialize the RTM Ollama reviewer pipeline.

        Args:
            model: Ollama model name (e.g., "llama3.1", "mistral", "qwen2.5")
            base_url: Ollama server URL
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
        Build the LangGraph with 9 parallel evaluators.

        Returns:
            Compiled LangGraph workflow
        """
        # Create all 9 evaluator nodes
        eval_functional = make_functional_coverage_evaluator(self.model, self.base_url, self.temperature)
        eval_input_output = make_input_output_coverage_evaluator(self.model, self.base_url, self.temperature)
        eval_boundary = make_boundary_coverage_evaluator(self.model, self.base_url, self.temperature)
        eval_negative = make_negative_test_coverage_evaluator(self.model, self.base_url, self.temperature)

        eval_risk = make_risk_coverage_evaluator(self.model, self.base_url, self.temperature)
        eval_traceability = make_traceability_completeness_evaluator(self.model, self.base_url, self.temperature)
        eval_acceptance = make_acceptance_criteria_coverage_evaluator(self.model, self.base_url, self.temperature)

        eval_sufficiency = make_test_sufficiency_evaluator(self.model, self.base_url, self.temperature)
        eval_gap = make_gap_analysis_evaluator(self.model, self.base_url, self.temperature)

        # Create conversion middleware node
        def convert_node(state: dict) -> dict:
            """Convert raw evaluator responses to RTMReviewLink objects."""
            review_links = convert_evaluator_responses_to_rtm_links(state)
            return {"rtm_links": review_links}

        # Create aggregator
        aggregator = create_aggregator_node(
            model=self.model,
            base_url=self.base_url,
            temperature=self.temperature,
            weights=self.weights
        )

        # Build graph
        graph = StateGraph(RTMState)

        # Add evaluator nodes
        graph.add_node("functional_coverage", eval_functional)
        graph.add_node("input_output_coverage", eval_input_output)
        graph.add_node("boundary_coverage", eval_boundary)
        graph.add_node("negative_test_coverage", eval_negative)
        graph.add_node("risk_coverage", eval_risk)
        graph.add_node("traceability_completeness", eval_traceability)
        graph.add_node("acceptance_criteria_coverage", eval_acceptance)
        graph.add_node("test_sufficiency", eval_sufficiency)
        graph.add_node("gap_analysis", eval_gap)

        # Add conversion and aggregation nodes
        graph.add_node("convert", convert_node)
        graph.add_node("aggregate", aggregator)

        # Parallel execution: all 9 evaluators start from START
        graph.add_edge(START, "functional_coverage")
        graph.add_edge(START, "input_output_coverage")
        graph.add_edge(START, "boundary_coverage")
        graph.add_edge(START, "negative_test_coverage")
        graph.add_edge(START, "risk_coverage")
        graph.add_edge(START, "traceability_completeness")
        graph.add_edge(START, "acceptance_criteria_coverage")
        graph.add_edge(START, "test_sufficiency")
        graph.add_edge(START, "gap_analysis")

        # All evaluators feed into conversion node
        graph.add_edge("functional_coverage", "convert")
        graph.add_edge("input_output_coverage", "convert")
        graph.add_edge("boundary_coverage", "convert")
        graph.add_edge("negative_test_coverage", "convert")
        graph.add_edge("risk_coverage", "convert")
        graph.add_edge("traceability_completeness", "convert")
        graph.add_edge("acceptance_criteria_coverage", "convert")
        graph.add_edge("test_sufficiency", "convert")
        graph.add_edge("gap_analysis", "convert")

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


def get_rtm_local_reviewer_runnable(
    model: str = "llama3.1",
    base_url: str = "http://localhost:11434",
    temperature: float = 0.0,
    weights: Optional[dict] = None
) -> RTMLocalReviewerRunnable:
    """
    Factory function to create an RTM local reviewer.

    Args:
        model: Ollama model name
        base_url: Ollama server URL
        temperature: LLM temperature
        weights: Optional score weights for aggregation

    Returns:
        RTMLocalReviewerRunnable instance
    """
    return RTMLocalReviewerRunnable(
        model=model,
        base_url=base_url,
        temperature=temperature,
        weights=weights
    )


async def run_batch_rtm_local_review(
    rtm_entries: List[RTMEntry],
    requirements: List[Requirement],
    model: str = "llama3.1",
    base_urls: Optional[List[str]] = None,
    temperature: float = 0.0,
    weights: Optional[dict] = None,
    max_concurrent: int = 3,
    auto_detect_ports: bool = True
) -> List[RTMReviewLink]:
    """
    Run batch review of RTM entries using Ollama with multi-port support.

    Args:
        rtm_entries: List of RTMEntry objects (req + test case summary)
        requirements: List of Requirement objects (matched by req_id)
        model: Ollama model name
        base_urls: List of Ollama URLs for multi-port execution
                   If None and auto_detect_ports=True, will auto-detect active instances
        temperature: LLM temperature
        weights: Optional score weights for aggregation
        max_concurrent: Maximum concurrent reviews
        auto_detect_ports: If True and base_urls is None, automatically detect Ollama ports

    Returns:
        List of RTMReviewLink results
    """
    if base_urls is None:
        if auto_detect_ports:
            logging.info("Auto-detecting active Ollama instances...")
            base_urls = detect_ollama_ports()
            logging.info(f"Found {len(base_urls)} active Ollama instance(s): {base_urls}")
        else:
            base_urls = ["http://localhost:11434"]

    results = []

    # Create requirement lookup
    req_dict = {req.req_id: req for req in requirements}

    # Create semaphore to limit concurrency
    semaphore = asyncio.Semaphore(max_concurrent)

    async def review_single_rtm_entry(
        rtm_entry: RTMEntry,
        port_index: int
    ) -> RTMReviewLink:
        """Review a single RTM entry using assigned Ollama port."""
        async with semaphore:
            # Select Ollama port in round-robin fashion
            base_url = base_urls[port_index % len(base_urls)]

            logging.info(f"Reviewing {rtm_entry.req_id} on {base_url}")

            # Get requirement
            req = req_dict.get(rtm_entry.req_id)
            if not req:
                logging.warning(f"Requirement {rtm_entry.req_id} not found in requirements list")
                return RTMReviewLink(
                    req_id=rtm_entry.req_id,
                    rationale="Requirement not found",
                    link_type="Error",
                    issues=["Requirement not provided"]
                )

            # Create reviewer for this port
            reviewer = get_rtm_local_reviewer_runnable(
                model=model,
                base_url=base_url,
                temperature=temperature,
                weights=weights
            )

            # Build initial state
            state = {
                "requirement": req,
                "rtm_entry": rtm_entry,
                "raw_evaluator_responses": [],
                "rtm_links": [],
                "final_result": None
            }

            try:
                # Run async review
                result = await reviewer.ainvoke(state)
                final = result.get("final_result")

                if final:
                    avg_score = sum([
                        final.functional_coverage_score,
                        final.input_output_coverage_score,
                        final.boundary_coverage_score,
                        final.negative_test_coverage_score,
                        final.risk_coverage_score,
                        final.traceability_completeness_score,
                        final.acceptance_criteria_coverage_score,
                        final.test_sufficiency_score,
                        final.gap_analysis_score
                    ]) / 9

                    logging.info(
                        f"Completed {rtm_entry.req_id} on {base_url}: "
                        f"avg_score={avg_score:.2f}"
                    )
                    return final
                else:
                    logging.warning(f"No final result for {rtm_entry.req_id}")
                    return RTMReviewLink(
                        req_id=rtm_entry.req_id,
                        rationale="Review failed to produce result",
                        link_type="Error",
                        issues=["No result returned"]
                    )

            except Exception as e:
                logging.error(f"Failed to review {rtm_entry.req_id} on {base_url}: {e}")
                return RTMReviewLink(
                    req_id=rtm_entry.req_id,
                    rationale=f"Review failed: {str(e)}",
                    link_type="Error",
                    issues=[str(e)]
                )

    # Create tasks for all RTM entries
    tasks = []
    for idx, rtm_entry in enumerate(rtm_entries):
        tasks.append(review_single_rtm_entry(rtm_entry, idx))

    # Execute all reviews concurrently
    results = await asyncio.gather(*tasks)

    return results
