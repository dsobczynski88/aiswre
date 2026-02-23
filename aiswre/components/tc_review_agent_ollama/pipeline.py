"""
Ollama Test Case Review Pipeline

LangGraph-based workflow using local Ollama models with multi-port support
for parallel execution via GraphProcessor.
"""

import asyncio
import logging
from typing import List, Optional, Dict, Any
from langgraph.graph import StateGraph, START, END
from .core import (
    TestCaseReviewState,
    TestCaseInput,
    ReviewResult
)
from .evaluators import (
    make_structure_evaluator,
    make_objective_evaluator,
    make_summary_generator
)
from .nodes import create_aggregator_node


class OllamaTestCaseReviewerRunnable:
    """
    LangGraph-based test case reviewer using Ollama models.

    Supports multi-port Ollama instances for parallel execution.

    Graph structure:
        START
          ↓
        ┌─────────────────────────────┐
        │  Parallel Evaluators        │
        │  - eval_structure           │
        │  - eval_objective           │
        └─────────────────────────────┘
          ↓
        get_review_summary
          ↓
        aggregate_results
          ↓
        END
    """

    def __init__(
        self,
        model: str = "llama3.1",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.0
    ):
        """
        Initialize Ollama test case reviewer.

        Args:
            model: Ollama model name (e.g., "llama3.1", "mistral", "qwen2.5")
            base_url: Ollama server URL (allows multi-port: http://localhost:11434, :11435, etc.)
            temperature: LLM temperature
        """
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.graph = self.build_graph()

    def build_graph(self) -> Any:
        """
        Build the LangGraph with parallel evaluators.

        Returns:
            Compiled LangGraph workflow
        """
        # Create evaluator nodes
        eval_structure = make_structure_evaluator(
            model=self.model,
            base_url=self.base_url,
            temperature=self.temperature
        )

        eval_objective = make_objective_evaluator(
            model=self.model,
            base_url=self.base_url,
            temperature=self.temperature
        )

        get_summary = make_summary_generator(
            model=self.model,
            base_url=self.base_url,
            temperature=self.temperature
        )

        aggregate_results = create_aggregator_node()

        # Build graph
        graph = StateGraph(TestCaseReviewState)

        # Add nodes
        graph.add_node("eval_structure", eval_structure)
        graph.add_node("eval_objective", eval_objective)
        graph.add_node("get_review_summary", get_summary)
        graph.add_node("aggregate_results", aggregate_results)

        # Parallel execution: both evaluators start from START
        graph.add_edge(START, "eval_structure")
        graph.add_edge(START, "eval_objective")

        # Both evaluators feed into summary
        graph.add_edge("eval_structure", "get_review_summary")
        graph.add_edge("eval_objective", "get_review_summary")

        # Summary feeds into aggregator
        graph.add_edge("get_review_summary", "aggregate_results")

        # Aggregator is the end
        graph.add_edge("aggregate_results", END)

        return graph.compile()

    async def ainvoke(self, state: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Async invocation of the graph."""
        return await self.graph.ainvoke(state, **kwargs)

    def invoke(self, state: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Sync invocation of the graph."""
        return self.graph.invoke(state, **kwargs)


def get_ollama_reviewer_runnable(
    model: str = "llama3.1",
    base_url: str = "http://localhost:11434",
    temperature: float = 0.0
) -> OllamaTestCaseReviewerRunnable:
    """
    Factory function to create Ollama test case reviewer.

    Args:
        model: Ollama model name
        base_url: Ollama server URL (for multi-port support)
        temperature: LLM temperature

    Returns:
        OllamaTestCaseReviewerRunnable instance
    """
    return OllamaTestCaseReviewerRunnable(
        model=model,
        base_url=base_url,
        temperature=temperature
    )


async def run_batch_ollama_test_case_review(
    test_cases: List[TestCaseInput],
    model: str = "llama3.1",
    base_urls: Optional[List[str]] = None,
    temperature: float = 0.0,
    max_concurrent: int = 3
) -> List[ReviewResult]:
    """
    Run batch review of test cases using Ollama with multi-port support.

    This function creates multiple reviewer instances, each using a different
    Ollama port for true parallel execution.

    Args:
        test_cases: List of TestCaseInput objects
        model: Ollama model name
        base_urls: List of Ollama URLs for multi-port execution
                   (e.g., ["http://localhost:11434", "http://localhost:11435"])
                   If None, uses single instance at default port
        temperature: LLM temperature
        max_concurrent: Maximum concurrent reviews

    Returns:
        List of ReviewResult objects
    """
    if base_urls is None:
        base_urls = ["http://localhost:11434"]

    results = []

    # Create semaphore to limit concurrency
    semaphore = asyncio.Semaphore(max_concurrent)

    async def review_single_test_case(
        test: TestCaseInput,
        port_index: int
    ) -> ReviewResult:
        """Review a single test case using assigned Ollama port."""
        async with semaphore:
            # Select Ollama port in round-robin fashion
            base_url = base_urls[port_index % len(base_urls)]

            logging.info(f"Reviewing {test.test_id} on {base_url}")

            # Create reviewer for this port
            reviewer = get_ollama_reviewer_runnable(
                model=model,
                base_url=base_url,
                temperature=temperature
            )

            # Build initial state
            initial_state = {
                "messages": [],
                "test_case_id": test.test_id,
                "test_case_text": test.test_case_text,
                "structure": "",
                "objective": "",
                "review_summary": "",
                "final_result": None
            }

            try:
                # Run async review
                final_state = await reviewer.ainvoke(initial_state)
                result = final_state.get("final_result")

                if result:
                    logging.info(
                        f"Completed {test.test_id} on {base_url}: "
                        f"score={result.overall_score:.2f}"
                    )
                    return result
                else:
                    logging.warning(f"No final result for {test.test_id}")
                    # Return fallback result
                    return ReviewResult(
                        test_id=test.test_id,
                        structure_verdict="inadequate",
                        structure_gaps=["No result returned"],
                        structure_recommendations=[],
                        objective_verdict="inadequate",
                        objective_gaps=["No result returned"],
                        objective_recommendations=[],
                        review_summary="Review failed to produce result",
                        overall_score=0.0
                    )

            except Exception as e:
                logging.error(f"Failed to review {test.test_id} on {base_url}: {e}")
                # Return error result
                return ReviewResult(
                    test_id=test.test_id,
                    structure_verdict="inadequate",
                    structure_gaps=[f"Error: {str(e)}"],
                    structure_recommendations=[],
                    objective_verdict="inadequate",
                    objective_gaps=[f"Error: {str(e)}"],
                    objective_recommendations=[],
                    review_summary=f"Review failed: {str(e)}",
                    overall_score=0.0
                )

    # Create tasks for all test cases
    tasks = [
        review_single_test_case(test, idx)
        for idx, test in enumerate(test_cases)
    ]

    # Execute all reviews concurrently
    results = await asyncio.gather(*tasks)

    return results


async def run_batch_with_graphprocessor(
    test_cases: List[TestCaseInput],
    model: str = "llama3.1",
    base_url: str = "http://localhost:11434",
    temperature: float = 0.0
) -> List[ReviewResult]:
    """
    Run batch review using GraphProcessor pattern (single port).

    This is an alternative to run_batch_ollama_test_case_review that uses
    a single reviewer instance with GraphProcessor-style execution.

    Args:
        test_cases: List of TestCaseInput objects
        model: Ollama model name
        base_url: Ollama server URL
        temperature: LLM temperature

    Returns:
        List of ReviewResult objects
    """
    # Create single reviewer instance
    reviewer = get_ollama_reviewer_runnable(
        model=model,
        base_url=base_url,
        temperature=temperature
    )

    results = []

    for test in test_cases:
        logging.info(f"Reviewing {test.test_id}...")

        # Build initial state
        initial_state = {
            "messages": [],
            "test_case_id": test.test_id,
            "test_case_text": test.test_case_text,
            "structure": "",
            "objective": "",
            "review_summary": "",
            "final_result": None
        }

        try:
            # Run async review
            final_state = await reviewer.ainvoke(initial_state)
            result = final_state.get("final_result")

            if result:
                results.append(result)
                logging.info(f"Completed {test.test_id}: score={result.overall_score:.2f}")
            else:
                logging.warning(f"No final result for {test.test_id}")

        except Exception as e:
            logging.error(f"Failed to review {test.test_id}: {e}")

    return results
