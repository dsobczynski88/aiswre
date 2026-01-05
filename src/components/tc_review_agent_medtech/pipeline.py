"""
MedTech Test Case Review Pipeline

LangGraph-based workflow for evaluating medical device software test cases
against FDA/IEC 62304 best practices.
"""

import asyncio
import logging
import pandas as pd
from typing import List, Optional, Dict, Any
from langgraph.graph import StateGraph
from langchain_openai import ChatOpenAI
from src.components.processors import df_to_prompt_items
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
from .nodes import MedtechAggregatorNode


class MedtechTestCaseReviewerRunnable:
    """
    LangGraph-based medtech test case reviewer pipeline.

    Evaluates test cases against 11 criteria organized into 3 categories:
    1. General Integrity & Structure (4 evaluators)
    2. Coverage & Technique (4 evaluators)
    3. Traceability & Compliance (3 evaluators)

    All evaluators run in parallel, then results are aggregated.
    """

    def __init__(self, client: ChatOpenAI, weights: Optional[dict] = None):
        """
        Initialize the medtech reviewer pipeline.

        Args:
            client: ChatOpenAI client for LLM calls
            weights: Optional dict of score weights for aggregation
        """
        self.client = client
        self.weights = weights
        self.graph = MedtechTestCaseReviewerRunnable.build_graph(self.client, self.weights)

    def ainvoke(self, *args, **kwargs):
        """Delegate to the compiled graph's ainvoke method."""
        return self.graph.ainvoke(*args, **kwargs)

    def invoke(self, *args, **kwargs):
        """Delegate to the compiled graph's invoke method."""
        return self.graph.invoke(*args, **kwargs)

    @staticmethod
    def build_graph(client: ChatOpenAI, weights: Optional[dict]) -> StateGraph:
        """
        Build the LangGraph state machine with 11 parallel evaluators.

        Graph structure:
            START
              ↓
            ┌─────────────────────────────────────────────┐
            │  General Integrity & Structure (parallel)   │
            │  - Unambiguity                              │
            │  - Independence                             │
            │  - Preconditions                            │
            │  - Postconditions                           │
            ├─────────────────────────────────────────────┤
            │  Coverage & Technique (parallel)            │
            │  - Technique Application                    │
            │  - Negative Testing                         │
            │  - Boundary Checks                          │
            │  - Risk Verification                        │
            ├─────────────────────────────────────────────┤
            │  Traceability & Compliance (parallel)       │
            │  - Traceability                             │
            │  - Safety Class Rigor                       │
            │  - Objective Evidence                       │
            └─────────────────────────────────────────────┘
              ↓
            AGGREGATOR
              ↓
            END
        """
        sg = StateGraph(TestCaseState)

        # ====================================================================
        # Create all 11 evaluator nodes
        # ====================================================================

        # General Integrity & Structure
        unambiguity = make_unambiguity_evaluator(client)
        independence = make_independence_evaluator(client)
        preconditions = make_preconditions_evaluator(client)
        postconditions = make_postconditions_evaluator(client)

        # Coverage & Technique
        technique_app = make_technique_application_evaluator(client)
        negative_test = make_negative_testing_evaluator(client)
        boundary_checks = make_boundary_checks_evaluator(client)
        risk_verify = make_risk_verification_evaluator(client)

        # Traceability & Compliance
        traceability = make_traceability_evaluator(client)
        safety_rigor = make_safety_class_rigor_evaluator(client)
        objective_evidence = make_objective_evidence_evaluator(client)

        # Aggregator (now requires LLM client for intelligent aggregation)
        aggregator = MedtechAggregatorNode(llm=client, weights=weights)

        # ====================================================================
        # Register nodes
        # ====================================================================

        # General Integrity & Structure
        sg.add_node("unambiguity", unambiguity)
        sg.add_node("independence", independence)
        sg.add_node("preconditions", preconditions)
        sg.add_node("postconditions", postconditions)

        # Coverage & Technique
        sg.add_node("technique_application", technique_app)
        sg.add_node("negative_testing", negative_test)
        sg.add_node("boundary_checks", boundary_checks)
        sg.add_node("risk_verification", risk_verify)

        # Traceability & Compliance
        sg.add_node("traceability", traceability)
        sg.add_node("safety_class_rigor", safety_rigor)
        sg.add_node("objective_evidence", objective_evidence)

        # Aggregator
        sg.add_node("aggregate", aggregator)

        # ====================================================================
        # Set up parallel execution: all 11 evaluators as entry points
        # ====================================================================

        # First evaluator as primary entry point
        sg.set_entry_point("unambiguity")

        # All other evaluators start from __start__ (run in parallel)
        sg.add_edge("__start__", "independence")
        sg.add_edge("__start__", "preconditions")
        sg.add_edge("__start__", "postconditions")
        sg.add_edge("__start__", "technique_application")
        sg.add_edge("__start__", "negative_testing")
        sg.add_edge("__start__", "boundary_checks")
        sg.add_edge("__start__", "risk_verification")
        sg.add_edge("__start__", "traceability")
        sg.add_edge("__start__", "safety_class_rigor")
        sg.add_edge("__start__", "objective_evidence")

        # ====================================================================
        # All evaluators feed into aggregator
        # ====================================================================

        sg.add_edge("unambiguity", "aggregate")
        sg.add_edge("independence", "aggregate")
        sg.add_edge("preconditions", "aggregate")
        sg.add_edge("postconditions", "aggregate")
        sg.add_edge("technique_application", "aggregate")
        sg.add_edge("negative_testing", "aggregate")
        sg.add_edge("boundary_checks", "aggregate")
        sg.add_edge("risk_verification", "aggregate")
        sg.add_edge("traceability", "aggregate")
        sg.add_edge("safety_class_rigor", "aggregate")
        sg.add_edge("objective_evidence", "aggregate")

        # Aggregator is the finish point
        sg.set_finish_point("aggregate")

        return sg.compile()


def get_medtech_reviewer_runnable(
    api_key: str,
    model: str = "gpt-4o-mini",
    weights: Optional[dict] = None
) -> MedtechTestCaseReviewerRunnable:
    """
    Factory function to create a medtech test case reviewer.

    Args:
        api_key: OpenAI API key
        model: Model name (default: gpt-4o-mini)
        weights: Optional score weights for aggregation

    Returns:
        MedtechTestCaseReviewerRunnable instance
    """
    client = ChatOpenAI(
        api_key=api_key,
        model=model,
        temperature=0.0
    )
    return MedtechTestCaseReviewerRunnable(client=client, weights=weights)


async def run_batch_medtech_test_case_review(
    reviewer: MedtechTestCaseReviewerRunnable,
    test_cases: List[TestCase],
    requirements: Optional[List[Requirement]] = None,
    batch_size: int = 10
) -> List[MedtechTraceLink]:
    """
    Run batch review of test cases against medtech checklist.

    Args:
        reviewer: MedtechTestCaseReviewerRunnable instance
        test_cases: List of TestCase objects
        requirements: Optional list of Requirement objects (matched by index)
        batch_size: Number of concurrent reviews

    Returns:
        List of MedtechTraceLink results
    """
    results = []

    # Match test cases with requirements
    for i, test in enumerate(test_cases):
        req = requirements[i] if requirements and i < len(requirements) else None

        # Build initial state
        state = {
            "test": test,
            "requirement": req,
            "medtech_links": [],
            "final_result": None
        }

        # Run async review
        try:
            result = await reviewer.ainvoke(state)
            final = result.get("final_result")
            if final:
                results.append(final)
            else:
                logging.warning(f"No final result for test {test.test_id}")
        except Exception as e:
            logging.error(f"Failed to review test {test.test_id}: {e}")
            # Create fallback result
            fallback = MedtechTraceLink(
                req_id=req.req_id if req else "unknown",
                test_id=test.test_id,
                rationale=f"Review failed: {str(e)}",
                link_type="Error",
                issues=[str(e)]
            )
            results.append(fallback)

        # Batch control (process in chunks to avoid overwhelming API)
        if len(results) % batch_size == 0:
            await asyncio.sleep(1)  # Brief pause between batches

    return results


def dataframe_to_medtech_inputs(
    df: pd.DataFrame,
    test_id_col: str = "test_id",
    test_desc_col: str = "test_description",
    req_id_col: Optional[str] = "req_id",
    req_text_col: Optional[str] = "requirement_text",
    **optional_cols
) -> tuple[List[TestCase], List[Requirement]]:
    """
    Convert DataFrame to TestCase and Requirement lists for medtech review.

    Args:
        df: Input DataFrame
        test_id_col: Column name for test ID
        test_desc_col: Column name for test description
        req_id_col: Optional column name for requirement ID
        req_text_col: Optional column name for requirement text
        **optional_cols: Additional column mappings (e.g., preconditions="precond_col")

    Returns:
        Tuple of (test_cases, requirements)
    """
    test_cases = []
    requirements = []

    for _, row in df.iterrows():
        # Build TestCase
        test = TestCase(
            test_id=row[test_id_col],
            description=row[test_desc_col],
            preconditions=row.get(optional_cols.get("preconditions")),
            steps=row.get(optional_cols.get("steps")),
            expected_result=row.get(optional_cols.get("expected_result")),
            postconditions=row.get(optional_cols.get("postconditions")),
            test_type=row.get(optional_cols.get("test_type")),
            technique=row.get(optional_cols.get("technique"))
        )
        test_cases.append(test)

        # Build Requirement (if columns present)
        if req_text_col and req_text_col in df.columns:
            req = Requirement(
                req_id=row.get(req_id_col) if req_id_col else None,
                text=row[req_text_col],
                risk_id=row.get(optional_cols.get("risk_id")),
                safety_class=row.get(optional_cols.get("safety_class"))
            )
            requirements.append(req)

    return test_cases, requirements
