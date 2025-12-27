import asyncio
import time
import pandas as pd
from typing import List, Optional, Union, Dict, Any, Sequence
from langgraph.graph import StateGraph
from langchain_core.runnables.graph import RunnableGraph
from src.components.clients import RateLimitOpenAIClient
from src.components.processors import PromptProcessor, df_to_prompt_items
from src.components.tc_review_agent.agents.evaluators import (
    make_traceability_evaluator,
    make_adequacy_evaluator,
    make_clarity_evaluator
)
from src.components.tc_review_agent.agents.aggregator import AggregatorNode
from src.components.tc_review_agent.models.core import Requirement, TestCase, TraceLink

class TestCaseReviewerRunnable(RunnableGraph):
    """
    LangGraph-based test case reviewer pipeline using RateLimitOpenAIClient.

    Evaluates traceability, adequacy, and clarity of test cases against requirements,
    then aggregates results into a TraceLink.

    Example:
        >>> from dotenv import dotenv_values
        >>> DOT_ENV = dotenv_values(".env")
        >>> reviewer = get_reviewer_runnable(
        ...     api_key=DOT_ENV["OPENAI_API_KEY"],
        ...     model_name="gpt-4o-mini",
        ...     weights={"confidence_score": 1, "adequacy_score": 2, "clarity_score": 1}
        ... )
        >>> req = Requirement(req_id="REQ-045", text="System shall trigger alarm when temp > 80°C")
        >>> test = TestCase(test_id="TC-112", description="Verify alarm triggers at > 80°C")
        >>> result = await reviewer.arun(requirement=req, test=test)
    """
    def __init__(self, client: RateLimitOpenAIClient, model: str = "gpt-4o-mini", weights: dict = None):
        self.client = client
        self.model = model
        graph = self.build_graph(client, model, weights)
        super().__init__(graph)

    @staticmethod
    def build_graph(client: RateLimitOpenAIClient, model: str, weights: dict) -> StateGraph:
        sg = StateGraph()

        # Create nodes
        trace = make_traceability_evaluator(client, model)
        adequacy = make_adequacy_evaluator(client, model)
        clarity = make_clarity_evaluator(client, model)
        aggregator = AggregatorNode(weights)

        # Register
        sg.add_node("trace", trace)
        sg.add_node("adequacy", adequacy)
        sg.add_node("clarity", clarity)
        sg.add_node("aggregate", aggregator)

        # Edges
        sg.add_edge("trace", "aggregate")
        sg.add_edge("adequacy", "aggregate")
        sg.add_edge("clarity", "aggregate")

        sg.set_entry_point("trace")
        sg.set_exit_point("aggregate")
        return sg

    async def arun(
        self,
        requirement: Requirement,
        test: TestCase,
        **kwargs
        ) -> TraceLink:
        result = await super().arun(requirement=requirement, test=test, **kwargs)
        return result

# Helper factory
def get_reviewer_runnable(
    api_key: str,
    model_name: str = "gpt-4o-mini",
    max_requests_per_minute: int = 490,
    max_tokens_per_minute: int = 200000,
    weights: dict = None
    ) -> TestCaseReviewerRunnable:
    """
    Create a TestCaseReviewerRunnable with rate-limited OpenAI client.

    Args:
        api_key: OpenAI API key
        model_name: Model to use (default: gpt-4o-mini)
        max_requests_per_minute: RPM limit (default: 490)
        max_tokens_per_minute: TPM limit (default: 200000)
        weights: Dict of weights for aggregation (e.g., {"confidence_score": 1, "adequacy_score": 2})

    Returns:
        TestCaseReviewerRunnable instance
    """
    client = RateLimitOpenAIClient(
        api_key=api_key,
        max_requests_per_minute=max_requests_per_minute,
        max_tokens_per_minute=max_tokens_per_minute
    )
    return TestCaseReviewerRunnable(client=client, model=model_name, weights=weights)


async def run_batch_test_case_review(
    reviewer: TestCaseReviewerRunnable,
    input_df: pd.DataFrame,
    test_id_col: str = "test_id",
    test_desc_col: str = "test_description",
    req_id_col: Optional[str] = None,
    req_text_col: Optional[str] = None,
    ) -> pd.DataFrame:
    """
    Run batch test case reviews using utility functions and async batch pattern.

    This helper method uses the df_to_prompt_items utility function to prepare data
    from a DataFrame and then runs TestCaseReviewerRunnable graphs asynchronously.
    Each task reviews a single test case, optionally against a corresponding requirement.

    Args:
        reviewer: TestCaseReviewerRunnable instance
        input_df: DataFrame containing test cases and optional requirements
        test_id_col: Column name for test case IDs (default: "test_id")
        test_desc_col: Column name for test descriptions (default: "test_description")
        req_id_col: Optional column name for requirement IDs
        req_text_col: Optional column name for requirement text

    Returns:
        DataFrame with review results

    Example:
        >>> from dotenv import dotenv_values
        >>> DOT_ENV = dotenv_values(".env")
        >>>
        >>> # Setup reviewer
        >>> reviewer = get_reviewer_runnable(api_key=DOT_ENV["OPENAI_API_KEY"])
        >>>
        >>> # DataFrame with requirements
        >>> df = pd.DataFrame({
        ...     "test_id": ["TC-001", "TC-002"],
        ...     "test_description": ["Verify alarm", "Check sound"],
        ...     "req_id": ["REQ-001", "REQ-002"],
        ...     "requirement_text": ["Trigger alarm", "Audible alarm"]
        ... })
        >>>
        >>> results = await run_batch_test_case_review(
        ...     reviewer=reviewer,
        ...     input_df=df,
        ...     req_id_col="req_id",
        ...     req_text_col="requirement_text"
        ... )
    """
    # Determine which columns to extract from DataFrame
    columns = [test_id_col, test_desc_col]
    if req_id_col:
        columns.append(req_id_col)
    if req_text_col:
        columns.append(req_text_col)

    # Use utility function to convert DataFrame to items
    items = df_to_prompt_items(input_df, columns)
    ids = [item[test_id_col] for item in items]

    print(f"🚀 Starting batch review for {len(items)} test cases...")
    start_time = time.time()

    # Build list of tasks comprised of TestCaseReviewerRunnable graphs
    tasks = []
    for item in items:
        # Create TestCase object
        test_case = TestCase(
            test_id=item[test_id_col],
            description=item[test_desc_col]
        )

        # Create Requirement object if requirement columns provided
        requirement = None
        if req_text_col and req_text_col in item:
            req_id = item.get(req_id_col) if req_id_col else None
            requirement = Requirement(
                req_id=req_id,
                text=item[req_text_col]
            )

        # Create task using reviewer's arun method
        task = reviewer.arun(requirement=requirement, test=test_case)
        tasks.append(task)

    # Run graphs asynchronously using asyncio.gather (like PromptProcessor.run_prompt_batch)
    results = await asyncio.gather(*tasks)

    elapsed = time.time() - start_time
    print(f"✅ Completed {len(results)} reviews in {elapsed:.2f} seconds")

    # Convert TraceLink results to DataFrame
    result_rows = []
    for i, result in enumerate(results):
        row_data = {
            "test_id": ids[i],
            "trace_link": result,
        }
        # Extract TraceLink attributes if available
        # TODO: Customize based on actual TraceLink model structure
        if hasattr(result, "__dict__"):
            for key, value in result.__dict__.items():
                row_data[f"trace_link.{key}"] = value

        result_rows.append(row_data)

    results_df = pd.DataFrame(result_rows)

    print(f"📊 Results DataFrame shape: {results_df.shape}")
    return results_df