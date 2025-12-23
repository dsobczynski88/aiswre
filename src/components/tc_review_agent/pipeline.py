from langgraph.graph import StateGraph
from langchain_core.runnables.graph import RunnableGraph
from src.components.clients import RateLimitOpenAIClient
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