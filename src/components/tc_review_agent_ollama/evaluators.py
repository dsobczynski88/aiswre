"""
Evaluator factory functions for Ollama-based test case review.

Creates node functions configured with specific prompts and response models.
"""

from typing import Callable
try:
    from langchain_ollama import ChatOllama
except ImportError:
    from langchain_community.chat_models import ChatOllama

from .core import StructureEvaluation, ObjectiveEvaluation, ReviewSummary


def make_structure_evaluator(
    model: str = "llama3.1",
    base_url: str = "http://localhost:11434",
    temperature: float = 0.0
) -> Callable:
    """
    Create structure evaluation node.

    Evaluates test case structure against acceptance criteria:
    - Logical, sequential steps
    - Clear documentation of data collection
    - Steps are concise and free of jargon
    - Steps are numbered/ordered
    - Outcome aligns with expected results

    Args:
        model: Ollama model name (e.g., "llama3.1", "mistral")
        base_url: Ollama server URL (for multi-port support)
        temperature: LLM temperature

    Returns:
        Node function for structure evaluation
    """

    llm = ChatOllama(
        model=model,
        base_url=base_url,
        temperature=temperature
    )

    # Create structured output LLM
    structured_llm = llm.with_structured_output(StructureEvaluation)

    SYSTEM_PROMPT = """You are a Senior Software QA Architect. Your task is to evaluate the structure of a test case against the provided Acceptance Criteria. You will determine whether the acceptance criteria has been appropriately satisfied:

Acceptance Criteria:
- Test case steps follow a logical, sequential path that ensures reproducibility.
- Actions are documented where data collection is required, and data collection is accurate and complete.
- Steps are clear, concise, and free of jargon; absolutes are avoided.
- Steps are numbered or ordered for easy execution.
- Outcome aligns with the expected results.

You will provide: assessment_verdict, assessment_rationale, identified_gaps, recommendations, and test_case_improvements. Propose improvements detecting misalignment between test case objective and actual steps.

Response Format (produce exactly this JSON structure):
{
    "assessment_verdict": "complete|partial|inadequate",
    "assessment_rationale": "<description of how and why the value for assessment_verdict was chosen>",
    "identified_gaps": ["<gap 1>", "<gap 2>", ...],
    "recommendations": ["<recommendation 1>", "<recommendation 2>", ...],
    "test_case_improvements": ["<improvement 1>", "<improvement 2>", ...]
}"""

    def evaluator_node(state: dict) -> dict:
        """Structure evaluation node function."""
        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

        test_case = state.get("test_case_text", "")
        test_id = state.get("test_case_id", "unknown")

        user_message = f"""Task: Evaluate the completeness of a test case against the acceptance criteria for Test Case Structure.

Input Variables:
- Test Case:
{test_case}

Produce output strictly in the Response Format JSON. Do not use Markdown.

Now perform the review on the provided Input Variables and return only the Response Format JSON."""

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_message)
        ]

        # Use structured output
        result = structured_llm.invoke(messages)

        # Create AI message for chat history
        ai_msg = AIMessage(content=result.model_dump_json(), name="eval_structure")

        return {
            "messages": [ai_msg],
            "structure": result.model_dump_json()
        }

    return evaluator_node


def make_objective_evaluator(
    model: str = "llama3.1",
    base_url: str = "http://localhost:11434",
    temperature: float = 0.0
) -> Callable:
    """
    Create objective/completeness evaluation node.

    Evaluates test case completeness and objective alignment:
    - Includes all required components (ID, title, preconditions, etc.)
    - Clear objective aligned with requirement
    - Meets intended objective
    - Positive and negative scenarios included
    - Expected results provide verification evidence

    Args:
        model: Ollama model name
        base_url: Ollama server URL (for multi-port support)
        temperature: LLM temperature

    Returns:
        Node function for objective evaluation
    """

    llm = ChatOllama(
        model=model,
        base_url=base_url,
        temperature=temperature
    )

    # Create structured output LLM
    structured_llm = llm.with_structured_output(ObjectiveEvaluation)

    SYSTEM_PROMPT = """You are a Senior Software QA Architect. Your task is to evaluate the completeness of a test case against the provided Acceptance Criteria. You will determine whether the acceptance criteria has been appropriately satisfied:

Acceptance Criteria:
- Test case includes all required components: unique ID, descriptive title, preconditions, input data, expected results, and detailed steps.
- Test case objective is clear, specific, and aligned with the requirement.
- Test case meets its intended objective (e.g., verifies actual outcome, not just UI interaction).
- Both positive and negative scenarios are included.
- Expected results align with the requirement and provide sufficient evidence for verification.
- Completeness score should meet or exceed 80%.

You will provide: assessment_verdict, assessment_rationale, identified_gaps, recommendations, and test_case_improvements. Propose improvements detecting misalignment between test case objective and actual steps.

Response Format (produce exactly this JSON structure):
{
    "assessment_verdict": "complete|partial|inadequate",
    "assessment_rationale": "<description of how and why the value for assessment_verdict was chosen>",
    "identified_gaps": ["<gap 1>", "<gap 2>", ...],
    "recommendations": ["<recommendation 1>", "<recommendation 2>", ...],
    "test_case_improvements": ["<improvement 1>", "<improvement 2>", ...]
}"""

    def evaluator_node(state: dict) -> dict:
        """Objective evaluation node function."""
        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

        test_case = state.get("test_case_text", "")
        test_id = state.get("test_case_id", "unknown")

        user_message = f"""Task: Evaluate the completeness of a test case against the acceptance criteria for Test Case Completeness.

Input Variables:
- Test Case:
{test_case}

Produce output strictly in the Response Format JSON. Do not use Markdown.

Now perform the review on the provided Input Variables and return only the Response Format JSON."""

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_message)
        ]

        # Use structured output
        result = structured_llm.invoke(messages)

        # Create AI message for chat history
        ai_msg = AIMessage(content=result.model_dump_json(), name="eval_objective")

        return {
            "messages": [ai_msg],
            "objective": result.model_dump_json()
        }

    return evaluator_node


def make_summary_generator(
    model: str = "llama3.1",
    base_url: str = "http://localhost:11434",
    temperature: float = 0.0
) -> Callable:
    """
    Create review summary aggregation node.

    Combines structure and objective evaluations into a comprehensive summary.

    Args:
        model: Ollama model name
        base_url: Ollama server URL
        temperature: LLM temperature

    Returns:
        Node function for summary generation
    """

    llm = ChatOllama(
        model=model,
        base_url=base_url,
        temperature=temperature
    )

    # Create structured output LLM
    structured_llm = llm.with_structured_output(ReviewSummary)

    SYSTEM_PROMPT = """You are a Senior Test Verification Traceability Analyst with expertise in software quality assurance.
You specialize in summarizing test case reviews, ensuring that critical improvements and recommendations are captured to provide a robust summary rationale.

Response Format (produce exactly this JSON structure):
{
    "testcase_review_summary": "<summary of the outputs from test case review nodes>"
}"""

    def summary_node(state: dict) -> dict:
        """Summary generation node function."""
        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
        import json

        # Gather all messages from parallel evaluators
        all_messages_text = []
        for m in state.get("messages", []):
            content = getattr(m, "content", str(m))
            all_messages_text.append(content)

        user_message = f"""Summarize the below test case review:

## Inputs
Test Case Review:
{json.dumps(all_messages_text, ensure_ascii=False, indent=2)}

## Notes
Produce output strictly in the described Response Format"""

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_message)
        ]

        # Use structured output
        result = structured_llm.invoke(messages)

        # Create AI message for chat history
        ai_msg = AIMessage(content=result.model_dump_json(), name="get_review_summary")

        return {
            "messages": [ai_msg],
            "review_summary": result.model_dump_json()
        }

    return summary_node
