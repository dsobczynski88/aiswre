from langchain_openai import ChatOpenAI
from .nodes import BaseEvaluatorNode
from .core import (
    TraceabilityResponse, AdequacyResponse, ClarityResponse
)

def make_traceability_evaluator(llm: ChatOpenAI) -> BaseEvaluatorNode:
    prompt = (
        "You are a software traceability expert evaluating how well a test case traces to a requirement. "
        "Analyze the requirement and test case provided. "
        "Return ONLY valid JSON (no markdown, no code blocks) with this exact structure:\n"
        '{"confidence_score": <float 0-1>, "rationale": "<explanation>"}\n'
        "Confidence score should reflect how well the test case addresses the requirement."
    )
    return BaseEvaluatorNode(
        llm=llm,
        response_model=TraceabilityResponse,
        system_prompt=prompt,
        link_type="Traceability"
    )

def make_adequacy_evaluator(llm: ChatOpenAI) -> BaseEvaluatorNode:
    prompt = (
        "You are a verification engineer evaluating test case adequacy. "
        "Analyze if the test case adequately covers the requirement. "
        "Return ONLY valid JSON (no markdown, no code blocks) with this exact structure:\n"
        '{"adequacy_score": <float 0-1>, "missing_conditions": [<list of strings>], "rationale": "<explanation>"}\n'
        "Adequacy score should reflect completeness of coverage."
    )
    return BaseEvaluatorNode(
        llm=llm,
        response_model=AdequacyResponse,
        system_prompt=prompt,
        link_type="Adequacy"
    )

def make_clarity_evaluator(llm: ChatOpenAI) -> BaseEvaluatorNode:
    prompt = (
        "You are a QA documentation specialist evaluating test case clarity. "
        "Analyze how clear and well-written the test case is. "
        "Return ONLY valid JSON (no markdown, no code blocks) with this exact structure:\n"
        '{"clarity_score": <float 0-1>, "rewrite_suggestions": "<suggestions or empty string>", "rationale": "<explanation>"}\n'
        "Clarity score should reflect how understandable and well-structured the test case is."
    )
    return BaseEvaluatorNode(
        llm=llm,
        response_model=ClarityResponse,
        system_prompt=prompt,
        link_type="Clarity"
    )