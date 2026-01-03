import json
import logging
from typing import Any, Dict, Type, TypeVar, Optional, List
from langchain_openai import ChatOpenAI
from pydantic import ValidationError
from .core import (
    Requirement, TestCase, TraceLink,
    TraceabilityResponse, AdequacyResponse, ClarityResponse
)

R = TypeVar("R", bound=Any)

class BaseEvaluatorNode:
    """Abstract evaluator. Subclasses define prompt templates & response model."""

    def __init__(
        self,
        llm: ChatOpenAI,
        response_model: Type[R],
        system_prompt: str,
        *,
        link_type: str
        ):
        self.llm = llm
        self.response_model = response_model
        self.system_prompt = system_prompt
        self.link_type = link_type

        # Create a structured output LLM that returns the response model directly
        self.structured_llm = llm.with_structured_output(response_model)

    async def __call__(self, state: dict) -> dict:
        """Process state and return updated state with new trace link."""
        requirement = state.get('requirement')
        test = state.get('test')

        if not test:
            raise ValueError("TestCase is required in state")

        # Build payload for LLM
        payload = {
            "requirement": requirement.text if requirement else "No requirement provided",
            "test_case": test.description
        }

        # Call LLM with structured output
        try:
            from langchain_core.messages import SystemMessage, HumanMessage
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=json.dumps(payload))
            ]
            # Use structured output to get Pydantic model directly
            parsed = await self.structured_llm.ainvoke(messages)
            trace_link = self._to_tracelink(parsed, requirement, test)
        except Exception as e:
            logging.error(f"LLM invocation or parsing failed in {self.link_type}: {e}")
            trace_link = self._fallback_link(requirement, test)

        # Return only the new trace link (not accumulated list)
        # LangGraph will use operator.add to merge with other parallel node outputs
        return {"trace_links": [trace_link]}

    def _fallback_link(self, requirement: Optional[Requirement], test: TestCase) -> TraceLink:
        return TraceLink(
            req_id=requirement.req_id if requirement else "unknown",
            test_id=test.test_id,
            rationale="LLM error or invalid response",
            link_type=self.link_type
        )

    def _to_tracelink(self, resp: R, requirement: Optional[Requirement], test: TestCase) -> TraceLink:
        data = resp.dict()
        tl = TraceLink(
            req_id=requirement.req_id if requirement else "unknown",
            test_id=test.test_id,
            link_type=self.link_type,
            rationale=data.get("rationale", "")
        )
        # Map known fields
        if isinstance(resp, TraceabilityResponse):
            tl.confidence_score = resp.confidence_score
        elif isinstance(resp, AdequacyResponse):
            tl.adequacy_score = resp.adequacy_score
        elif isinstance(resp, ClarityResponse):
            tl.clarity_score = resp.clarity_score
        return tl

class AggregatorNode:
    def __init__(self, weights: Optional[dict] = None):
        # default equal weights
        self.weights = weights or {
            "confidence_score": 1.0,
            "adequacy_score": 1.0,
            "clarity_score": 1.0
        }

    def __call__(self, state: dict) -> dict:
        """Aggregate trace links from state and return final result."""
        links = state.get('trace_links', [])

        if not links:
            raise ValueError("No trace links to aggregate")

        # Weighted sum / normalization
        total_w = sum(self.weights.values())
        def weighted_avg(field):
            return round(
                sum(getattr(t, field, 0.0) * self.weights.get(field, 0.0) for t in links) / total_w if total_w > 0 else 0.0,
                3
            )

        combined = TraceLink(
            req_id=links[0].req_id,
            test_id=links[0].test_id,
            confidence_score=weighted_avg("confidence_score"),
            adequacy_score=weighted_avg("adequacy_score"),
            clarity_score=weighted_avg("clarity_score"),
            rationale=" | ".join(t.rationale for t in links if t.rationale),
            link_type="Aggregated"
        )
        logging.debug(f"Aggregated link: {combined}")

        return {"final_result": combined}