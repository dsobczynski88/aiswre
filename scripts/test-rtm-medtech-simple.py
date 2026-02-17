"""
Test script for the simple RTM MedTech graph (decomposer + summarizer only).

Usage:
    python scripts/test-rtm-medtech-simple.py
"""

import asyncio
import pandas as pd
from dotenv import dotenv_values
from langchain_openai import ChatOpenAI
from aiswre.components.rtm_review_agent_medtech.pipeline import RTMReviewerRunnable
from aiswre.components.rtm_review_agent_medtech.core import Requirement, TestCase
from aiswre.components.processors import process_json_responses

# ============================================================================
# Configuration
# ============================================================================

DOT_ENV = dotenv_values(".env")
OPENAI_API_KEY = DOT_ENV["OPENAI_API_KEY"]
MODEL = "gpt-4o-mini"


# ============================================================================
# Sample Data
# ============================================================================

def create_sample_data():
    """Create a sample requirement and test cases for demonstration."""

    requirement = Requirement(
        req_id="REQ-001",
        text="The system shall measure temperature in the range 35.0-42.0\u00b0C with accuracy \u00b10.1\u00b0C",
    )

    test_cases = [
        TestCase(
            test_id="TC-001",
            description="Verify temperature measurement at lower boundary",
            setup="System powered on, calibrated sensor connected",
            steps="1. Set reference temp to 35.0\u00b0C\n2. Initiate measurement\n3. Record result",
            expectedResults="Display reads 35.0\u00b0C \u00b10.1\u00b0C",
        ),
        TestCase(
            test_id="TC-002",
            description="Verify temperature measurement at upper boundary",
            setup="System powered on, calibrated sensor connected",
            steps="1. Set reference temp to 42.0\u00b0C\n2. Initiate measurement\n3. Record result",
            expectedResults="Display reads 42.0\u00b0C \u00b10.1\u00b0C",
        ),
        TestCase(
            test_id="TC-003",
            description="Verify temperature measurement at nominal value",
            setup="System powered on, calibrated sensor connected",
            steps="1. Set reference temp to 37.0\u00b0C\n2. Initiate measurement\n3. Record result",
            expectedResults="Display reads 37.0\u00b0C \u00b10.1\u00b0C",
        ),
    ]

    return requirement, test_cases


# ============================================================================
# Main Execution
# ============================================================================

async def main():
    """Run the simple decomposer + summarizer graph."""

    print("=" * 70)
    print("RTM MedTech Simple Graph Test (Decomposer + Summarizer)")
    print("=" * 70)
    print()

    requirement, test_cases = create_sample_data()

    # Build the simple graph
    client = ChatOpenAI(model=MODEL, api_key=OPENAI_API_KEY, temperature=0.0)
    simple_graph = RTMReviewerRunnable.build_simple_graph(client)

    # Prepare input state
    input_state = {
        "requirement": requirement,
        "test_cases": test_cases,
    }

    print(f"Requirement: {requirement.req_id} - {requirement.text}")
    print(f"Test Cases: {len(test_cases)}")
    print()
    print("Running simple graph (decomposer + summarizer)...")
    print()

    # Invoke the graph
    result = await simple_graph.ainvoke(input_state)

    # ============================================================================
    # Convert Pydantic outputs to JSON strings for process_json_responses
    # ============================================================================
    decomposed = result.get("decomposed_requirement")
    test_suite = result.get("test_suite")

    responses = []
    ids = []

    if decomposed:
        responses.append(decomposed.model_dump_json())
        ids.append(f"{requirement.req_id}_decomposed")

    if test_suite:
        responses.append(test_suite.model_dump_json())
        ids.append(f"{requirement.req_id}_summarized")

    processed = process_json_responses(
        responses=responses,
        ids=ids,
        prompt_type="simple_graph",
    )

    # ============================================================================
    # Convert to DataFrame and save
    # ============================================================================
    df = pd.DataFrame(processed)
    output_path = "output/test-simple-graph.xlsx"
    df.to_excel(output_path, index=False)
    print(f"Results saved to: {output_path}")
    print(f"DataFrame shape: {df.shape}")
    print()
    print(df.to_string())
    print()

    print("=" * 70)
    print("Simple graph test complete.")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
