"""
Test script for the simple RTM MedTech graph (decomposer + summarizer + boundary evaluator).

Usage:
    python scripts/test-rtm-medtech-simple.py
"""

import asyncio
import pandas as pd
from dotenv import dotenv_values
from langchain_openai import ChatOpenAI
from aiswre.components.rtm_review_agent_medtech.pipeline import RTMReviewerRunnable
from aiswre.components.rtm_review_agent_medtech.core import Requirement, TestCase

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
    print("RTM MedTech Simple Graph Test (Decomposer + Summarizer + Boundary)")
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
    print("Running simple graph (decomposer + summarizer + boundary)...")
    print()

    # Invoke the graph
    result = await simple_graph.ainvoke(input_state)

    # ============================================================================
    # Build 6-tab Excel workbook
    # ============================================================================
    decomposed = result.get("decomposed_requirement")
    test_suite = result.get("test_suite")
    coverage_responses = result.get("coverage_responses", [])

    # Tab 1: inputs
    inputs_rows = [
        {
            "requirement_id": requirement.req_id,
            "requirement_text": requirement.text,
            "test_id": tc.test_id,
            "description": tc.description,
            "setup": tc.setup,
            "steps": tc.steps,
            "expectedResults": tc.expectedResults,
        }
        for tc in test_cases
    ]
    df_inputs = pd.DataFrame(inputs_rows)

    # Tab 2: decomposer (specs without rationale)
    decomposer_rows = []
    if decomposed:
        for spec in decomposed.decomposed_specifications:
            eca = spec.edge_case_analysis
            decomposer_rows.append(
                {
                    "spec_id": spec.spec_id,
                    "type": spec.type,
                    "description": spec.description,
                    "verification_method": spec.verification_method,
                    "acceptance_criteria": spec.acceptance_criteria,
                    "potential_edge_cases": "; ".join(eca.potential_edge_cases),
                    "risk_of_escaped_defect": eca.risk_of_escaped_defect,
                    "recommended_mitigation": eca.recommended_mitigation,
                }
            )
    df_decomposer = pd.DataFrame(decomposer_rows)

    # Tab 3: summarizer
    summarizer_rows = []
    if test_suite:
        for s in test_suite.summary:
            summarizer_rows.append(
                {
                    "test_case_id": s.test_case_id,
                    "objective": s.objective,
                    "verifies": s.verifies,
                    "protocol": "; ".join(s.protocol),
                    "acceptance_criteria": "; ".join(s.acceptance_criteria),
                }
            )
    df_summarizer = pd.DataFrame(summarizer_rows)

    # Tab 4: covered
    covered_rows = []
    for coverage in coverage_responses:
        for cb in coverage.covered:
            covered_rows.append(
                {
                    "spec_id": cb.spec_id,
                    "edge_case_summary": cb.edge_case_summary,
                    "mapped_test_case_id": cb.mapped_test_case_id,
                    "coverage_rationale": cb.coverage_rationale,
                }
            )
    df_covered = pd.DataFrame(covered_rows)

    # Tab 5: missing
    missing_rows = []
    for coverage in coverage_responses:
        for mb in coverage.missing:
            stc = mb.summarized_test_case
            missing_rows.append(
                {
                    "spec_id": "",
                    "test_case_id": stc.test_case_id,
                    "objective": stc.objective,
                    "verifies": stc.verifies,
                    "protocol": "; ".join(stc.protocol),
                    "acceptance_criteria": "; ".join(stc.acceptance_criteria),
                    "gap_description": mb.gap_description,
                    "escaped_defect_risk": mb.escaped_defect_risk,
                    "rationale": mb.rationale,
                }
            )
    df_missing = pd.DataFrame(missing_rows)

    # Tab 6: rationale (decomposer rationale per spec)
    rationale_rows = []
    if decomposed:
        for spec in decomposed.decomposed_specifications:
            rationale_rows.append(
                {
                    "spec_id": spec.spec_id,
                    "rationale": spec.rationale,
                }
            )
    df_rationale = pd.DataFrame(rationale_rows)

    # Write all tabs
    output_path = "output/test-simple-graph.xlsx"
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df_inputs.to_excel(writer, sheet_name="inputs", index=False)
        df_decomposer.to_excel(writer, sheet_name="decomposer", index=False)
        df_summarizer.to_excel(writer, sheet_name="summarizer", index=False)
        df_covered.to_excel(writer, sheet_name="covered", index=False)
        df_missing.to_excel(writer, sheet_name="missing", index=False)
        df_rationale.to_excel(writer, sheet_name="rationale", index=False)

    print(f"Results saved to: {output_path}")
    print(
        f"  inputs: {len(df_inputs)} rows | decomposer: {len(df_decomposer)} rows | "
        f"summarizer: {len(df_summarizer)} rows"
    )
    print(
        f"  covered: {len(df_covered)} rows | missing: {len(df_missing)} rows | "
        f"rationale: {len(df_rationale)} rows"
    )
    print()

    # Quick console summary
    if coverage_responses:
        print("-" * 70)
        print("Boundary Coverage Summary")
        print("-" * 70)
        for coverage in coverage_responses:
            print(f"  Covered: {len(coverage.covered)}  |  Missing: {len(coverage.missing)}")
        print()

    print("=" * 70)
    print("Simple graph test complete.")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
