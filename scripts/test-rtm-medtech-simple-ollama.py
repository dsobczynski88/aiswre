"""
Test script for the simple RTM MedTech graph using local Ollama.

Runs the same decomposer → summarizer (parallel) → generator → coverage_evaluator
pipeline as test-rtm-medtech-simple.py, but uses ChatOllama instead of ChatOpenAI.

Prerequisites:
    1. Install Ollama: https://ollama.ai
    2. Pull a model that supports structured output: ollama pull llama3.1
    3. Start Ollama: ollama serve

Usage:
    python scripts/test-rtm-medtech-simple-ollama.py
"""

import asyncio
import pandas as pd

try:
    from langchain_ollama import ChatOllama
except ImportError:
    from langchain_community.chat_models import ChatOllama

from aiswre.components.rtm_review_agent_medtech.pipeline import RTMReviewerRunnable
from aiswre.components.rtm_review_agent_medtech.core import Requirement, TestCase
from aiswre.utils import save_graph_png

# ============================================================================
# Configuration
# ============================================================================

MODEL = "mistral"           # Change to "llama3.1", "qwen2.5", or another Ollama model
BASE_URL = "http://localhost:11434"


# ============================================================================
# Sample Data
# ============================================================================

def create_sample_data():
    """Create a sample requirement and test cases for demonstration."""

    requirement = Requirement(
        req_id="REQ-001",
        text="If the user logs in successfully, display the application. If a user attempts to log in to the application, and the user is not authenticated diplay a generic login error message which provides no indication of the failure reason. If a user attempts to login and does not have the correct network configuration a network failure message will be displayed",
    )

    test_cases = [
        TestCase(
            test_id="TEST-001",
            description="If the user attempt to log in to the application and the user is not authenticated, the System shall display a generic login error message on top of the login section and provides no indication of the failure reason. The System shall display the application, if the user logs in successfully. The functionality described above shall operate properly on all supported operating systems and browsers.",
            setup="Data sets & Modules :\nAuthentication as 'AUT'\nAuthorization level as 'AutLevel'\nServer information as 'ServerInfo'\nSupported Operating System(s) & Browser(s) AS 'BOS'",
            steps="1. On Login screen enter invalid user name/password\n- Click on Sign in Button\n2. Select Domain as specified in [AutLevel].[Domain]\n3. - Select System Map - Repeat Step 1-2.\n4. Logout and Close the System Map and Logout from System Portal.\n5. Repeat step 1-4 for [AutLevel].[UN & PW].[2].\n6. - Launch the System Map of respective server from [ServerInfo].[System Map] in all [BOS].[Browser].\n7. Repeat for each of the records in [BOS].",
            expectedResults="1.1. User shall not be logged in to application. 2. Generic login error message 'You entered an incorrect username, password or both' is displayed on top of the login section with no indication of the failure reason\n2.User shall be logged in to application successfully.\n3.Respective expected result\n4.N/A\n5.Respective expected result shall be achieved.\n6.Respective expected result shall be achieved.\n7.Respective expected result shall be achieved."
        )
    ]

    return requirement, test_cases


# ============================================================================
# Main Execution
# ============================================================================

async def main():
    """Run the simple decomposer + summarizer + generator + coverage graph."""

    print("=" * 70)
    print("RTM MedTech Simple Graph Test (Ollama) — Decomposer + Summarizer +")
    print("Generator + Coverage Evaluator")
    print("=" * 70)
    print()
    print(f"Model:    {MODEL}")
    print(f"Base URL: {BASE_URL}")
    print()

    requirement, test_cases = create_sample_data()

    # Build the simple graph
    client = ChatOllama(model=MODEL, base_url=BASE_URL, temperature=0.0)
    simple_graph = RTMReviewerRunnable.build_simple_graph(client)
    save_graph_png(simple_graph, "output/simple_graph_ollama.png")

    # Prepare input state
    input_state = {
        "requirement": requirement,
        "test_cases": test_cases,
    }

    print(f"Requirement: {requirement.req_id} - {requirement.text}")
    print(f"Test Cases: {len(test_cases)}")
    print()
    print("Running simple graph...")
    print()

    # Invoke the graph
    result = await simple_graph.ainvoke(input_state)

    # ============================================================================
    # Build 7-tab Excel workbook
    # ============================================================================
    decomposed = result.get("decomposed_requirement")
    test_suite = result.get("test_suite")
    ai_test_suite = result.get("ai_test_suite")
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
            decomposer_rows.append(
                {
                    "spec_id": spec.spec_id,
                    "type": spec.type,
                    "description": spec.description,
                    "acceptance_criteria": spec.acceptance_criteria,
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

    # Tab 4: covered (EvaluatedSpec where covered_exists=True)
    covered_rows = []
    for coverage in coverage_responses:
        for spec in coverage.evaluations:
            if spec.covered_exists:
                covered_rows.append(
                    {
                        "spec_id": spec.spec_id,
                        "covered_by_test_cases": "; ".join(spec.covered_by_test_cases),
                        "escaped_defect_risk": spec.escaped_defect_risk,
                        "escaped_defect_risk_rationale": spec.escaped_defect_risk_rationale,
                        "coverage_rationale": spec.coverage_rationale,
                    }
                )
    df_covered = pd.DataFrame(covered_rows)

    # Tab 5: missing (EvaluatedSpec where covered_exists=False)
    missing_rows = []
    for coverage in coverage_responses:
        for spec in coverage.evaluations:
            if not spec.covered_exists:
                missing_rows.append(
                    {
                        "spec_id": spec.spec_id,
                        "escaped_defect_risk": spec.escaped_defect_risk,
                        "escaped_defect_risk_rationale": spec.escaped_defect_risk_rationale,
                        "coverage_rationale": spec.coverage_rationale,
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

    # Tab 7: aitestsuite (full merged SummarizedTestCase list from AITestSuite)
    aitestsuite_rows = []
    if ai_test_suite:
        for s in ai_test_suite.ai_test_suite:
            aitestsuite_rows.append(
                {
                    "test_case_id": s.test_case_id,
                    "objective": s.objective,
                    "verifies": s.verifies,
                    "protocol": "; ".join(s.protocol),
                    "acceptance_criteria": "; ".join(s.acceptance_criteria),
                }
            )
    df_aitestsuite = pd.DataFrame(aitestsuite_rows)

    # Write all tabs
    output_path = "output/test-simple-graph-ollama.xlsx"
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df_inputs.to_excel(writer, sheet_name="inputs", index=False)
        df_decomposer.to_excel(writer, sheet_name="decomposer", index=False)
        df_summarizer.to_excel(writer, sheet_name="summarizer", index=False)
        df_covered.to_excel(writer, sheet_name="covered", index=False)
        df_missing.to_excel(writer, sheet_name="missing", index=False)
        df_rationale.to_excel(writer, sheet_name="rationale", index=False)
        df_aitestsuite.to_excel(writer, sheet_name="aitestsuite", index=False)

    print(f"Results saved to: {output_path}")
    print(
        f"  inputs: {len(df_inputs)} rows | decomposer: {len(df_decomposer)} rows | "
        f"summarizer: {len(df_summarizer)} rows"
    )
    print(
        f"  covered: {len(df_covered)} rows | missing: {len(df_missing)} rows | "
        f"rationale: {len(df_rationale)} rows | aitestsuite: {len(df_aitestsuite)} rows"
    )
    print()

    # Quick console summary
    if coverage_responses:
        print("-" * 70)
        print("Coverage Summary")
        print("-" * 70)
        for coverage in coverage_responses:
            n_covered = sum(1 for s in coverage.evaluations if s.covered_exists)
            n_missing = sum(1 for s in coverage.evaluations if not s.covered_exists)
            print(f"  Covered: {n_covered}  |  Missing: {n_missing}")
        print()

    print("=" * 70)
    print("Simple graph test complete.")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
