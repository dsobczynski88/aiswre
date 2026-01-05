"""
Example script demonstrating medtech test case review agent.

This script shows how to use the MedTech Test Case Reviewer to evaluate
test cases against FDA/IEC 62304 best practices.

Usage:
    python scripts/readme_medtech_reviewer_example.py
"""

import asyncio
import pandas as pd
from dotenv import dotenv_values
from src.components.tc_review_agent_medtech.pipeline import (
    get_medtech_reviewer_runnable,
    run_batch_medtech_test_case_review,
    dataframe_to_medtech_inputs
)
from src.components.tc_review_agent_medtech.core import TestCase, Requirement

# ============================================================================
# Configuration
# ============================================================================

DOT_ENV = dotenv_values(".env")
OPENAI_API_KEY = DOT_ENV["OPENAI_API_KEY"]
MODEL = "gpt-4o-mini"

# Optional: Custom weights for score aggregation
# Default weights emphasize traceability and compliance
CUSTOM_WEIGHTS = {
    # General Integrity & Structure (25% total)
    "unambiguity_score": 0.10,
    "independence_score": 0.05,
    "preconditions_score": 0.05,
    "postconditions_score": 0.05,

    # Coverage & Technique (40% total)
    "technique_application_score": 0.10,
    "negative_testing_score": 0.10,
    "boundary_checks_score": 0.10,
    "risk_verification_score": 0.10,

    # Traceability & Compliance (35% total)
    "traceability_score": 0.15,
    "safety_class_rigor_score": 0.10,
    "objective_evidence_score": 0.10,
}


# ============================================================================
# Sample Data (Replace with your actual data)
# ============================================================================

def create_sample_data():
    """Create sample test cases and requirements for demonstration."""

    test_cases = [
        TestCase(
            test_id="TC-001",
            description="Verify temperature alarm triggers when reading exceeds 38.5°C",
            preconditions="System powered on, sensor connected, baseline temp 37.0°C",
            steps="1. Set temperature to 38.6°C\n2. Wait 2 seconds\n3. Observe alarm",
            expected_result="Audible alarm sounds within 2 seconds, display shows 'TEMP HIGH'",
            postconditions="Reset alarm, return temperature to 37.0°C",
            test_type="System",
            technique="BVA"
        ),
        TestCase(
            test_id="TC-002",
            description="Test the system works correctly",
            preconditions=None,
            steps="Turn it on and check",
            expected_result="System is correct",
            postconditions=None,
            test_type="Integration",
            technique=None
        ),
        TestCase(
            test_id="TC-003",
            description="Verify minimum heart rate boundary at 30 BPM",
            preconditions="Device initialized, ECG sensor connected, test signal generator ready",
            steps="1. Generate 29 BPM signal\n2. Generate 30 BPM signal\n3. Generate 31 BPM signal",
            expected_result="29 BPM: Error 'OUT_OF_RANGE'\n30 BPM: Display '30 BPM'\n31 BPM: Display '31 BPM'",
            postconditions="Disconnect test generator, clear error state",
            test_type="Unit",
            technique="BVA"
        )
    ]

    requirements = [
        Requirement(
            req_id="REQ-101",
            text="The system shall trigger an audible alarm when temperature exceeds 38.5°C",
            risk_id="RISK-015",
            safety_class="B"
        ),
        Requirement(
            req_id="REQ-102",
            text="The system shall process patient data accurately",
            risk_id=None,
            safety_class="A"
        ),
        Requirement(
            req_id="REQ-103",
            text="The system shall accept heart rate values between 30 and 300 BPM",
            risk_id="RISK-008",
            safety_class="C"
        )
    ]

    return test_cases, requirements


# ============================================================================
# Main Execution
# ============================================================================

async def main():
    """Run medtech test case review."""

    print("=" * 70)
    print("MedTech Test Case Reviewer - FDA/IEC 62304 Compliance Check")
    print("=" * 70)
    print()

    # Create sample data
    test_cases, requirements = create_sample_data()

    print(f"Reviewing {len(test_cases)} test cases...")
    print()

    # Initialize reviewer
    reviewer = get_medtech_reviewer_runnable(
        api_key=OPENAI_API_KEY,
        model=MODEL,
        weights=CUSTOM_WEIGHTS
    )

    # Run batch review
    results = await run_batch_medtech_test_case_review(
        reviewer=reviewer,
        test_cases=test_cases,
        requirements=requirements,
        batch_size=5
    )

    # Display results
    print("=" * 70)
    print("REVIEW RESULTS")
    print("=" * 70)
    print()

    for result in results:
        print(f"Test: {result.test_id} -> Requirement: {result.req_id}")
        print(f"Link Type: {result.link_type}")
        print()

        print("SCORES (0-1 scale):")
        print(f"  General Integrity & Structure:")
        print(f"    - Unambiguity:        {result.unambiguity_score:.2f}")
        print(f"    - Independence:       {result.independence_score:.2f}")
        print(f"    - Preconditions:      {result.preconditions_score:.2f}")
        print(f"    - Postconditions:     {result.postconditions_score:.2f}")
        print()
        print(f"  Coverage & Technique:")
        print(f"    - Technique App:      {result.technique_application_score:.2f}")
        print(f"    - Negative Testing:   {result.negative_testing_score:.2f}")
        print(f"    - Boundary Checks:    {result.boundary_checks_score:.2f}")
        print(f"    - Risk Verification:  {result.risk_verification_score:.2f}")
        print()
        print(f"  Traceability & Compliance:")
        print(f"    - Traceability:       {result.traceability_score:.2f}")
        print(f"    - Safety Class Rigor: {result.safety_class_rigor_score:.2f}")
        print(f"    - Objective Evidence: {result.objective_evidence_score:.2f}")
        print()

        if result.issues:
            print(f"ISSUES FOUND ({len(result.issues)}):")
            for issue in result.issues[:5]:  # Show first 5 issues
                print(f"  - {issue}")
            if len(result.issues) > 5:
                print(f"  ... and {len(result.issues) - 5} more")
            print()

        print("REVIEW SUMMARY:")
        # Print first 300 chars of summary
        summary_preview = result.review_summary[:300] + "..." if len(result.review_summary) > 300 else result.review_summary
        print(f"  {summary_preview}")
        print()

        print("RECOMMENDED IMPROVEMENTS:")
        # Print first 300 chars of improvements
        improvements_preview = result.test_case_improvements[:300] + "..." if len(result.test_case_improvements) > 300 else result.test_case_improvements
        print(f"  {improvements_preview}")
        print()

        print("DETAILED RATIONALE:")
        # Print first 200 chars of rationale
        rationale_preview = result.rationale[:200] + "..." if len(result.rationale) > 200 else result.rationale
        print(f"  {rationale_preview}")
        print()
        print("-" * 70)
        print()

    # Convert to DataFrame for export
    results_data = []
    for r in results:
        results_data.append({
            "test_id": r.test_id,
            "req_id": r.req_id,
            "unambiguity_score": r.unambiguity_score,
            "independence_score": r.independence_score,
            "preconditions_score": r.preconditions_score,
            "postconditions_score": r.postconditions_score,
            "technique_application_score": r.technique_application_score,
            "negative_testing_score": r.negative_testing_score,
            "boundary_checks_score": r.boundary_checks_score,
            "risk_verification_score": r.risk_verification_score,
            "traceability_score": r.traceability_score,
            "safety_class_rigor_score": r.safety_class_rigor_score,
            "objective_evidence_score": r.objective_evidence_score,
            "issues_count": len(r.issues),
            "issues": " | ".join(r.issues),
            "review_summary": r.review_summary,
            "test_case_improvements": r.test_case_improvements,
            "rationale": r.rationale,
            "link_type": r.link_type
        })

    df_results = pd.DataFrame(results_data)

    # Save to Excel
    output_path = "output/medtech_test_review_results.xlsx"
    df_results.to_excel(output_path, index=False)
    print(f"Results saved to: {output_path}")
    print()

    # Summary statistics
    print("=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    print()
    print("Average Scores:")
    for col in df_results.columns:
        if col.endswith("_score"):
            avg = df_results[col].mean()
            category = col.replace("_score", "").replace("_", " ").title()
            print(f"  {category:.<30} {avg:.2f}")
    print()
    print(f"Total Issues Found: {df_results['issues_count'].sum()}")
    print(f"Tests with Issues:  {(df_results['issues_count'] > 0).sum()} / {len(df_results)}")
    print()


if __name__ == "__main__":
    asyncio.run(main())
