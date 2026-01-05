"""
Example script for RTM (Requirement Traceability Matrix) review using local Ollama.

This script demonstrates how to use the rtm_review_agent_medtech_local to evaluate
requirement verification coverage in a traceability matrix.
"""

import asyncio
import logging
from src.components.rtm_review_agent_medtech_local import (
    Requirement,
    RTMEntry,
    run_batch_rtm_local_review
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


async def main():
    """Main function to demonstrate RTM review."""

    # ===============================================================
    # Define Requirements
    # ===============================================================
    requirements = [
        Requirement(
            req_id="REQ-001",
            text="The system shall measure temperature in the range 35.0-42.0°C with accuracy ±0.1°C",
            safety_class="C",
            risk_id="RISK-012",
            acceptance_criteria="Temperature readings within ±0.1°C of reference standard"
        ),
        Requirement(
            req_id="REQ-002",
            text="The system shall display measurement results within 500ms of completion",
            safety_class="B",
            acceptance_criteria="Display latency < 500ms measured via oscilloscope"
        ),
        Requirement(
            req_id="REQ-003",
            text="The system shall store up to 1000 measurement records with timestamp and patient ID",
            safety_class="A",
            acceptance_criteria="Verified storage of 1000 records without data loss"
        )
    ]

    # ===============================================================
    # Define RTM Entries (Requirement + Test Case Summary)
    # ===============================================================
    rtm_entries = [
        RTMEntry(
            req_id="REQ-001",
            test_case_summary="""
            TC-001: Verify temperature measurement at 35.0°C (lower boundary)
            TC-002: Verify temperature measurement at 42.0°C (upper boundary)
            TC-003: Verify temperature measurement at 37.0°C (nominal)
            TC-004: Verify accuracy within ±0.1°C using NIST-traceable reference thermometer
            TC-005: Verify measurement at 34.9°C (below range) triggers error
            TC-006: Verify measurement at 42.1°C (above range) triggers error
            """,
            test_case_count=6,
            test_ids="TC-001, TC-002, TC-003, TC-004, TC-005, TC-006"
        ),
        RTMEntry(
            req_id="REQ-002",
            test_case_summary="""
            TC-010: Verify display updates within 500ms after measurement completion
            TC-011: Verify display shows correct temperature value
            TC-012: Verify display shows correct units (°C)
            """,
            test_case_count=3,
            test_ids="TC-010, TC-011, TC-012"
        ),
        RTMEntry(
            req_id="REQ-003",
            test_case_summary="""
            TC-020: Store single measurement record
            TC-021: Store 1000 measurement records sequentially
            TC-022: Verify each record contains timestamp
            TC-023: Verify each record contains patient ID
            """,
            test_case_count=4,
            test_ids="TC-020, TC-021, TC-022, TC-023"
        )
    ]

    # ===============================================================
    # Run Batch RTM Review with Auto-Detected Ollama Ports
    # ===============================================================
    print("\n" + "="*80)
    print("RTM REVIEW - Evaluating Requirement Verification Coverage")
    print("="*80 + "\n")

    results = await run_batch_rtm_local_review(
        rtm_entries=rtm_entries,
        requirements=requirements,
        model="llama3.1",
        auto_detect_ports=True,  # Automatically detect active Ollama instances
        max_concurrent=3,
        temperature=0.0
    )

    # ===============================================================
    # Display Results
    # ===============================================================
    for result in results:
        print("\n" + "="*80)
        print(f"REQUIREMENT: {result.req_id}")
        print("="*80)

        # Calculate average score
        avg_score = sum([
            result.functional_coverage_score,
            result.input_output_coverage_score,
            result.boundary_coverage_score,
            result.negative_test_coverage_score,
            result.risk_coverage_score,
            result.traceability_completeness_score,
            result.acceptance_criteria_coverage_score,
            result.test_sufficiency_score,
            result.gap_analysis_score
        ]) / 9

        print(f"\n📊 COVERAGE SCORES:")
        print(f"  Overall Score:               {avg_score:.2f}")
        print(f"  Functional Coverage:         {result.functional_coverage_score:.2f}")
        print(f"  Input/Output Coverage:       {result.input_output_coverage_score:.2f}")
        print(f"  Boundary Coverage:           {result.boundary_coverage_score:.2f}")
        print(f"  Negative Test Coverage:      {result.negative_test_coverage_score:.2f}")
        print(f"  Risk Coverage:               {result.risk_coverage_score:.2f}")
        print(f"  Traceability Completeness:   {result.traceability_completeness_score:.2f}")
        print(f"  Acceptance Criteria Coverage:{result.acceptance_criteria_coverage_score:.2f}")
        print(f"  Test Sufficiency:            {result.test_sufficiency_score:.2f}")
        print(f"  Gap Analysis:                {result.gap_analysis_score:.2f}")

        print(f"\n📝 REVIEW SUMMARY:")
        print(f"{result.review_summary}")

        print(f"\n⚠️  VERIFICATION GAPS:")
        print(f"{result.verification_gaps}")

        print(f"\n💡 RECOMMENDATIONS:")
        print(f"{result.recommendations}")

        if result.issues:
            print(f"\n🔴 ISSUES IDENTIFIED:")
            for issue in result.issues[:5]:  # Show first 5 issues
                print(f"  - {issue}")
            if len(result.issues) > 5:
                print(f"  ... and {len(result.issues) - 5} more issues")

        print("\n" + "-"*80)

    print("\n✅ RTM Review Complete!\n")


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
