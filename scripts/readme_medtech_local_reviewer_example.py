"""
Example script demonstrating medtech test case review agent with Ollama (local execution).

This script shows how to use the MedTech Test Case Reviewer (Local/Ollama version) to evaluate
test cases against FDA/IEC 62304 best practices using local models with complete privacy.

Prerequisites:
    1. Install Ollama: https://ollama.ai
    2. Pull a model: ollama pull llama3.1
    3. Start Ollama: ollama serve

Usage:
    # Single Ollama instance
    python scripts/readme_medtech_local_reviewer_example.py

    # Multi-port with auto-detection (automatically finds running Ollama instances)
    python scripts/readme_medtech_local_reviewer_example.py --multi-port

Note:
    - With --multi-port, the script automatically detects all running Ollama instances
    - To start multiple Ollama instances, use: shell/gpus_setup_for_ollama.sh
    - Or manually start instances on different ports:
        Terminal 1: OLLAMA_HOST=0.0.0.0:11434 ollama serve
        Terminal 2: OLLAMA_HOST=0.0.0.0:11435 ollama serve
        Terminal 3: OLLAMA_HOST=0.0.0.0:11436 ollama serve
"""

import asyncio
import argparse
import pandas as pd
from aiswre.components.tc_review_agent_medtech_local import (
    get_medtech_local_reviewer_runnable,
    run_batch_medtech_local_test_case_review,
    detect_ollama_ports,
    TestCase,
    Requirement
)

# ============================================================================
# Configuration
# ============================================================================

MODEL = "mistral"  # Change to "llama3.1", "qwen2.5", or other Ollama model
SINGLE_PORT = "http://localhost:11434"
MULTI_PORTS = [
    "http://localhost:11434",
    "http://localhost:11435",
    "http://localhost:11436"
]

# Optional: Custom weights for score aggregation
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
# Sample Data
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

async def main(use_multi_port: bool = False):
    """Run medtech test case review using local Ollama."""

    print("=" * 70)
    print("MedTech Test Case Reviewer (Ollama/Local) - FDA/IEC 62304")
    print("=" * 70)
    print()
    print(f"Model: {MODEL}")
    print()

    # Create sample data
    test_cases, requirements = create_sample_data()

    print(f"Reviewing {len(test_cases)} test cases...")
    print()

    # Run batch review
    if use_multi_port:
        print("Using multi-port parallelization for faster processing...")
        print("Auto-detecting active Ollama instances...")
        print()

        # Auto-detect available ports
        results = await run_batch_medtech_local_test_case_review(
            test_cases=test_cases,
            requirements=requirements,
            model=MODEL,
            base_urls=None,  # Auto-detect
            weights=CUSTOM_WEIGHTS,
            auto_detect_ports=True,
            max_concurrent=10  # Allow up to 10 concurrent reviews
        )
    else:
        print("Using single port mode...")
        print(f"Port: {SINGLE_PORT}")
        print()

        results = await run_batch_medtech_local_test_case_review(
            test_cases=test_cases,
            requirements=requirements,
            model=MODEL,
            base_urls=[SINGLE_PORT],
            weights=CUSTOM_WEIGHTS,
            auto_detect_ports=False,
            max_concurrent=1
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
    output_path = "output/medtech_local_test_review_results.xlsx"
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
    parser = argparse.ArgumentParser(description="MedTech Test Case Reviewer (Ollama/Local)")
    parser.add_argument(
        "--multi-port",
        action="store_true",
        help="Use multiple Ollama ports for parallel execution"
    )

    args = parser.parse_args()

    asyncio.run(main(use_multi_port=args.multi_port))
