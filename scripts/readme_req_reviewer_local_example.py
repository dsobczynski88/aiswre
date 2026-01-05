"""
Example script demonstrating Ollama-based test case review agent.

This script shows both single-port and multi-port Ollama execution.

Requirements:
    - Ollama installed and running (https://ollama.ai)
    - Model pulled: ollama pull llama3.1

Usage:
    # Single port (default)
    python scripts/readme_ollama_reviewer_example.py

    # Multi-port (requires multiple Ollama instances)
    # Terminal 1: OLLAMA_HOST=0.0.0.0:11434 ollama serve
    # Terminal 2: OLLAMA_HOST=0.0.0.0:11435 ollama serve
    # Terminal 3: OLLAMA_HOST=0.0.0.0:11436 ollama serve
    # Then run: python scripts/readme_ollama_reviewer_example.py --multi-port
"""

import asyncio
import pandas as pd
import argparse
from src.components.tc_review_agent_ollama import (
    TestCaseInput,
    run_batch_ollama_test_case_review,
    run_batch_with_graphprocessor
)

# ============================================================================
# Configuration
# ============================================================================

MODEL = "llama3.1"  # Change to your preferred Ollama model
TEMPERATURE = 0.0

# Multi-port configuration (for parallel execution)
OLLAMA_PORTS = [
    "http://localhost:11434",
    "http://localhost:11435",
    "http://localhost:11436",
]

# ============================================================================
# Sample Test Cases
# ============================================================================

SAMPLE_TEST_CASES = [
    TestCaseInput(
        test_id="TC-001",
        test_case_text="""TC-001: Verify user login functionality
Title: User Login with Valid Credentials
Preconditions: User account exists and is active
Input Data: username='valid_user', password='valid_pass'
Steps:
1. Navigate to the login page.
2. Enter valid username.
3. Enter valid password.
4. Click 'Login'.
Expected Result: User is redirected to the dashboard; session is created; user name appears in header."""
    ),
    TestCaseInput(
        test_id="TC-002",
        test_case_text="""TC-002: Test the password reset feature
Title: Password Reset
Steps:
1. Click forgot password
2. Enter email
3. Check result
Expected Result: It works"""
    ),
    TestCaseInput(
        test_id="TC-003",
        test_case_text="""TC-003: Verify shopping cart functionality
Title: Add Item to Shopping Cart
Preconditions:
- User is logged in
- Product catalog is populated with test data
- Shopping cart is empty
Input Data:
- Product ID: "PROD-12345"
- Quantity: 2
- Product Name: "Wireless Mouse"
- Unit Price: $29.99
Steps:
1. Navigate to product catalog page
2. Locate product "Wireless Mouse" (PROD-12345)
3. Verify product is in stock (stock quantity > 0)
4. Click "Add to Cart" button
5. Enter quantity: 2
6. Click "Confirm" button
7. Wait for cart update confirmation message
8. Navigate to shopping cart page
9. Verify cart contents
Expected Result:
- Cart contains 1 line item
- Line item product name: "Wireless Mouse"
- Line item product ID: "PROD-12345"
- Line item quantity: 2
- Line item unit price: $29.99
- Line item subtotal: $59.98 (2 × $29.99)
- Cart total: $59.98
- Success message displayed: "Item added to cart"
Postconditions:
- Shopping cart contains added item
- Product stock quantity decremented by 2"""
    ),
    TestCaseInput(
        test_id="TC-004",
        test_case_text="""TC-004: API endpoint test
Title: Test GET /api/users endpoint
Steps:
1. Send GET request
2. Check response
Expected: Status 200"""
    ),
    TestCaseInput(
        test_id="TC-005",
        test_case_text="""TC-005: Verify data export functionality
Title: Export User Data to CSV
Preconditions:
- User is logged in as administrator
- Database contains test user data (minimum 100 records)
- Export module is enabled
Input Data:
- Date range: 2024-01-01 to 2024-12-31
- Export format: CSV
- Fields to export: [user_id, username, email, registration_date, status]
Steps:
1. Navigate to Admin Dashboard
2. Click "Data Export" menu item
3. Select "User Data" from export type dropdown
4. Set date range: Start=2024-01-01, End=2024-12-31
5. Select export format: "CSV"
6. Check all desired fields in field selector
7. Click "Export" button
8. Wait for export processing (max 30 seconds)
9. Verify download prompt appears
10. Save file to local directory
11. Open exported CSV file
12. Verify file structure and content
Expected Result:
- Export completes within 30 seconds
- Download prompt displays filename: "user_data_export_YYYYMMDD_HHMMSS.csv"
- CSV file size > 0 bytes
- CSV contains header row: user_id,username,email,registration_date,status
- CSV contains data rows matching filter criteria (users registered in 2024)
- All exported records have registration_date between 2024-01-01 and 2024-12-31
- Data accuracy: spot-check 5 random records against database
- No duplicate records
- No missing required fields
Postconditions:
- Export log entry created with timestamp, user, record count
- Original database data unchanged
- Exported file stored in user's download directory"""
    )
]


# ============================================================================
# Display Functions
# ============================================================================

def display_result(result):
    """Display a single review result."""
    print(f"\n{'='*70}")
    print(f"TEST CASE: {result.test_id}")
    print(f"{'='*70}")
    print(f"Overall Score: {result.overall_score:.2f} / 1.00")
    print()

    print("STRUCTURE EVALUATION:")
    print(f"  Verdict: {result.structure_verdict.upper()}")
    if result.structure_gaps:
        print(f"  Gaps ({len(result.structure_gaps)}):")
        for gap in result.structure_gaps[:3]:
            print(f"    - {gap}")
        if len(result.structure_gaps) > 3:
            print(f"    ... and {len(result.structure_gaps) - 3} more")
    if result.structure_recommendations:
        print(f"  Recommendations ({len(result.structure_recommendations)}):")
        for rec in result.structure_recommendations[:3]:
            print(f"    - {rec}")
        if len(result.structure_recommendations) > 3:
            print(f"    ... and {len(result.structure_recommendations) - 3} more")
    print()

    print("OBJECTIVE/COMPLETENESS EVALUATION:")
    print(f"  Verdict: {result.objective_verdict.upper()}")
    if result.objective_gaps:
        print(f"  Gaps ({len(result.objective_gaps)}):")
        for gap in result.objective_gaps[:3]:
            print(f"    - {gap}")
        if len(result.objective_gaps) > 3:
            print(f"    ... and {len(result.objective_gaps) - 3} more")
    if result.objective_recommendations:
        print(f"  Recommendations ({len(result.objective_recommendations)}):")
        for rec in result.objective_recommendations[:3]:
            print(f"    - {rec}")
        if len(result.objective_recommendations) > 3:
            print(f"    ... and {len(result.objective_recommendations) - 3} more")
    print()

    print("REVIEW SUMMARY:")
    summary_preview = result.review_summary[:300] + "..." if len(result.review_summary) > 300 else result.review_summary
    print(f"  {summary_preview}")
    print()


def display_summary_statistics(results):
    """Display summary statistics."""
    print(f"\n{'='*70}")
    print("SUMMARY STATISTICS")
    print(f"{'='*70}")
    print(f"Total Test Cases Reviewed: {len(results)}")
    print()

    avg_score = sum(r.overall_score for r in results) / len(results) if results else 0
    print(f"Average Overall Score: {avg_score:.2f}")
    print()

    structure_verdicts = {}
    objective_verdicts = {}
    for r in results:
        structure_verdicts[r.structure_verdict] = structure_verdicts.get(r.structure_verdict, 0) + 1
        objective_verdicts[r.objective_verdict] = objective_verdicts.get(r.objective_verdict, 0) + 1

    print("Structure Verdicts:")
    for verdict, count in structure_verdicts.items():
        print(f"  {verdict.upper()}: {count}")
    print()

    print("Objective Verdicts:")
    for verdict, count in objective_verdicts.items():
        print(f"  {verdict.upper()}: {count}")
    print()

    total_gaps = sum(len(r.structure_gaps) + len(r.objective_gaps) for r in results)
    total_recs = sum(len(r.structure_recommendations) + len(r.objective_recommendations) for r in results)
    print(f"Total Gaps Identified: {total_gaps}")
    print(f"Total Recommendations: {total_recs}")
    print()


# ============================================================================
# Main Execution
# ============================================================================

async def main_single_port():
    """Run reviews using single Ollama instance (GraphProcessor pattern)."""
    print("="*70)
    print("OLLAMA TEST CASE REVIEWER - Single Port Mode")
    print("="*70)
    print(f"Model: {MODEL}")
    print(f"Ollama URL: http://localhost:11434")
    print(f"Test Cases: {len(SAMPLE_TEST_CASES)}")
    print()

    print("Starting reviews...")
    results = await run_batch_with_graphprocessor(
        test_cases=SAMPLE_TEST_CASES,
        model=MODEL,
        base_url="http://localhost:11434",
        temperature=TEMPERATURE
    )

    # Display results
    for result in results:
        display_result(result)

    display_summary_statistics(results)

    # Export to Excel
    export_results_to_excel(results, "output/ollama_review_single_port.xlsx")


async def main_multi_port():
    """Run reviews using multiple Ollama instances (true parallel execution)."""
    print("="*70)
    print("OLLAMA TEST CASE REVIEWER - Multi-Port Mode")
    print("="*70)
    print(f"Model: {MODEL}")
    print(f"Ollama URLs: {OLLAMA_PORTS}")
    print(f"Test Cases: {len(SAMPLE_TEST_CASES)}")
    print(f"Max Concurrent: {len(OLLAMA_PORTS)}")
    print()

    print("Starting parallel reviews...")
    results = await run_batch_ollama_test_case_review(
        test_cases=SAMPLE_TEST_CASES,
        model=MODEL,
        base_urls=OLLAMA_PORTS,
        temperature=TEMPERATURE,
        max_concurrent=len(OLLAMA_PORTS)
    )

    # Display results
    for result in results:
        display_result(result)

    display_summary_statistics(results)

    # Export to Excel
    export_results_to_excel(results, "output/ollama_review_multi_port.xlsx")


def export_results_to_excel(results, filename):
    """Export results to Excel file."""
    data = []
    for r in results:
        data.append({
            "test_id": r.test_id,
            "overall_score": r.overall_score,
            "structure_verdict": r.structure_verdict,
            "structure_gaps_count": len(r.structure_gaps),
            "structure_gaps": " | ".join(r.structure_gaps),
            "structure_recommendations": " | ".join(r.structure_recommendations),
            "objective_verdict": r.objective_verdict,
            "objective_gaps_count": len(r.objective_gaps),
            "objective_gaps": " | ".join(r.objective_gaps),
            "objective_recommendations": " | ".join(r.objective_recommendations),
            "review_summary": r.review_summary
        })

    df = pd.DataFrame(data)
    df.to_excel(filename, index=False)
    print(f"Results exported to: {filename}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ollama Test Case Reviewer Example")
    parser.add_argument(
        "--multi-port",
        action="store_true",
        help="Use multi-port mode with parallel Ollama instances"
    )
    args = parser.parse_args()

    if args.multi_port:
        print("\nNOTE: Multi-port mode requires multiple Ollama instances running.")
        print("Start them with:")
        print("  Terminal 1: OLLAMA_HOST=0.0.0.0:11434 ollama serve")
        print("  Terminal 2: OLLAMA_HOST=0.0.0.0:11435 ollama serve")
        print("  Terminal 3: OLLAMA_HOST=0.0.0.0:11436 ollama serve")
        print()
        asyncio.run(main_multi_port())
    else:
        asyncio.run(main_single_port())
