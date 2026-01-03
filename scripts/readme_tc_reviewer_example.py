"""
Test Case Reviewer Example Using GraphProcessor.run_graph_batch()

This example demonstrates using the lower-level GraphProcessor API instead of
the run_batch_test_case_review() helper function. This approach provides more
control over item preparation and result processing.

Key Differences from run_batch_test_case_review():
1. Manual item preparation (create state dicts)
2. Direct GraphProcessor usage
3. More control over result extraction
4. Suitable for custom workflows
"""
import asyncio
import pandas as pd
from dotenv import dotenv_values
from langchain_openai import ChatOpenAI
from src.components.processors import GraphProcessor
from src.components.tc_review_agent.core import TestCase, Requirement
from src.components.tc_review_agent.pipeline import TestCaseReviewerRunnable

async def main():
    # ===============================================================
    # Configuration
    # ===============================================================
    DOT_ENV = dotenv_values(".env")

    # ===============================================================
    # Create sample data with requirements
    # ===============================================================
    sample_data = {
        'test_id': ['TC-001', 'TC-002', 'TC-003'],
        'test_description': [
            'Test Case: Verify user login with valid credentials. Steps: 1. Navigate to login page. 2. Enter valid username and password. 3. Click login button. Expected: User is redirected to dashboard.',
            'Test Case: Verify password reset functionality. Steps: 1. Click "Forgot Password". 2. Enter email address. 3. Check email for reset link. 4. Click link and enter new password. Expected: Password is successfully reset.',
            'Test Case: Verify shopping cart total calculation. Steps: 1. Add item A ($10) to cart. 2. Add item B ($20) to cart. 3. View cart. Expected: Total shows $30.'
        ],
        'requirement_id': ['REQ-AUTH-001', 'REQ-AUTH-002', 'REQ-CART-001'],
        'requirement_text': [
            'The system shall allow users to authenticate using a username and password combination. Upon successful authentication, the user shall be redirected to the main dashboard.',
            'The system shall provide a password reset mechanism that sends a secure reset link to the user\'s registered email address. The link shall be valid for 24 hours.',
            'The shopping cart shall calculate and display the total price of all items added to the cart, including any applicable taxes and shipping fees.'
        ]
    }

    input_df = pd.DataFrame(sample_data)

    print("=" * 70)
    print("Test Case Review Using GraphProcessor - Demo")
    print("=" * 70)
    print("\nInput Data:")
    print(f"Test Cases: {len(input_df)}")
    print(f"Columns: {list(input_df.columns)}\n")

    for i, row in input_df.iterrows():
        print(f"\n--- Test Case {i+1}: {row['test_id']} ---")
        print(f"Requirement: {row['requirement_id']}")
        print(f"Requirement Text: {row['requirement_text'][:80]}...")
        print(f"Test Description: {row['test_description'][:80]}...")

    print("\n" + "=" * 70)

    # ===============================================================
    # Step 1: Create the graph runnable
    # ===============================================================
    print("\n[Step 1] Creating TestCaseReviewerRunnable...")

    graph_runnable = TestCaseReviewerRunnable(
        client=ChatOpenAI(
            api_key=DOT_ENV["OPENAI_API_KEY"],
            model="gpt-4o-mini",
            temperature=0.3
        ),
        weights={
            "confidence_score": 1.0,
            "adequacy_score": 2.0,   # Weight adequacy higher
            "clarity_score": 1.0
        }
    )
    print("   Graph created successfully")

    # ===============================================================
    # Step 2: Initialize GraphProcessor
    # ===============================================================
    print("\n[Step 2] Initializing GraphProcessor...")

    processor = GraphProcessor(
        graph_runnable=graph_runnable,
        input_df=input_df,  # Required parameter
        output_dir="./output"
    )
    print("   GraphProcessor initialized")

    # ===============================================================
    # Step 3: Prepare items for graph execution
    # ===============================================================
    print("\n[Step 3] Preparing state items for graph execution...")

    # IMPORTANT: Each item must be a dict with keys matching the graph's state
    # The graph expects: requirement, test, trace_links (initialized to [])
    items = []
    test_ids = []

    for _, row in input_df.iterrows():
        # Create Pydantic models
        test_case = TestCase(
            test_id=row['test_id'],
            description=row['test_description']
        )

        requirement = Requirement(
            req_id=row['requirement_id'],
            text=row['requirement_text']
        )

        # Create state dict for this item
        state_item = {
            'test': test_case,
            'requirement': requirement,
            'trace_links': []  # Initialize empty list for evaluator results
        }

        items.append(state_item)
        test_ids.append(row['test_id'])

    print(f"   Prepared {len(items)} state items")

    # ===============================================================
    # Step 4: Execute batch using GraphProcessor.run_graph_batch()
    # ===============================================================
    print("\n[Step 4] Executing graph batch...")

    # This is the key method - runs all graphs asynchronously
    results = await processor.run_graph_batch(
        items=items,
        ids=test_ids,
        graph_name="TestCaseReview"
    )

    print(f"   Completed {len(results)} graph executions")

    # ===============================================================
    # Step 5: Process results
    # ===============================================================
    print("\n[Step 5] Processing results...")

    # Results are state dicts with 'final_result' containing the TraceLink
    result_rows = []

    for i, result_state in enumerate(results):
        # Extract the final TraceLink from the state
        if isinstance(result_state, dict) and 'final_result' in result_state:
            trace_link = result_state['final_result']

            # Build result row
            row_data = {
                'test_id': test_ids[i],
                'requirement_id': input_df.iloc[i]['requirement_id'],
                'trace_link': trace_link
            }

            # Flatten TraceLink attributes
            if trace_link and hasattr(trace_link, '__dict__'):
                for key, value in trace_link.__dict__.items():
                    row_data[f'trace_link.{key}'] = value

            result_rows.append(row_data)
        else:
            # Handle error case
            result_rows.append({
                'test_id': test_ids[i],
                'requirement_id': input_df.iloc[i]['requirement_id'],
                'error': 'Invalid result state'
            })

    results_df = pd.DataFrame(result_rows)
    print(f"   Created results DataFrame with {len(results_df)} rows")

    # ===============================================================
    # Step 6: Display detailed results
    # ===============================================================
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    for i, row in results_df.iterrows():
        print(f"\n{'=' * 70}")
        print(f"Test Case: {row['test_id']}")
        print(f"Requirement: {row['requirement_id']}")
        print(f"{'=' * 70}")

        if 'trace_link.confidence_score' in row:
            print(f"Traceability Score:  {row['trace_link.confidence_score']:.2f} / 1.00")
            print(f"Adequacy Score:      {row['trace_link.adequacy_score']:.2f} / 1.00")
            print(f"Clarity Score:       {row['trace_link.clarity_score']:.2f} / 1.00")
            print(f"\nRationale:")

            # Truncate long rationales
            rationale = row['trace_link.rationale']
            if len(rationale) > 300:
                print(f"{rationale[:300]}...")
            else:
                print(rationale)
        else:
            print(f"ERROR: {row.get('error', 'Unknown error')}")

    # ===============================================================
    # Step 7: Summary statistics
    # ===============================================================
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    score_cols = [
        'trace_link.confidence_score',
        'trace_link.adequacy_score',
        'trace_link.clarity_score'
    ]

    if all(col in results_df.columns for col in score_cols):
        print("\nAverage Scores:")
        for col in score_cols:
            score_name = col.replace('trace_link.', '').replace('_', ' ').title()
            avg_score = results_df[col].mean()
            print(f"  {score_name:20s}: {avg_score:.3f}")

        print("\nScore Distribution:")
        print(results_df[score_cols].describe())

    # ===============================================================
    # Step 8: Save results
    # ===============================================================
    output_path = './output/testcase_review_graphprocessor.xlsx'
    results_df.to_excel(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    # ===============================================================
    # Step 9: Recommendations
    # ===============================================================
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)

    threshold = 0.7
    for i, row in results_df.iterrows():
        if 'trace_link.confidence_score' in row:
            test_id = row['test_id']
            confidence = row['trace_link.confidence_score']
            adequacy = row['trace_link.adequacy_score']
            clarity = row['trace_link.clarity_score']

            if confidence < threshold or adequacy < threshold or clarity < threshold:
                print(f"\n! {test_id} needs improvement:")
                if confidence < threshold:
                    print(f"  - Low traceability ({confidence:.2f}): Review requirement alignment")
                if adequacy < threshold:
                    print(f"  - Low adequacy ({adequacy:.2f}): Add missing test conditions")
                if clarity < threshold:
                    print(f"  - Low clarity ({clarity:.2f}): Improve test documentation")

    # ===============================================================
    # Comparison: GraphProcessor vs Helper Function
    # ===============================================================
    print("\n" + "=" * 70)
    print("GRAPHPROCESSOR vs HELPER FUNCTION COMPARISON")
    print("=" * 70)
    print("""
GraphProcessor.run_graph_batch() Approach (THIS SCRIPT):
  + More control over item preparation
  + Direct access to state structure
  + Flexible result processing
  + Suitable for custom workflows
  - More manual setup required
  - More code to write

run_batch_test_case_review() Approach:
  + Simple, high-level API
  + Automatic DataFrame conversion
  + Built-in result formatting
  + Less code
  - Less flexibility
  - Opinionated structure

When to use GraphProcessor:
  - Custom state initialization needed
  - Non-standard DataFrame structures
  - Advanced result processing
  - Integration with other graph-based workflows
  - Need to inspect intermediate state

When to use run_batch_test_case_review():
  - Standard test case review workflow
  - DataFrame with standard columns
  - Quick prototyping
  - Simple use cases
""")

    print("\n" + "=" * 70)
    print("Evaluation complete!")
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(main())
