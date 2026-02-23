"""
Example script demonstrating Ollama-based requirements review with automatic port detection.

This script shows how to use OllamaPromptProcessor to review software requirements
against INCOSE best practices using local Ollama models with multi-port support.

Requirements:
    - Ollama installed and running (https://ollama.ai)
    - Model pulled: ollama pull llama3.1

Usage:
    # Single port (default)
    python scripts/readme_req_reviewer_local_example.py

    # Multi-port (requires multiple Ollama instances)
    # Terminal 1: OLLAMA_HOST=0.0.0.0:11434 ollama serve
    # Terminal 2: OLLAMA_HOST=0.0.0.0:11435 ollama serve
    # Terminal 3: OLLAMA_HOST=0.0.0.0:11436 ollama serve
    # Then run: python scripts/readme_req_reviewer_local_example.py --multi-port
"""

import asyncio
import argparse
import logging
import pandas as pd
from langchain_ollama import ChatOllama
from aiswre import utils
from aiswre.components import prompteval as pe
from aiswre.components.processors import OllamaPromptProcessor, df_to_prompt_items, process_json_responses
from aiswre.utils import load_prompt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ===============================================================
# Configuration
# ===============================================================
CONFIG = utils.load_config("config.yaml")
MODEL = "llama3.1"  # Change to your preferred Ollama model
MODEL_KWARGS = {
    "temperature": 0.0,
    "format": "json",
    "keep_alive": "1h"
}
PROMPT_TEMPLATE_PATH = CONFIG["FILE_LOCATIONS"]["PROMPT_TEMPLATE_PATH"]
PROMPT_NAME = CONFIG["PROMPT_TEMPLATE"]
OUTPUT_DIRECTORY = utils.make_output_directory(CONFIG["FILE_LOCATIONS"], "OUTPUT_FOLDER")
SYSTEM_PROMPT = load_prompt(PROMPT_TEMPLATE_PATH, PROMPT_NAME, "system")
USER_PROMPT_TEMPLATE = load_prompt(PROMPT_TEMPLATE_PATH, PROMPT_NAME, "user")
DATASET_FILE_PATH = CONFIG["FILE_LOCATIONS"]["DATASET_FILE_PATH"]
SELECTED_EVAL_FUNCS = CONFIG["SELECTED_EVAL_FUNCS"]
SELECTED_EVAL_WEIGHTS = CONFIG["SELECTED_EVAL_WEIGHTS"]
EVAL_CONFIG = pe.make_eval_config(pe, include_funcs=SELECTED_EVAL_FUNCS)

# Ollama port configuration
START_PORT = 11434
MAX_PORTS_TO_CHECK = 10


# ===============================================================
# Port Detection Utility
# ===============================================================

def detect_ollama_ports(
    base_port: int = 11434,
    max_ports: int = 10,
    host: str = "localhost"
) -> list:
    """
    Detect active Ollama instances by checking which ports respond to API calls.

    Args:
        base_port: Starting port to check (default: 11434)
        max_ports: Maximum number of ports to check (default: 10)
        host: Hostname to check (default: localhost)

    Returns:
        List of active port numbers (e.g., [11434, 11435, 11436])
    """
    import urllib.request
    import urllib.error

    active_ports = []

    for i in range(max_ports):
        port = base_port + i
        url = f"http://{host}:{port}/api/version"

        try:
            # Try to connect to the Ollama API version endpoint
            req = urllib.request.Request(url, method='GET')
            with urllib.request.urlopen(req, timeout=1) as response:
                if response.status == 200:
                    active_ports.append(port)
                    logging.info(f"✓ Found active Ollama instance at port {port}")
        except (urllib.error.URLError, OSError, TimeoutError):
            # Port is not responding, skip it
            pass

    if not active_ports:
        logging.warning(
            f"⚠ No active Ollama instances found on {host}:{base_port}-{base_port + max_ports - 1}. "
            f"Using default port {base_port}."
        )
        # Return default port as fallback
        return [base_port]

    logging.info(f"✓ Total active Ollama instances: {len(active_ports)}")
    return active_ports


# ===============================================================
# Async Runner
# ===============================================================

async def run_req_review_with_ollama(input_df, model, model_kwargs, num_ports=1, start_port=11434):
    """
    Run requirements review using OllamaPromptProcessor.

    Args:
        input_df: Input dataframe with requirements
        model: Ollama model name
        model_kwargs: Model configuration kwargs
        num_ports: Number of Ollama ports to use
        start_port: Starting port number

    Returns:
        DataFrame with review results
    """
    # Create ChatOllama client (base_url will be overridden by OllamaPromptProcessor)
    ollama_client = ChatOllama

    # Initialize processor
    processor = OllamaPromptProcessor(
        client=ollama_client,
        input_df=input_df,
        model=model,
        model_kwargs=model_kwargs
    )

    # Prepare items and IDs
    items = df_to_prompt_items(input_df, ["requirement_id", "requirements"])
    ids = [item["requirement_id"] for item in items]

    # Run batch processing with multi-port support
    print(f"\n{'='*70}")
    print(f"Processing {len(items)} requirements using {num_ports} Ollama instance(s)")
    print(f"Ports: {start_port} to {start_port + num_ports - 1}")
    print(f"{'='*70}\n")

    results = await processor.run_prompt_batch(
        system_message=SYSTEM_PROMPT,
        user_message_template=USER_PROMPT_TEMPLATE,
        prompt_name=PROMPT_NAME,
        items=items,
        ids=ids,
        start_port=start_port,
        num_ports=num_ports
    )

    # Process JSON responses
    results = process_json_responses(results, ids, PROMPT_NAME)
    return pd.DataFrame(results)


# ===============================================================
# Display Functions
# ===============================================================

def display_summary(review_df):
    """Display summary statistics of the review."""
    print(f"\n{'='*70}")
    print("REVIEW SUMMARY")
    print(f"{'='*70}")
    print(f"Total Requirements Reviewed: {len(review_df)}")

    if "weighted_value" in review_df.columns:
        avg_score = review_df["weighted_value"].mean()
        print(f"Average Weighted Score: {avg_score:.2f}")
        print(f"Min Score: {review_df['weighted_value'].min():.2f}")
        print(f"Max Score: {review_df['weighted_value'].max():.2f}")

    # Count failed evaluations
    failed_cols = [col for col in review_df.columns if col.startswith("failed_")]
    if failed_cols:
        total_failures = review_df[failed_cols].sum().sum()
        print(f"\nTotal Evaluation Failures: {int(total_failures)}")
        print(f"\nTop Failed Evaluations:")
        failures_summary = review_df[failed_cols].sum().sort_values(ascending=False).head(5)
        for col, count in failures_summary.items():
            if count > 0:
                eval_name = col.replace("failed_", "").replace("_", " ").title()
                print(f"  - {eval_name}: {int(count)}")

    print(f"{'='*70}\n")


# ===============================================================
# Main Execution
# ===============================================================

async def main_single_port():
    """Run requirements review using single Ollama instance."""
    print("\n" + "="*70)
    print("OLLAMA REQUIREMENTS REVIEWER - Single Port Mode")
    print("="*70)
    print(f"Model: {MODEL}")
    print(f"Port: {START_PORT}")
    print()

    # Load input data
    df_input = pd.read_excel(DATASET_FILE_PATH)
    print(f"Loaded {len(df_input)} requirements from {DATASET_FILE_PATH}")

    # Run review
    review_df = await run_req_review_with_ollama(
        input_df=df_input,
        model=MODEL,
        model_kwargs=MODEL_KWARGS,
        num_ports=1,
        start_port=START_PORT
    )

    # Run INCOSE evaluations
    print("\nRunning INCOSE evaluation functions...")
    review_df = pe.call_evals(review_df, col="requirements_review.proposed_rewrite", eval_config=EVAL_CONFIG)
    review_df = pe.get_failed_evals(review_df)
    pe.add_weighted_column(review_df, SELECTED_EVAL_FUNCS, SELECTED_EVAL_WEIGHTS, "weighted_value")

    # Display summary
    display_summary(review_df)

    # Save results
    output_file = f"{OUTPUT_DIRECTORY}/reviewed_requirements_ollama_single.xlsx"
    review_df.to_excel(output_file, index=False)
    print(f"✓ Results saved to: {output_file}\n")


async def main_multi_port():
    """Run requirements review using multiple Ollama instances with auto-detection."""
    print("\n" + "="*70)
    print("OLLAMA REQUIREMENTS REVIEWER - Multi-Port Mode (Auto-Detect)")
    print("="*70)
    print(f"Model: {MODEL}")
    print()

    # Detect active Ollama ports
    print("Detecting active Ollama instances...")
    active_ports = detect_ollama_ports(base_port=START_PORT, max_ports=MAX_PORTS_TO_CHECK)
    num_ports = len(active_ports)

    if num_ports == 1:
        print(f"\n⚠ Only 1 Ollama instance detected. Consider starting more instances for parallel processing.")
        print("To start multiple instances:")
        print("  Terminal 1: OLLAMA_HOST=0.0.0.0:11434 ollama serve")
        print("  Terminal 2: OLLAMA_HOST=0.0.0.0:11435 ollama serve")
        print("  Terminal 3: OLLAMA_HOST=0.0.0.0:11436 ollama serve")
        print()

    # Load input data
    df_input = pd.read_excel(DATASET_FILE_PATH)
    print(f"Loaded {len(df_input)} requirements from {DATASET_FILE_PATH}")

    # Run review
    review_df = await run_req_review_with_ollama(
        input_df=df_input,
        model=MODEL,
        model_kwargs=MODEL_KWARGS,
        num_ports=num_ports,
        start_port=active_ports[0]
    )

    # Run INCOSE evaluations
    print("\nRunning INCOSE evaluation functions...")
    review_df = pe.call_evals(review_df, col="requirements_review.proposed_rewrite", eval_config=EVAL_CONFIG)
    review_df = pe.get_failed_evals(review_df)
    pe.add_weighted_column(review_df, SELECTED_EVAL_FUNCS, SELECTED_EVAL_WEIGHTS, "weighted_value")

    # Display summary
    display_summary(review_df)

    # Save results
    output_file = f"{OUTPUT_DIRECTORY}/reviewed_requirements_ollama_multi.xlsx"
    review_df.to_excel(output_file, index=False)
    print(f"✓ Results saved to: {output_file}\n")


# ===============================================================
# Entry Point
# ===============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ollama-based Requirements Reviewer with Auto Port Detection"
    )
    parser.add_argument(
        "--multi-port",
        action="store_true",
        help="Use multi-port mode with automatic detection of active Ollama instances"
    )
    args = parser.parse_args()

    try:
        if args.multi_port:
            print("\n📡 Multi-Port Mode: Automatically detecting active Ollama instances...")
            asyncio.run(main_multi_port())
        else:
            print("\n🔌 Single Port Mode: Using default Ollama instance at port 11434")
            asyncio.run(main_single_port())
    except FileNotFoundError as e:
        print(f"\n❌ Error: Dataset file not found.")
        print(f"   Please check the path in config.yaml: {DATASET_FILE_PATH}")
        print(f"   Error details: {e}\n")
    except Exception as e:
        print(f"\n❌ Error occurred: {e}\n")
        raise
