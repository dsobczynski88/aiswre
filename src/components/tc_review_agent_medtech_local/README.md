# MedTech Test Case Review Agent (Ollama/Local)

LangGraph-based AI agent for evaluating medical device software test cases against FDA/IEC 62304 best practices using **local Ollama models**. Provides comprehensive compliance review with complete privacy and no API costs.

## Overview

This module combines the comprehensive 11-evaluator medtech analysis with local Ollama execution, providing:

- ✅ **Full Privacy**: All processing happens locally on your machine
- ✅ **No API Costs**: Uses open-source Ollama models (Llama 3.1, Mistral, Qwen, etc.)
- ✅ **Comprehensive Analysis**: Same 11 specialized evaluators as the OpenAI version
- ✅ **Multi-Port Parallelization**: Supports multiple Ollama instances for faster batch processing
- ✅ **FDA/IEC 62304 Aligned**: Evaluates against medical device software testing standards

### 11 Specialized Evaluators

Organized into three categories:

#### 1. General Integrity & Structure
- **Unambiguity**: Are instructions clear enough for someone with no prior knowledge?
- **Independence**: Is the test self-contained and does it clean up its own data?
- **Pre-conditions**: Are initial states (Power, Network, Database) explicitly defined?
- **Post-conditions**: Does the test return the system to a safe/neutral state?

#### 2. Coverage & Technique
- **Technique Application**: Does the test utilize EP, BVA, or Decision Tables appropriately?
- **Negative Testing**: Are there tests for Invalid Inputs, Timeouts, and Error States?
- **Boundary Checks**: Are edges (Min, Min-1, Max, Max+1) explicitly verified?
- **Risk Verification**: Does the test verify effectiveness of risk controls?

#### 3. Traceability & Compliance
- **Traceability**: Is there a correct link to a Requirement ID or Risk ID?
- **Safety Class Rigor**: For Class C units, is there MC/DC coverage?
- **Objective Evidence**: Are expected results specific values (e.g., "5V +/- 0.1V") vs subjective ("correct")?

## Architecture

```
Input (TestCase + Requirement)
    ↓
┌─────────────────────────────────────────────┐
│  11 Parallel Evaluators (Ollama LLM)       │
│  ┌───────────────────────────────────────┐ │
│  │ General Integrity (4)                 │ │
│  │ Coverage & Technique (4)              │ │
│  │ Traceability & Compliance (3)         │ │
│  └───────────────────────────────────────┘ │
└─────────────────────────────────────────────┘
    ↓
Response Conversion (Dict → MedtechTraceLink)
    ↓
Aggregator (Ollama-Powered Analysis)
  - Weighted score averaging
  - Ollama LLM synthesis of all findings
  - Comprehensive review summary
  - Actionable improvement recommendations
    ↓
MedtechTraceLink (11 scores + issues + summary + improvements)
```

## Prerequisites

### 1. Install Ollama

Download and install Ollama from [ollama.ai](https://ollama.ai)

```bash
# macOS/Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Windows: Download installer from ollama.ai
```

### 2. Pull a Model

```bash
# Recommended: Llama 3.1 (8B parameters)
ollama pull llama3.1

# Alternatives:
ollama pull mistral        # Fast, good for smaller machines
ollama pull qwen2.5        # Excellent for structured output
ollama pull llama3.2       # Latest Llama version
```

### 3. Start Ollama Server

```bash
# Single instance (default)
ollama serve

# Multi-port for parallel execution (optional but recommended for batches)
# Terminal 1
OLLAMA_HOST=0.0.0.0:11434 ollama serve

# Terminal 2
OLLAMA_HOST=0.0.0.0:11435 ollama serve

# Terminal 3
OLLAMA_HOST=0.0.0.0:11436 ollama serve
```

## Quick Start

### Basic Usage (Single Test Case)

```python
import asyncio
from src.components.tc_review_agent_medtech_local import (
    get_medtech_local_reviewer_runnable,
    TestCase,
    Requirement
)

# Create test case
test = TestCase(
    test_id="TC-001",
    description="Verify temperature alarm triggers when reading exceeds 38.5°C",
    preconditions="System powered on, sensor connected, baseline temp 37.0°C",
    steps="1. Set temperature to 38.6°C\n2. Wait 2 seconds\n3. Observe alarm",
    expected_result="Audible alarm sounds within 2 seconds, display shows 'TEMP HIGH'",
    postconditions="Reset alarm, return temperature to 37.0°C",
    test_type="System",
    technique="BVA"
)

# Create requirement
req = Requirement(
    req_id="REQ-101",
    text="System shall trigger an audible alarm when temperature exceeds 38.5°C",
    safety_class="B",
    risk_id="RISK-015"
)

# Initialize reviewer
reviewer = get_medtech_local_reviewer_runnable(
    model="llama3.1",
    base_url="http://localhost:11434"
)

# Run review
async def main():
    state = {
        "test": test,
        "requirement": req,
        "medtech_links": [],
        "final_result": None
    }

    result = await reviewer.ainvoke(state)
    final = result["final_result"]

    print(f"Test ID: {final.test_id}")
    print(f"Unambiguity Score: {final.unambiguity_score}")
    print(f"Traceability Score: {final.traceability_score}")
    print(f"\nReview Summary:\n{final.review_summary}")
    print(f"\nRecommended Improvements:\n{final.test_case_improvements}")
    print(f"\nIssues: {final.issues}")

asyncio.run(main())
```

### Batch Processing with Multi-Port Parallelization

```python
import asyncio
from src.components.tc_review_agent_medtech_local import (
    run_batch_medtech_local_test_case_review,
    TestCase,
    Requirement
)

# Create test cases and requirements
test_cases = [...]  # List of TestCase objects
requirements = [...]  # List of Requirement objects

# Run batch review with multiple Ollama ports for true parallelization
async def main():
    results = await run_batch_medtech_local_test_case_review(
        test_cases=test_cases,
        requirements=requirements,
        model="llama3.1",
        base_urls=[
            "http://localhost:11434",
            "http://localhost:11435",
            "http://localhost:11436"
        ],
        max_concurrent=3
    )

    # Process results
    for result in results:
        print(f"Test {result.test_id}: Avg Score = {sum([
            result.unambiguity_score, result.independence_score,
            result.preconditions_score, result.postconditions_score,
            result.technique_application_score, result.negative_testing_score,
            result.boundary_checks_score, result.risk_verification_score,
            result.traceability_score, result.safety_class_rigor_score,
            result.objective_evidence_score
        ]) / 11:.2f}")

asyncio.run(main())
```

## Custom Weights

Adjust score weights to match your regulatory priorities:

```python
# Example: Emphasize compliance over structure
weights = {
    # General Integrity & Structure (20% total)
    "unambiguity_score": 0.08,
    "independence_score": 0.04,
    "preconditions_score": 0.04,
    "postconditions_score": 0.04,

    # Coverage & Technique (30% total)
    "technique_application_score": 0.08,
    "negative_testing_score": 0.08,
    "boundary_checks_score": 0.08,
    "risk_verification_score": 0.06,

    # Traceability & Compliance (50% total - emphasized)
    "traceability_score": 0.20,
    "safety_class_rigor_score": 0.15,
    "objective_evidence_score": 0.15,
}

reviewer = get_medtech_local_reviewer_runnable(
    model="llama3.1",
    weights=weights
)
```

## Output Format

Each `MedtechTraceLink` contains:

```python
{
    "test_id": "TC-001",
    "req_id": "REQ-101",

    # Scores (0-1)
    "unambiguity_score": 0.85,
    "independence_score": 0.90,
    "preconditions_score": 0.75,
    "postconditions_score": 0.80,
    "technique_application_score": 0.95,
    "negative_testing_score": 0.60,
    "boundary_checks_score": 0.90,
    "risk_verification_score": 0.85,
    "traceability_score": 1.0,
    "safety_class_rigor_score": 0.70,
    "objective_evidence_score": 0.85,

    # Metadata
    "issues": ["Missing timeout test", "Subjective expected result"],
    "rationale": "Detailed explanation from each evaluator...",
    "link_type": "MedtechLocalAggregated",

    # Ollama-Generated Aggregated Analysis
    "review_summary": "Comprehensive paragraph synthesizing all 11 evaluator findings...",
    "test_case_improvements": "Specific, actionable recommendations..."
}
```

## Performance Considerations

### Single Port vs Multi-Port

| Configuration | Test Cases | Time (Llama 3.1 8B) | Speedup |
|---------------|-----------|---------------------|---------|
| Single port   | 10        | ~15 minutes         | 1x      |
| 3 ports       | 10        | ~5 minutes          | 3x      |
| 5 ports       | 10        | ~3 minutes          | 5x      |

### Model Selection

| Model | Size | Speed | Quality | Best For |
|-------|------|-------|---------|----------|
| llama3.1 | 8B | Medium | Excellent | General use, best balance |
| mistral | 7B | Fast | Good | Quick reviews, resource-constrained |
| qwen2.5 | 7B | Fast | Excellent | Structured output, compliance |
| llama3.2 | 3B | Very Fast | Good | High-volume, simple cases |

### Hardware Requirements

- **Minimum**: 8GB RAM, CPU-only (slow but functional)
- **Recommended**: 16GB RAM + GPU (NVIDIA/AMD/Apple Silicon)
- **Optimal**: 32GB RAM + GPU with 8GB+ VRAM

## Comparison with OpenAI Version

| Feature | MedTech Local (Ollama) | MedTech (OpenAI) |
|---------|------------------------|------------------|
| Privacy | ✅ 100% local | ❌ Cloud-based |
| Cost | ✅ Free (hardware only) | ❌ ~$0.10-0.50 per test case |
| Speed (single) | ⚠️ Slower (2-3 min/test) | ✅ Faster (30-60 sec/test) |
| Speed (parallel) | ✅ Scales with ports | ⚠️ API rate limits |
| Quality | ✅ Excellent (Llama 3.1+) | ✅ Excellent (GPT-4) |
| Setup | ⚠️ Requires Ollama install | ✅ API key only |
| Internet | ✅ Not required | ❌ Required |

## Troubleshooting

### Issue: "Connection refused" error

**Solution**: Ensure Ollama is running
```bash
ollama serve
```

### Issue: Slow performance

**Solutions**:
1. Use a smaller model (`ollama pull mistral`)
2. Enable GPU acceleration (automatic if available)
3. Use multi-port parallelization for batches

### Issue: Out of memory errors

**Solutions**:
1. Use a smaller model (llama3.2 3B instead of llama3.1 8B)
2. Reduce `max_concurrent` parameter
3. Close other applications

### Issue: Poor quality results

**Solutions**:
1. Use a larger, more capable model (`ollama pull llama3.1`)
2. Adjust temperature (try 0.1 or 0.2 instead of 0.0)
3. Provide more detailed test case information

## Example Script

See `scripts/readme_medtech_local_reviewer_example.py` for a complete working example.

## Files

- `core.py` - Pydantic models (TestCase, Requirement, MedtechTraceLink, Response schemas)
- `evaluators.py` - Factory functions for 11 Ollama-based evaluator nodes
- `nodes.py` - Response conversion and Ollama-powered aggregator
- `pipeline.py` - LangGraph orchestration with multi-port support
- `__init__.py` - Public API exports

## Requirements

- Python 3.8+
- `langchain-ollama` (or `langchain-community`)
- `langgraph`
- `pydantic`
- Ollama installed and running

## License

Same as parent project.
