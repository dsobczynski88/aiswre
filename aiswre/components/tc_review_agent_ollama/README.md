# Ollama Test Case Review Agent

LangGraph-based test case reviewer using local Ollama models with support for multi-port parallel execution.

## Overview

This module provides automated test case review using locally-hosted Ollama LLMs. Unlike the OpenAI-based reviewers, this implementation:

- **Runs locally** - No API costs, full data privacy
- **Multi-port support** - Run multiple Ollama instances for true parallel execution
- **Async execution** - Compatible with GraphProcessor pattern
- **Structured outputs** - Uses Pydantic models for reliable parsing

### Evaluation Criteria

**Structure Evaluation:**
- Logical, sequential steps
- Clear documentation
- Steps are concise and jargon-free
- Numbered/ordered for execution
- Outcome aligns with expected results

**Objective/Completeness Evaluation:**
- All required components present (ID, title, preconditions, etc.)
- Clear objective aligned with requirement
- Positive and negative scenarios
- Expected results provide verification evidence
- Completeness score ≥ 80%

## Architecture

```
Input (TestCaseInput)
    ↓
┌────────────────────────────────┐
│  Parallel Evaluators (LLM)    │
│  ┌──────────────────────────┐ │
│  │ eval_structure           │ │
│  │ eval_objective           │ │
│  └──────────────────────────┘ │
└────────────────────────────────┘
    ↓
get_review_summary (LLM)
    ↓
aggregate_results (Parser)
    ↓
ReviewResult (structured output)
```

## Multi-Port Architecture

The key innovation is support for running each graph on a separate Ollama port:

```
Test Case 1 → Ollama :11434 → Graph Instance 1 ─┐
Test Case 2 → Ollama :11435 → Graph Instance 2 ─┼→ Results
Test Case 3 → Ollama :11436 → Graph Instance 3 ─┘
```

This enables **true parallel execution** with local models, dramatically improving throughput.

## Quick Start

### Prerequisites

1. **Install Ollama**: https://ollama.ai
2. **Pull a model**:
   ```bash
   ollama pull llama3.1
   # or
   ollama pull mistral
   # or
   ollama pull qwen2.5
   ```

### Single-Port Usage (Simple)

```python
import asyncio
from aiswre.components.tc_review_agent_ollama import (
    TestCaseInput,
    run_batch_with_graphprocessor
)

# Create test case
test = TestCaseInput(
    test_id="TC-001",
    test_case_text="""TC-001: Verify login
Steps: 1. Enter username 2. Enter password 3. Click login
Expected: User is logged in"""
)

# Run review
results = await run_batch_with_graphprocessor(
    test_cases=[test],
    model="llama3.1",
    base_url="http://localhost:11434"
)

# Access result
result = results[0]
print(f"Score: {result.overall_score}")
print(f"Structure: {result.structure_verdict}")
print(f"Objective: {result.objective_verdict}")
print(f"Summary: {result.review_summary}")
```

### Multi-Port Usage (Parallel)

**Step 1: Start multiple Ollama instances**

```bash
# Terminal 1
OLLAMA_HOST=0.0.0.0:11434 ollama serve

# Terminal 2
OLLAMA_HOST=0.0.0.0:11435 ollama serve

# Terminal 3
OLLAMA_HOST=0.0.0.0:11436 ollama serve
```

**Step 2: Run parallel reviews**

```python
import asyncio
from aiswre.components.tc_review_agent_ollama import (
    TestCaseInput,
    run_batch_ollama_test_case_review
)

# Define test cases
test_cases = [
    TestCaseInput(test_id="TC-001", test_case_text="..."),
    TestCaseInput(test_id="TC-002", test_case_text="..."),
    TestCaseInput(test_id="TC-003", test_case_text="..."),
    # ... more test cases
]

# Run with multi-port parallelization
results = await run_batch_ollama_test_case_review(
    test_cases=test_cases,
    model="llama3.1",
    base_urls=[
        "http://localhost:11434",
        "http://localhost:11435",
        "http://localhost:11436"
    ],
    max_concurrent=3
)
```

## Usage Patterns

### Using GraphProcessor (Recommended for Single Port)

```python
from aiswre.components.processors import GraphProcessor
from aiswre.components.tc_review_agent_ollama import get_ollama_reviewer_runnable

# Create reviewer graph
graph = get_ollama_reviewer_runnable(
    model="llama3.1",
    base_url="http://localhost:11434"
)

# Use with GraphProcessor
processor = GraphProcessor(
    graph_runnable=graph,
    input_file="test_cases.xlsx"
)

# Prepare items
items = [
    {
        "test_case_id": "TC-001",
        "test_case_text": "...",
        "messages": [],
        "structure": "",
        "objective": "",
        "review_summary": "",
        "final_result": None
    }
    # ... more items
]

# Run batch
results = await processor.run_graph_batch(
    items=items,
    ids=["TC-001", "TC-002", ...],
    graph_name="OllamaTestReview"
)
```

### Direct Graph Invocation

```python
from aiswre.components.tc_review_agent_ollama import get_ollama_reviewer_runnable

reviewer = get_ollama_reviewer_runnable(model="llama3.1")

initial_state = {
    "messages": [],
    "test_case_id": "TC-001",
    "test_case_text": "Your test case here...",
    "structure": "",
    "objective": "",
    "review_summary": "",
    "final_result": None
}

# Async
final_state = await reviewer.ainvoke(initial_state)
result = final_state["final_result"]

# Sync
final_state = reviewer.invoke(initial_state)
result = final_state["final_result"]
```

## Output Format

Each `ReviewResult` contains:

```python
{
    "test_id": "TC-001",

    # Structure evaluation
    "structure_verdict": "complete|partial|inadequate",
    "structure_gaps": ["gap 1", "gap 2", ...],
    "structure_recommendations": ["rec 1", "rec 2", ...],

    # Objective evaluation
    "objective_verdict": "complete|partial|inadequate",
    "objective_gaps": ["gap 1", "gap 2", ...],
    "objective_recommendations": ["rec 1", "rec 2", ...],

    # Summary
    "review_summary": "Comprehensive review summary...",

    # Composite score
    "overall_score": 0.75,  # complete=1.0, partial=0.5, inadequate=0.0

    # Metadata
    "link_type": "OllamaReview"
}
```

## Performance Comparison

| Configuration | Test Cases | Time (approx) |
|---------------|------------|---------------|
| Single port (sequential) | 10 | ~15-20 min |
| Single port (GraphProcessor) | 10 | ~15-20 min |
| Multi-port (3 instances) | 10 | ~5-7 min |
| Multi-port (5 instances) | 10 | ~3-4 min |

*Times vary based on model size and hardware*

## Recommended Models

| Model | Size | Speed | Quality | Use Case |
|-------|------|-------|---------|----------|
| llama3.1:8b | 4.7GB | Fast | Good | Development/testing |
| mistral:7b | 4.1GB | Fast | Good | General use |
| qwen2.5:7b | 4.7GB | Fast | Very Good | Recommended |
| llama3.1:70b | 40GB | Slow | Excellent | Production (requires GPU) |

## Advanced Configuration

### Custom Temperature

```python
reviewer = get_ollama_reviewer_runnable(
    model="llama3.1",
    temperature=0.2  # Higher = more creative, Lower = more deterministic
)
```

### Custom Ollama Host

```python
# Remote Ollama server
reviewer = get_ollama_reviewer_runnable(
    model="llama3.1",
    base_url="http://192.168.1.100:11434"
)
```

### Integration with GraphProcessor

```python
from aiswre.components.processors import GraphProcessor

# Create graph
graph = get_ollama_reviewer_runnable(model="llama3.1")

# Use GraphProcessor for async batch execution
processor = GraphProcessor(graph_runnable=graph, input_file="tests.xlsx")
results = await processor.run_graph_batch(items, ids, "OllamaReview")
```

## Troubleshooting

### "Connection refused" Error
- Ensure Ollama is running: `ollama serve`
- Check port availability: `lsof -i :11434` (macOS/Linux)

### Slow Performance
- Use smaller models (7b-8b) for faster inference
- Run on GPU if available
- Use multi-port mode for parallel execution

### JSON Parsing Errors
- The module uses structured outputs which are more reliable
- If errors persist, check Ollama version (recommend 0.1.0+)
- Try a different model (qwen2.5 is very good at structured output)

### Out of Memory
- Use smaller models
- Reduce `max_concurrent` parameter
- Close other applications

## Files

- `core.py` - Pydantic models and state definitions
- `evaluators.py` - Factory functions for evaluator nodes
- `nodes.py` - Result aggregation logic
- `pipeline.py` - LangGraph orchestration and batch execution
- `__init__.py` - Public API

## Example Script

See `scripts/readme_ollama_reviewer_example.py` for a complete working example with both single-port and multi-port modes.

## Requirements

- Python 3.8+
- Ollama installed and running
- `langchain-ollama` or `langchain-community`
- `langgraph`
- `pydantic`

## License

Same as parent project.
