## # RTM Review Agent (Ollama/Local Version)

A LangGraph-based agent for reviewing **Requirement Traceability Matrices (RTM)** using local Ollama models. This agent evaluates how well a suite of test cases verifies a requirement, aligned with FDA/IEC 62304 verification principles.

## Overview

Unlike test case review agents that evaluate individual test cases, this agent:
- Takes a **requirement** and a **summary of test cases** traced to that requirement
- Evaluates **verification coverage** - how completely the test suite verifies the requirement
- Identifies **verification gaps** - what aspects of the requirement are NOT adequately tested
- Provides **actionable recommendations** to improve coverage

This is designed for reviewing **requirement traceability matrices** where each row contains:
- Requirement ID + Text
- Summary/description of all test cases that verify that requirement

## Architecture

### Parallel Evaluators + Aggregator Pattern

The agent uses 9 parallel evaluators organized into 3 categories:

**Coverage Evaluators (4)**
1. **Functional Coverage** - Are all functional aspects of the requirement verified?
2. **Input/Output Coverage** - Are all inputs tested and outputs verified?
3. **Boundary Coverage** - Are boundary conditions tested (min, max, edges)?
4. **Negative Test Coverage** - Are error cases and invalid inputs tested?

**Risk & Traceability (3)**
5. **Risk Coverage** - Are risk controls verified (if requirement has Risk ID)?
6. **Traceability Completeness** - Is the req-to-test mapping clear and complete?
7. **Acceptance Criteria Coverage** - Are all acceptance criteria verified?

**Sufficiency & Gaps (2)**
8. **Test Sufficiency** - Are there enough tests to fully verify the requirement?
9. **Gap Analysis** - What critical/moderate verification gaps exist?

All evaluators run in **parallel** using LangGraph, then results are **aggregated** using an Ollama-powered LLM to generate:
- Comprehensive review summary
- Specific verification gaps
- Actionable recommendations

## Installation

```bash
# Install aiswre package
pip install -e .

# Install Ollama (if not already installed)
# Visit https://ollama.ai for installation instructions

# Pull a recommended model
ollama pull llama3.1
```

## Quick Start

### Single RTM Entry Review

```python
import asyncio
from aiswre.components.rtm_review_agent_medtech_local import (
    Requirement,
    RTMEntry,
    get_rtm_local_reviewer_runnable
)

# Define requirement
requirement = Requirement(
    req_id="REQ-001",
    text="The system shall measure temperature in the range 35.0-42.0°C with accuracy ±0.1°C",
    safety_class="C",
    risk_id="RISK-012"
)

# Define RTM entry (requirement + test case summary)
rtm_entry = RTMEntry(
    req_id="REQ-001",
    test_case_summary="""
    TC-001: Verify temperature measurement at 35.0°C (lower boundary)
    TC-002: Verify temperature measurement at 42.0°C (upper boundary)
    TC-003: Verify temperature measurement at 37.0°C (nominal)
    TC-004: Verify accuracy within ±0.1°C using calibrated reference
    """,
    test_case_count=4,
    test_ids="TC-001, TC-002, TC-003, TC-004"
)

# Create reviewer
reviewer = get_rtm_local_reviewer_runnable(
    model="llama3.1",
    base_url="http://localhost:11434"
)

# Build state
state = {
    "requirement": requirement,
    "rtm_entry": rtm_entry,
    "raw_evaluator_responses": [],
    "rtm_links": [],
    "final_result": None
}

# Run review
async def main():
    result = await reviewer.ainvoke(state)
    final = result["final_result"]

    print(f"Requirement: {final.req_id}")
    print(f"Functional Coverage Score: {final.functional_coverage_score:.2f}")
    print(f"Boundary Coverage Score: {final.boundary_coverage_score:.2f}")
    print(f"Gap Analysis Score: {final.gap_analysis_score:.2f}")
    print(f"\nReview Summary:\n{final.review_summary}")
    print(f"\nVerification Gaps:\n{final.verification_gaps}")
    print(f"\nRecommendations:\n{final.recommendations}")

asyncio.run(main())
```

### Batch RTM Review

```python
import asyncio
from aiswre.components.rtm_review_agent_medtech_local import (
    Requirement,
    RTMEntry,
    run_batch_rtm_local_review
)

# Define requirements
requirements = [
    Requirement(
        req_id="REQ-001",
        text="The system shall measure temperature...",
        safety_class="C"
    ),
    Requirement(
        req_id="REQ-002",
        text="The system shall display results...",
        safety_class="B"
    )
]

# Define RTM entries
rtm_entries = [
    RTMEntry(
        req_id="REQ-001",
        test_case_summary="TC-001: Measure at 35°C, TC-002: Measure at 42°C...",
        test_case_count=4
    ),
    RTMEntry(
        req_id="REQ-002",
        test_case_summary="TC-005: Verify display update, TC-006: Verify units...",
        test_case_count=3
    )
]

# Run batch review with auto-detected Ollama ports
async def main():
    results = await run_batch_rtm_local_review(
        rtm_entries=rtm_entries,
        requirements=requirements,
        model="llama3.1",
        auto_detect_ports=True,  # Automatically detect active Ollama instances
        max_concurrent=3
    )

    for result in results:
        print(f"\n{'='*60}")
        print(f"Requirement: {result.req_id}")
        print(f"Average Score: {sum([
            result.functional_coverage_score,
            result.input_output_coverage_score,
            result.boundary_coverage_score,
            result.negative_test_coverage_score,
            result.risk_coverage_score,
            result.traceability_completeness_score,
            result.acceptance_criteria_coverage_score,
            result.test_sufficiency_score,
            result.gap_analysis_score
        ]) / 9:.2f}")
        print(f"\nVerification Gaps:\n{result.verification_gaps}")
        print(f"\nRecommendations:\n{result.recommendations}")

asyncio.run(main())
```

## Multi-Port Ollama for Parallel Execution

For faster batch processing, run multiple Ollama instances on different ports:

### Windows (PowerShell)

```powershell
# Terminal 1
$env:OLLAMA_HOST="0.0.0.0:11434"; ollama serve

# Terminal 2
$env:OLLAMA_HOST="0.0.0.0:11435"; ollama serve

# Terminal 3
$env:OLLAMA_HOST="0.0.0.0:11436"; ollama serve
```

### Linux/macOS

```bash
# Terminal 1
OLLAMA_HOST=0.0.0.0:11434 ollama serve

# Terminal 2
OLLAMA_HOST=0.0.0.0:11435 ollama serve

# Terminal 3
OLLAMA_HOST=0.0.0.0:11436 ollama serve
```

### Explicit Multi-Port Configuration

```python
results = await run_batch_rtm_local_review(
    rtm_entries=rtm_entries,
    requirements=requirements,
    model="llama3.1",
    base_urls=[
        "http://localhost:11434",
        "http://localhost:11435",
        "http://localhost:11436"
    ],
    auto_detect_ports=False,  # Use explicit URLs
    max_concurrent=3
)
```

## Output Format

### RTMReviewLink Fields

```python
class RTMReviewLink(BaseModel):
    req_id: str

    # Coverage scores (0-1)
    functional_coverage_score: float
    input_output_coverage_score: float
    boundary_coverage_score: float
    negative_test_coverage_score: float

    # Risk & Traceability scores (0-1)
    risk_coverage_score: float
    traceability_completeness_score: float
    acceptance_criteria_coverage_score: float

    # Sufficiency & Gaps scores (0-1)
    test_sufficiency_score: float
    gap_analysis_score: float  # Inverse of gap severity (1.0 = no gaps)

    # LLM-generated analysis
    review_summary: str
    verification_gaps: str
    recommendations: str

    # Metadata
    issues: List[str]
    rationale: str
    link_type: str
```

## Custom Weights

Adjust evaluator weights based on your priorities:

```python
# Emphasize gap analysis and risk coverage
custom_weights = {
    # Coverage (40% total)
    "functional_coverage_score": 0.10,
    "input_output_coverage_score": 0.10,
    "boundary_coverage_score": 0.10,
    "negative_test_coverage_score": 0.10,

    # Risk & Traceability (35% total)
    "risk_coverage_score": 0.15,  # Higher priority
    "traceability_completeness_score": 0.10,
    "acceptance_criteria_coverage_score": 0.10,

    # Sufficiency & Gaps (25% total)
    "test_sufficiency_score": 0.10,
    "gap_analysis_score": 0.15  # Higher priority
}

reviewer = get_rtm_local_reviewer_runnable(
    model="llama3.1",
    weights=custom_weights
)
```

## Integration with pandas/Excel

### Reading RTM from Excel

```python
import pandas as pd
from aiswre.components.rtm_review_agent_medtech_local import Requirement, RTMEntry

# Read RTM from Excel
df = pd.read_excel("rtm.xlsx")

# Create requirements
requirements = [
    Requirement(
        req_id=row['req_id'],
        text=row['requirement_text'],
        safety_class=row.get('safety_class'),
        risk_id=row.get('risk_id'),
        acceptance_criteria=row.get('acceptance_criteria')
    )
    for _, row in df.iterrows()
]

# Create RTM entries
rtm_entries = [
    RTMEntry(
        req_id=row['req_id'],
        test_case_summary=row['test_case_summary'],
        test_case_count=row.get('test_count'),
        test_ids=row.get('test_ids')
    )
    for _, row in df.iterrows()
]

# Run batch review
results = await run_batch_rtm_local_review(rtm_entries, requirements)

# Convert results to DataFrame
results_df = pd.DataFrame([
    {
        "req_id": r.req_id,
        "functional_coverage": r.functional_coverage_score,
        "boundary_coverage": r.boundary_coverage_score,
        "gap_analysis": r.gap_analysis_score,
        "verification_gaps": r.verification_gaps,
        "recommendations": r.recommendations
    }
    for r in results
])

results_df.to_excel("rtm_review_results.xlsx", index=False)
```

## Recommended Models

| Model | Size | Speed | Quality | Use Case |
|-------|------|-------|---------|----------|
| `llama3.1` | 8B | Fast | Good | General RTM review |
| `llama3.1:70b` | 70B | Slow | Excellent | High-stakes compliance |
| `mistral` | 7B | Fast | Good | Quick gap analysis |
| `qwen2.5` | 7B | Fast | Good | Alternative to llama3.1 |

## Performance Characteristics

### Single Ollama Instance
- **Throughput:** ~1-2 RTM entries per minute (9 evaluators + aggregator)
- **Bottleneck:** Sequential processing

### Multi-Port (3 Instances)
- **Throughput:** ~3-6 RTM entries per minute
- **Speedup:** ~3x with proper load balancing

### Batch Processing (10 RTM entries)
- **Single port:** ~5-10 minutes
- **3 ports:** ~2-3 minutes
- **5 ports:** ~1-2 minutes

## Troubleshooting

### Issue: "No active Ollama instances found"

**Solution:**
```bash
# Start Ollama server
ollama serve

# Or specify port explicitly
OLLAMA_HOST=0.0.0.0:11434 ollama serve
```

### Issue: "Model not found"

**Solution:**
```bash
# Pull the model first
ollama pull llama3.1

# List available models
ollama list
```

### Issue: Slow performance

**Solutions:**
1. Use a smaller model (`llama3.1` vs `llama3.1:70b`)
2. Run multiple Ollama instances on different ports
3. Reduce `max_concurrent` to avoid overwhelming the LLM
4. Use GPU acceleration if available

### Issue: Out of memory errors

**Solutions:**
1. Use a smaller model
2. Reduce batch size
3. Lower `max_concurrent` parameter
4. Close other applications using RAM/VRAM

## Comparison with Test Case Review Agents

| Feature | RTM Review Agent | Test Case Review Agent |
|---------|-----------------|------------------------|
| **Input** | Requirement + Test Case Summary | Individual Test Case |
| **Focus** | Verification Coverage | Test Case Quality |
| **Output** | Gap Analysis | Test Case Improvements |
| **Use Case** | RTM validation | Test case authoring |
| **Evaluators** | 9 (coverage-focused) | 11 (quality-focused) |

## Example RTM Input Format

```
| req_id  | requirement_text                        | test_case_summary                                    |
|---------|----------------------------------------|------------------------------------------------------|
| REQ-001 | System shall measure temperature...     | TC-001: Boundary test at 35°C, TC-002: At 42°C...   |
| REQ-002 | System shall display results...         | TC-005: Verify display, TC-006: Units, TC-007: ...  |
| REQ-003 | System shall store 1000 measurements... | TC-010: Store max, TC-011: Overflow handling...     |
```

## FDA/IEC 62304 Alignment

This agent evaluates verification coverage according to:
- **IEC 62304 Section 5.5** - Software Unit Verification
- **IEC 62304 Section 5.6** - Software Integration and Integration Testing
- **IEC 62304 Section 5.7** - Software System Testing
- **FDA Guidance** - General Principles of Software Validation

Key principles:
- **Complete verification** of all requirement aspects
- **Objective evidence** of verification
- **Risk-based testing** for safety-critical requirements
- **Traceability** between requirements and test cases

## Future Enhancements

Planned improvements:
- [ ] Support for test case detail expansion (not just summary)
- [ ] Integration with test management systems (Jira, TestRail)
- [ ] Automated gap closure tracking
- [ ] Historical trend analysis (coverage over time)
- [ ] OpenAI variant for cloud-based execution
- [ ] Customizable evaluator selection

## Contributing

To add new evaluators:
1. Create response model in `core.py`
2. Add evaluator factory in `evaluators.py`
3. Update `RTMReviewLink` with new score field
4. Register in `pipeline.py` graph
5. Update aggregator weights in `nodes.py`

See `src/components/tc_review_agent_medtech_local/` for reference architecture.

## License

GPL-3.0-or-later (same as parent aiswre project)

## Related Documentation

- Test Case Review Agent (Local): `../tc_review_agent_medtech_local/README.md`
- INCOSE Requirements Review: Main project README
- LangGraph Documentation: https://python.langchain.com/docs/langgraph
