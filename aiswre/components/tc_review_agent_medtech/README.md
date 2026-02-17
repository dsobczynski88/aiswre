# MedTech Test Case Review Agent

LangGraph-based AI agent for evaluating medical device software test cases against FDA/IEC 62304 best practices.

## Overview

This module provides automated review of test cases using 11 specialized evaluators organized into three categories:

### 1. General Integrity & Structure
- **Unambiguity**: Are instructions clear enough for someone with no prior knowledge?
- **Independence**: Is the test self-contained and does it clean up its own data?
- **Pre-conditions**: Are initial states (Power, Network, Database) explicitly defined?
- **Post-conditions**: Does the test return the system to a safe/neutral state?

### 2. Coverage & Technique
- **Technique Application**: Does the test utilize EP, BVA, or Decision Tables appropriately?
- **Negative Testing**: Are there tests for Invalid Inputs, Timeouts, and Error States?
- **Boundary Checks**: Are edges (Min, Min-1, Max, Max+1) explicitly verified?
- **Risk Verification**: Does the test verify effectiveness of risk controls?

### 3. Traceability & Compliance
- **Traceability**: Is there a correct link to a Requirement ID or Risk ID?
- **Safety Class Rigor**: For Class C units, is there MC/DC coverage?
- **Objective Evidence**: Are expected results specific values (e.g., "5V +/- 0.1V") vs subjective ("correct")?

## Architecture

```
Input (TestCase + Requirement)
    ↓
┌─────────────────────────────────────┐
│  11 Parallel Evaluators (LLM)      │
│  ┌───────────────────────────────┐ │
│  │ General Integrity (4)         │ │
│  │ Coverage & Technique (4)      │ │
│  │ Traceability & Compliance (3) │ │
│  └───────────────────────────────┘ │
└─────────────────────────────────────┘
    ↓
Aggregator (LLM-Powered Analysis)
  - Weighted score averaging
  - LLM synthesis of all findings
  - Comprehensive review summary
  - Actionable improvement recommendations
    ↓
MedtechTraceLink (11 scores + issues + summary + improvements)
```

## Quick Start

### Basic Usage

```python
import asyncio
from aiswre.components.tc_review_agent_medtech import (
    get_medtech_reviewer_runnable,
    run_batch_medtech_test_case_review,
    TestCase,
    Requirement
)

# Create test case
test = TestCase(
    test_id="TC-001",
    description="Verify alarm triggers at 38.5°C",
    preconditions="System on, sensor connected",
    expected_result="Alarm sounds within 2 sec",
    test_type="System",
    technique="BVA"
)

# Create requirement
req = Requirement(
    req_id="REQ-101",
    text="System shall alarm when temp > 38.5°C",
    safety_class="B",
    risk_id="RISK-015"
)

# Initialize reviewer
reviewer = get_medtech_reviewer_runnable(
    api_key="your-openai-key",
    model="gpt-4o-mini"
)

# Run review
results = await run_batch_medtech_test_case_review(
    reviewer=reviewer,
    test_cases=[test],
    requirements=[req]
)

# Access scores and analysis
result = results[0]
print(f"Unambiguity: {result.unambiguity_score}")
print(f"Traceability: {result.traceability_score}")
print(f"Issues: {result.issues}")
print(f"\nReview Summary:\n{result.review_summary}")
print(f"\nRecommended Improvements:\n{result.test_case_improvements}")
```

### From DataFrame

```python
import pandas as pd
from aiswre.components.tc_review_agent_medtech.pipeline import dataframe_to_medtech_inputs

# Load your test case data
df = pd.read_excel("test_cases.xlsx")

# Convert to domain objects
test_cases, requirements = dataframe_to_medtech_inputs(
    df,
    test_id_col="test_id",
    test_desc_col="description",
    req_id_col="req_id",
    req_text_col="requirement",
    preconditions="precond",
    expected_result="expected",
    safety_class="class"
)

# Run review
results = await run_batch_medtech_test_case_review(
    reviewer=reviewer,
    test_cases=test_cases,
    requirements=requirements
)
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

reviewer = get_medtech_reviewer_runnable(
    api_key="key",
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
    "link_type": "MedtechAggregated",

    # LLM-Generated Aggregated Analysis
    "review_summary": "Comprehensive paragraph synthesizing all 11 evaluator findings, highlighting critical issues and overall test case quality assessment...",
    "test_case_improvements": "Specific, actionable recommendations such as: 1) Add timeout handling test for invalid inputs, 2) Replace subjective expected result 'system works' with specific values..."
}
```

## Extending the System

### Adding New Evaluators

1. **Create Response Model** in `core.py`:
```python
class SecurityResponse(BaseModel):
    security_score: float
    vulnerabilities: List[str]
    rationale: str
```

2. **Add Factory Function** in `evaluators.py`:
```python
def make_security_evaluator(llm: ChatOpenAI) -> MedtechEvaluatorNode:
    prompt = "You are a security expert evaluating test security..."
    return MedtechEvaluatorNode(
        llm=llm,
        response_model=SecurityResponse,
        system_prompt=prompt,
        link_type="Security"
    )
```

3. **Update MedtechTraceLink** in `core.py`:
```python
class MedtechTraceLink(BaseModel):
    # ... existing scores ...
    security_score: float = 0.0
```

4. **Register in Pipeline** `pipeline.py`:
```python
# In build_graph()
security = make_security_evaluator(client)
sg.add_node("security", security)
sg.add_edge("__start__", "security")
sg.add_edge("security", "aggregate")
```

5. **Update Node Mapping** in `nodes.py`:
```python
# In _to_medtech_tracelink()
elif isinstance(resp, SecurityResponse):
    tl.security_score = resp.security_score
    tl.issues.extend(resp.vulnerabilities)
```

6. **Update Weights** in `nodes.py`:
```python
# In MedtechAggregatorNode.__init__()
self.weights = weights or {
    # ... existing weights ...
    "security_score": 0.05,
}
```

## Files

- `core.py` - Pydantic models (TestCase, Requirement, MedtechTraceLink, Response schemas)
- `evaluators.py` - Factory functions for 11 evaluator nodes
- `nodes.py` - MedtechEvaluatorNode and MedtechAggregatorNode implementations
- `pipeline.py` - LangGraph orchestration and batch processing
- `__init__.py` - Public API exports

## Best Practices

### Input Data Quality
- Provide complete test case fields (preconditions, steps, expected_result, postconditions)
- Link requirements with safety_class and risk_id when available
- Use consistent terminology across test cases

### Score Interpretation
- **0.0-0.4**: Major issues, test case needs significant revision
- **0.4-0.7**: Moderate issues, improvements recommended
- **0.7-0.9**: Minor issues, generally acceptable
- **0.9-1.0**: Excellent, meets best practices

### Handling Issues
- Review `issues` list for specific actionable feedback
- Read `rationale` for detailed evaluation reasoning
- Prioritize fixes based on safety_class and risk_id

## Example Script

See `scripts/readme_medtech_reviewer_example.py` for a complete working example.

## Requirements

- Python 3.8+
- `langchain-openai`
- `langgraph`
- `pydantic`
- OpenAI API key

## License

Same as parent project.
