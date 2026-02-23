# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**aiswre** (AI System for Writing Requirements Engineering) is a Python tool using AI/NLP to improve software requirement and test case quality. It has five pipelines:

1. **INCOSE Requirements Review** - Evaluates requirements against INCOSE Guide to Writing Requirements using 40+ NLP evaluation functions. Has both OpenAI and local Ollama variants.
2. **MedTech Test Case Review** - 11 parallel LangGraph evaluators aligned to FDA/IEC 62304. Has OpenAI (`tc_review_agent_medtech`) and local Ollama (`tc_review_agent_medtech_local`) variants.
3. **Ollama Generic Test Case Review** (`tc_review_agent_ollama`) - Lightweight 2-evaluator local LLM reviewer for general QA.
4. **RTM Review - Local** (`rtm_review_agent_medtech_local`) - 9 evaluators assessing how well test suites verify requirements (verification coverage, not individual test quality). Uses Ollama.
5. **RTM Review - OpenAI** (`rtm_review_agent_medtech`) - **In active development.** 4 coverage evaluators with a more sophisticated decomposer→summarizer→assembler→evaluators→aggregator graph pattern. Uses OpenAI.

## Common Commands

```bash
# Setup
python -m venv venv
venv\Scripts\activate          # Windows
pip install -e .               # Core deps (recommended)
pip install -e ".[dev]"        # + pytest, black, flake8, mypy
pip install -e ".[all]"        # Everything

# Run pipelines
python scripts/readme_req_reviewer_example.py              # INCOSE review (OpenAI)
python scripts/readme_req_reviewer_local_example.py         # INCOSE review (Ollama)
python scripts/readme_req_reviewer_local_example.py --multi-port  # Ollama multi-port
python scripts/readme_medtech_reviewer_example.py           # MedTech TC review (OpenAI)
python scripts/readme_medtech_local_reviewer_example.py     # MedTech TC review (Ollama)
python scripts/readme_rtm_reviewer_example.py               # RTM review (Ollama)
python scripts/test-rtm-medtech-simple.py                   # RTM review (OpenAI, dev/test)

# Formatting and linting
black aiswre/ scripts/         # Line length: 100
flake8 aiswre/ scripts/
mypy aiswre/

# Tests (pytest configured but no test files exist yet)
pytest                         # testpaths = ["tests"]
pytest -m "not slow"           # Markers: slow, integration
```

## Architecture

### Package Structure

The package root is `aiswre/` (not `src/` - the codebase was migrated). Imports use `from aiswre.components...`.

```
aiswre/
  __init__.py
  utils.py                    # Config loading, file I/O, prompt loading
  prj_exception.py
  prj_logger.py
  components/
    __init__.py
    clients.py                # RateLimitOpenAIClient (dual RPM+TPM rate limiting)
    processors.py             # OpenAIPromptProcessor, OllamaPromptProcessor, GraphProcessor
    prompteval.py             # 40+ INCOSE eval functions (NLP via NLTK)
    incose.py                 # INCOSE reference data
    preprocess.py             # Data preprocessing utilities
    tc_review_agent_medtech/         # MedTech TC review (OpenAI, 11 evaluators)
    tc_review_agent_medtech_local/   # MedTech TC review (Ollama, 11 evaluators)
    tc_review_agent_ollama/          # Generic TC review (Ollama, 2 evaluators)
    rtm_review_agent_medtech/        # RTM review (OpenAI, 4 evaluators) — IN DEVELOPMENT
    rtm_review_agent_medtech_local/  # RTM review (Ollama, 9 evaluators)
  data/                       # Demo datasets, INCOSE PDF
  prompts/                    # System/user prompt templates (A-H variants + prewarm)
```

### LangGraph Agent Pattern

All **completed** review agents (`tc_review_agent_*`, `rtm_review_agent_medtech_local`) follow the same structure:

- `core.py` - Pydantic response models and data structures (e.g., `MedtechTraceLink`, `RTMReviewLink`)
- `evaluators.py` - Factory functions that create LangGraph evaluation nodes (e.g., `make_unambiguity_evaluator()`)
- `nodes.py` - Node mapping, aggregator logic, and graph state definitions
- `pipeline.py` - `get_*_reviewer_runnable()` factory that wires the LangGraph state machine

The standard graph pattern is: **parallel evaluator nodes → aggregator node → output**. Each evaluator uses Pydantic `.with_structured_output()` for reliable JSON from LLMs.

### RTM OpenAI Agent (In Development)

`rtm_review_agent_medtech` uses a more sophisticated multi-stage graph pattern:

```
START → [decomposer, summarizer] (parallel) → assembler → [4 coverage evaluators] (parallel) → aggregator → END
```

- **Decomposer**: Breaks a requirement into atomic testable specifications with edge-case analysis
- **Summarizer**: Distills test cases to objective/protocol/acceptance criteria
- **Assembler**: Pure data node (no LLM) that collects decomposer + summarizer outputs
- **Evaluators**: Functional, I/O, boundary, and negative test coverage
- **Aggregator**: Synthesizes evaluator assessments into actionable recommendations

This agent does NOT have a separate `evaluators.py`; all node factory functions live in `nodes.py`. Entry point is `RTMReviewerRunnable` class in `pipeline.py`.

### Two Processing Paths

1. **INCOSE pipeline** (non-graph): `RateLimitOpenAIClient` or `ChatOllama` → `OpenAIPromptProcessor`/`OllamaPromptProcessor` → `prompteval.py` NLP evaluation functions → Excel output
2. **LangGraph agents**: `get_*_reviewer_runnable()` factory → `GraphProcessor.run_graph_batch()` → Excel output. The OpenAI RTM agent uses `RTMReviewerRunnable` class directly instead of a `get_*` factory function.

### Key Processors in `processors.py`

- `OpenAIPromptProcessor` / `OllamaPromptProcessor` - Async batch processing for INCOSE pipeline
- `GraphProcessor` - Runs any LangGraph agent asynchronously via `run_graph_batch()`
- `parse_llm_json_like()` - 4-step fallback JSON parser (json.loads → ast.literal_eval → pattern repair)
- `df_to_prompt_items()` / `load_input_data()` - Data loading utilities

### Multi-Port Ollama Parallelization

Local agents support running multiple Ollama instances on ports 11434-11439 for parallel execution. On Windows:
```cmd
set OLLAMA_HOST=0.0.0.0:11434 && ollama serve   # Terminal 1
set OLLAMA_HOST=0.0.0.0:11435 && ollama serve   # Terminal 2
```
`detect_ollama_ports()` in RTM and MedTech local agents auto-discovers active ports.

## Configuration

- **Primary**: Code-based configuration via factory function parameters (see `scripts/` for examples)
- **Legacy**: `config.yaml` at root for older INCOSE pipeline examples (model, prompts, eval funcs, weights)
- **Environment**: `.env` file with `OPENAI_API_KEY` (required for OpenAI variants)

## Extending Agents

### Adding a new evaluator to any LangGraph agent

1. Add Pydantic response model in `core.py`
2. Add `make_<name>_evaluator()` factory in `evaluators.py` (or `nodes.py` for `rtm_review_agent_medtech`)
3. Add `<name>_score` field to the output model (e.g., `MedtechTraceLink`)
4. Register the node in `nodes.py` and `pipeline.py`
5. Update aggregator weights

Each agent subdirectory has a `README.md` with detailed instructions (except `rtm_review_agent_medtech` which is still in development).

### Adding INCOSE evaluation functions

1. Create `eval_<name>()` in `prompteval.py`
2. Map to rule group in `config.yaml` under `INCOSE_RULE_GROUPS`
3. Add to `SELECTED_EVAL_FUNCS` to activate

## Implementation Notes

- **Windows async**: Uses `ProactorEventLoop` + `nest_asyncio.apply()` for Jupyter compatibility
- **Rate limiting** (`clients.py`): Proactive throttling (waits before hitting limits) + reactive exponential backoff on 429s
- **Python**: Requires 3.10+. Black configured at 100 char line length.
- **Package install**: Always use `pip install -e .` (pyproject.toml), not `requirements.txt`
