This project (titled `aiswre`) seeks to integrate the best practices described in the INCOSE Guide to Writing Requirements to enhance software requirement quality using NLP and AI.

### Overview of the `aiswre` project

This project, `aiswre` intends to apply AI, NLP, and data science methodologies to improve the quality of software quality processes. The initial features of the project focus on using prompt engineering techniques to refine software requirements based on the rules described in section 4 of the [INCOSE Guide](https://www.incose.org/docs/default-source/working-groups/requirements-wg/gtwr/incose_rwg_gtwr_v4_040423_final_drafts.pdf?sfvrsn=5c877fc7_2). This project was inspired by the desire to enhance the field of software quality with AI and system engineering best practices. Application of LLMs bear the opportunity to advance the field of requirements engineering as initial studies have shown promising results<sup>1,2</sup>.

### Design description

The project will take a requirement as input, assess it against a variety of criteria, and based on the identified gaps, refine the requirement to align with rules as described in INCOSE Guide to Writing Requirements Section 4. At present, the application only leverages the input requirement and INCOSE Guide (no other information about the project) to perform the revision.

### Getting started

- Set up your OpenAI API key [OPEN AI Developer quickstart](https://platform.openai.com/)
- Add requirements dataset to the directory
- Open a powershell terminal and enter the following to clone the repository:
	- `git clone https://github.com/dsobczynski88/aiswre.git <your_desired_folder_name>`
- Navigate to the folder containing the cloned repository:
	- `cd <your_desired_folder_name>`
- Create a blank `.env` file in this location and enter:
	- `OPENAI_API_KEY = <your_api_key>`
- Create a virtual env:
	- `python -m venv venv` 
- Activate the environment (Windows Powershell):
	- `.\\venv\Scripts\activate.bat`
- Enter the following commands to install the code and dependencies:
	- `python -m pip install -r requirements.txt`
	- `python -m pip install -e .`

### Future Work

Prompt engineering best practices have been applied to improve the results. However at present, this work is structured in a pre-defined way that confines the workflow. The program itself may benefit from usage of more advanced approaches in AI such as langgraph flows and agent frameworks. Furthermore, the work at best is designed for a handful of INCOSE rules and therefore is still in progress. To improve the robustness and utility of this work, there are opportunities to leverage more efficient design patterns, and this too is a subject of ongoing project activities.

### Example Usage

```python
import os
import asyncio
import json
import flatdict
from typing import Any, Dict, List, Optional
from dotenv import dotenv_values

import pandas as pd
import flatdict
from pydantic import BaseModel, Field, SecretStr

from src import utils
from src.components import prompteval as pe
from src.components.promptrunner import RateLimitOpenAIClient

async def process_json_responses(
    responses, ids, prompt_type, json_key: str = "requirements_review"
    ) -> List[Dict[str, Any]]:
    """Process OpenAI responses and flatten extracted JSON structures."""
    processed = []

    for i, response in enumerate(responses):
        output = dict(response)
        message = getattr(response.choices[0], "message", None)
        if not message:
            continue

        # Parse structured JSON content if available
        if getattr(message, "content", None):
            try:
                response_json = json.loads(message.content)
                if json_key in response_json:
                    nested_dicts = response_json[json_key]
                    flat_dicts = [flatdict.FlatDict(d, delimiter=".") for d in nested_dicts]
                    for d in flat_dicts:
                        output.update(d)
            except (json.JSONDecodeError, TypeError):
                output["json_parse_error"] = message.content

        # Include usage info
        if getattr(response, "usage", None):
            usage = dict(response.usage)
            usage.update(dict(getattr(response.usage, "completion_tokens_details", {})))
            usage.update(dict(getattr(response.usage, "prompt_tokens_details", {})))
            output.update(usage)

        # Include parsed content if provided
        if getattr(message, "parsed", None):
            output.update(dict(message.parsed))
        output.update(
            {
                "requirement_id": ids[i],
                "prompt_type": prompt_type,
            }
        )
        processed.append(output)
    return processed

async def run_requirement_review(
    openai_client,
    system_message: str,
    user_message: str,
    prompt_name: str,
    requirements: List[str],
    ids: Optional[List[int]] = None,
    model: str = "gpt-4o-mini",
    json_key: str = "requirements_review",
    ) -> List[Dict[str, Any]]:
    """Execute concurrent review prompts and process JSON responses."""
    if ids is None:
        ids = list(range(len(requirements)))
    # Build concurrent tasks
    tasks = [
        openai_client.chat_completion_parse(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user",
                    "content": user_message
                    .replace("{requirements}", f"{req_id}: {req}")
                    .replace("{enable_split}", "True"),
                },
            ],
            response_format={"type": "json_object"},
        )
        for req, req_id in zip(requirements, ids)
    ]
    # Run all requests concurrently
    responses = await asyncio.gather(*tasks)

    # Process structured JSON responses
    return await process_json_responses(responses, ids, prompt_name, json_key)

# Instantiate the openai client and define model
DOT_ENV = dotenv_values("../.env")
OPENAI_API_KEY = DOT_ENV['OPENAI_API_KEY']
rl_openai_client = RateLimitOpenAIClient(api_key=OPENAI_API_KEY)
MODEL = 'gpt-4o-mini'

eval_funcs = [
    'eval_avoids_vague_terms',
    'eval_definite_articles_usage',
    'eval_has_appropriate_subject_verb',
    'eval_has_common_units_of_measure',
    'eval_has_escape_clauses',
    'eval_has_no_open_ended_clauses',
    'eval_is_active_voice',
]
eval_weights = [
    0.35,
    0.05,
    0.15,
    0.05,
    0.10,
    0.10,
    0.20
]
# Make eval config
eval_config = pe.make_eval_config(pe, include_funcs=eval_funcs)

# Define prompt messages
SYSTEM_PROMPT = """
You are a Senior Requirements Quality Analyst and technical editor. 
You specialize in detecting and fixing requirement defects using authoritative quality rules. 
Be rigorous, consistent, and concise. Maintain the author's technical intent while removing ambiguity. 
Do not add new functionality. Ask targeted clarification questions when needed.

Response Format (produce exactly this JSON structure):
{
  "requirements_review": [
    {
      "requirement_id": "<ID>",
      "original": "<original requirement>",
      "checks": {
        "R2": {"status": "pass|fail", "active_voice": ["<issues>"], "explanation": "<brief>"},
        "R3": {"status": "pass|fail", "appropriate_subj_verb": ["<issues>"], "explanation": "<brief>"},
        "R5": {"status": "pass|fail", "definite_articles": ["<issues>"], "explanation": "<brief>"},
        "R6": {"status": "pass|fail", "units": ["<issues>"], "explanation": "<brief>"},
        "R7": {"status": "pass|fail", "vague terms": ["<issues>"], "explanation": "<brief>"},
        "R8": {"status": "pass|fail", "escape_clauses": ["<issues>"], "explanation": "<brief>"},
        "R9": {"status": "pass|fail", "open_ended_clauses": ["<issues>"], "explanation": "<brief>"}
      },
      "proposed_rewrite": "<single improved requirement that resolves all detected issues>",
      "split_recommendation": {
        "needed": true|false,
        "because": "<why>",
        "split_into": ["<Req A>", "<Req B>"]
      },
    }
  ]
}

Evaluation method:
1) Parse inputs and normalize IDs. 
2) For each requirement, test 2, R3, R5, R6, R7, R8, R9. 
3) Explain each failure succinctly. 
4) Rewrite to a single, verifiable sentence unless a split is recommended. 
5) Apply glossary rules for abbreviations; on first use of allowed abbreviations, prefer the expanded form with abbreviation in parentheses. 
6) If required numbers are missing and no defaults are provided, use TBD placeholders and ask explicit questions to resolve them. 
7) Summarize compliance.

Important: If {requirements} is empty, respond with a single clarifying question requesting requirements to review and stop.
"""

USER_PROMPT = """
Task: Review and improve the following requirement statements using the provided variables.

Variables:
- Requirements (list or newline-separated; may include IDs):
  {requirements}
- Enable split recommendations (true|false; default true): {enable_split}

Produce output strictly in the Response Format JSON. Do not use Markdown.

Now perform the review on the provided inputs and return only the Response Format JSON.
"""

PROMPT_NAME = 'basic-incose'

# Define the requirements to be revised
requirements = [
    "If projected the data must be readable.  On a 10x10 projection screen  90% of viewers must be able to read Event / Activity data from a viewing distance of 30",
    "The product shall ensure that it can only be accessed by authorized users.  The product will be able to distinguish between authorized and unauthorized users in all access attempts",
    "All business rules specified in the Disputes System shall be in compliance to the guidelines of Regulation E and Regulation Z",
]
df = pd.DataFrame({'requirements': requirements})

# Run revisions and cast to dataframe
revisions = asyncio.run(run_requirement_review(
    openai_client=rl_openai_client,
    system_message=SYSTEM_PROMPT,
    user_message=USER_PROMPT,
    prompt_name=PROMPT_NAME,
    requirements=requirements,
    ids=None,
    model="gpt-4o-mini",
    json_key="requirements_review"
    )
)
final_df = pd.DataFrame(revisions)

# Get post-revision Accuracy Score
final_df = pe.call_evals(final_df, col='proposed_rewrite', eval_config=eval_config)
final_df = pe.get_failed_evals(final_df)
pe.add_weighted_column(final_df, eval_funcs, eval_weights, "weighted_value")

# View original and rewritten requirement statements
print(final_df[['original', 'proposed_rewrite']])
```