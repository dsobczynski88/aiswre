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

Enter from the command line: `python scripts/readme_req_reviewer_example.py` 

```python
# ===============================================================
# Revised ReqReview with dynamic prompt loading
# ===============================================================
import os
import asyncio
import pandas as pd
from dotenv import dotenv_values
from src import utils
from src.components import prompteval as pe
from src.components.promptrunner import RateLimitOpenAIClient
from src.components.promptprocessor import PromptProcessor  # adjust import
from src.utils import load_prompt  # adjust import

# ===============================================================
# Configuration
# ===============================================================
DOT_ENV = dotenv_values(".env")
OPENAI_API_KEY = DOT_ENV["OPENAI_API_KEY"]
MODEL = "gpt-4o-mini"
MODEL_KWARGS = {"response_format": {"type":"json_object"}}
PROMPT_BASE_PATH = "src/prompts"
MAX_REQUESTS_PER_MIN=490
MAX_TOKENS_PER_MIN=200_000
  
# ===============================================================
# Load prompts dynamically
# ===============================================================
PROMPT_NAME = "test_D" # <-- user chooses which prompt set to use
SYSTEM_PROMPT = load_prompt(PROMPT_BASE_PATH, PROMPT_NAME, "system")
USER_PROMPT_TEMPLATE = load_prompt(PROMPT_BASE_PATH, PROMPT_NAME, "user")
JSON_KEY = "requirements_review"

# ===============================================================
# Evaluation functions
# ===============================================================
eval_funcs = [
    "eval_avoids_vague_terms",
    "eval_definite_articles_usage",
    "eval_has_appropriate_subject_verb",
    "eval_has_common_units_of_measure",
    "eval_has_escape_clauses",
    "eval_has_no_open_ended_clauses",
    "eval_is_active_voice",
]
eval_weights = [0.35, 0.05, 0.15, 0.05, 0.10, 0.10, 0.20]
eval_config = pe.make_eval_config(pe, include_funcs=eval_funcs)

# ===============================================================
# Input requirements
# ===============================================================
requirements = [
    "If projected the data must be readable. On a 10x10 projection screen 90% of viewers must be able to read Event / Activity data from a viewing distance of 30",
]
df_input = pd.DataFrame({
    "requirement_id": list(range(len(requirements))),
    "requirements": requirements,
    })

rl_client = RateLimitOpenAIClient(
    api_key=OPENAI_API_KEY,
    max_requests_per_minute=MAX_REQUESTS_PER_MIN,
    max_tokens_per_minute=MAX_TOKENS_PER_MIN
    )

# ===============================================================
# Async runner
# ===============================================================
async def run_req_review_with_processor(client, input_df ,model, model_kwargs):
    processor = PromptProcessor(client=client, input_df=input_df, model=model, model_kwargs=model_kwargs)
    items = processor.df_to_prompt_items(df_input, ["requirement_id", "requirements"])
    ids = [item["requirement_id"] for item in items]
    results = await processor.run_prompt_batch(
        system_message=SYSTEM_PROMPT,
        user_message_template=USER_PROMPT_TEMPLATE,
        prompt_name=PROMPT_NAME,
        items=items,
        ids=ids,
        json_key=JSON_KEY,
    )
    return pd.DataFrame(results)

# ===============================================================
# Main execution
# ===============================================================
review_df = asyncio.run(run_req_review_with_processor(rl_client, df_input, MODEL, MODEL_KWARGS))
# Evaluate rewritten requirements
review_df.to_excel('src/data/test-output.xlsx')
review_df = pe.call_evals(review_df, col="proposed_rewrite", eval_config=eval_config)
review_df = pe.get_failed_evals(review_df)
review_df.to_excel('src/data/test-output.xlsx')
pe.add_weighted_column(review_df, eval_funcs, eval_weights, "weighted_value")    
print(review_df[["original", "proposed_rewrite"]])