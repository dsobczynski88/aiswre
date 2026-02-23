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

**INCOSE-Based Requirement Review**

Enter from the command line: `python scripts/readme_req_reviewer_example.py` 

Source code file:
```python
import asyncio
import pandas as pd
from dotenv import dotenv_values
from src import utils
from aiswre.components import prompteval as pe
from aiswre.components.promptrunner import RateLimitOpenAIClient
from aiswre.components.promptprocessor import PromptProcessor  
from aiswre.utils import load_prompt

# ===============================================================
# Configuration
# ===============================================================
DOT_ENV = dotenv_values(".env")
CONFIG = utils.load_config("config.yaml")
MODEL = CONFIG["MODEL"]
MODEL_KWARGS = CONFIG["MODEL_KWARGS"]
PROMPT_TEMPLATE_PATH = CONFIG["FILE_LOCATIONS"]["PROMPT_TEMPLATE_PATH"]
PROMPT_NAME = CONFIG["PROMPT_TEMPLATE"]
OPENAI_API_KEY = DOT_ENV["OPENAI_API_KEY"]
MAX_REQUESTS_PER_MIN = 490
MAX_TOKENS_PER_MIN = 200000
OUTPUT_DIRECTORY = utils.make_output_directory(CONFIG["FILE_LOCATIONS"], "OUTPUT_FOLDER")
SYSTEM_PROMPT = load_prompt(PROMPT_TEMPLATE_PATH, PROMPT_NAME, "system")
USER_PROMPT_TEMPLATE = load_prompt(PROMPT_TEMPLATE_PATH, PROMPT_NAME, "user")
DATASET_FILE_PATH = CONFIG["FILE_LOCATIONS"]["DATASET_FILE_PATH"]
SELECTED_EVAL_FUNCS = CONFIG["SELECTED_EVAL_FUNCS"]
SELECTED_EVAL_WEIGHTS = CONFIG["SELECTED_EVAL_WEIGHTS"]
EVAL_CONFIG = pe.make_eval_config(pe, include_funcs=SELECTED_EVAL_FUNCS)

# ===============================================================
# Input requirements
# ===============================================================
df_input = pd.read_excel(DATASET_FILE_PATH)

# ===============================================================
# Instantiate LLM client
# ===============================================================
rl_client = RateLimitOpenAIClient(
    api_key=OPENAI_API_KEY,
    max_requests_per_minute=MAX_REQUESTS_PER_MIN,
    max_tokens_per_minute=MAX_TOKENS_PER_MIN
    )

# ===============================================================
# Async runner
# ===============================================================
async def run_req_review_with_processor(client, input_df, model, model_kwargs):
    processor = PromptProcessor(client=client, input_df=input_df, model=model, model_kwargs=model_kwargs)
    items = processor.df_to_prompt_items(df_input, ["requirement_id", "requirements"])
    ids = [item["requirement_id"] for item in items]
    results = await processor.run_prompt_batch(
        system_message=SYSTEM_PROMPT,
        user_message_template=USER_PROMPT_TEMPLATE,
        prompt_name=PROMPT_NAME,
        items=items,
        ids=ids,
    )
    return pd.DataFrame(results)

# ===============================================================
# Main execution
# ===============================================================
review_df = asyncio.run(run_req_review_with_processor(rl_client, df_input, MODEL, MODEL_KWARGS))
review_df = pe.call_evals(review_df, col="requirements_review.proposed_rewrite", eval_config=EVAL_CONFIG)
review_df = pe.get_failed_evals(review_df)
pe.add_weighted_column(review_df, SELECTED_EVAL_FUNCS, SELECTED_EVAL_WEIGHTS, "weighted_value")    
review_df.to_excel(f"{OUTPUT_DIRECTORY}/reviewed_requirements.xlsx")