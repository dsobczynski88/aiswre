# ===============================================================
# Revised ReqReview with dynamic prompt loading
# ===============================================================
import asyncio
import pandas as pd
from dotenv import dotenv_values
from src import utils
from src.components import prompteval as pe
from src.components.clients import RateLimitOpenAIClient
from src.components.processors import OpenAIPromptProcessor, df_to_prompt_items, process_json_responses  # adjust import
from src.utils import load_prompt  # adjust import

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

rl_client = RateLimitOpenAIClient(
    api_key=OPENAI_API_KEY,
    max_requests_per_minute=MAX_REQUESTS_PER_MIN,
    max_tokens_per_minute=MAX_TOKENS_PER_MIN
    )

# ===============================================================
# Async runner
# ===============================================================
async def run_req_review_with_processor(client, input_df, model, model_kwargs):
    processor = OpenAIPromptProcessor(client=client, input_df=input_df, model=model, model_kwargs=model_kwargs)
    items = df_to_prompt_items(df_input, ["requirement_id", "requirements"])
    ids = [item["requirement_id"] for item in items]
    results = await processor.run_prompt_batch(
        system_message=SYSTEM_PROMPT,
        user_message_template=USER_PROMPT_TEMPLATE,
        prompt_name=PROMPT_NAME,
        items=items,
        ids=ids,
    )
    results = process_json_responses(results, ids, PROMPT_NAME)
    return pd.DataFrame(results)

# ===============================================================
# Main execution
# ===============================================================
review_df = asyncio.run(run_req_review_with_processor(rl_client, df_input, MODEL, MODEL_KWARGS))
review_df = pe.call_evals(review_df, col="requirements_review.proposed_rewrite", eval_config=EVAL_CONFIG)
review_df = pe.get_failed_evals(review_df)
pe.add_weighted_column(review_df, SELECTED_EVAL_FUNCS, SELECTED_EVAL_WEIGHTS, "weighted_value")    
review_df.to_excel(f"{OUTPUT_DIRECTORY}/reviewed_requirements.xlsx")