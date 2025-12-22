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