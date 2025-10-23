from dotenv import dotenv_values
from src import utils
import json
import flatdict
from src import utils

# Load config settings
DOT_ENV = dotenv_values("../.env")
config = utils.load_config("../config.yaml")

# Create a unique run-id folder to store outputs
config["FILE_LOCATIONS"]["MAIN_DATA_FOLDER"] = "../src/data"
output_directory = utils.make_output_directory(config["FILE_LOCATIONS"])

all_rule_groups = [
    'Conditions', 
    'Singularity', 
    'Uniformity_Of_Language', 
    'Concision', 
    'Modularity', 
    'Non_Ambiguity', 
    'Tolerance', 
    'Quantifiers', 
    'Quantification', 
    'Completeness', 
    'Accuracy', 
    'Abstraction',
    'Realism'    
]                

from src.components import prompteval as pe
# Functions currently requiring remediation
include_funcs = [
    'eval_avoids_vague_terms',
    'eval_definite_articles_usage',
    'eval_has_appropriate_subject_verb',
    'eval_has_common_units_of_measure',
    'eval_has_escape_clauses',
    'eval_has_no_open_ended_clauses',
    'eval_is_active_voice',
]

score_weights = [
    0.35,
    0.05,
    0.15,
    0.05,
    0.10,
    0.10,
    0.20
]
# Make evaluation function config
eval_config = pe.make_eval_config(pe, include_funcs=include_funcs)
import ast
import os
import asyncio
from pydantic import BaseModel, Field, SecretStr
from src.components.promptrunner import RateLimitOpenAIClient
import prompts

# Instantiate the openai client
openai_api_key = SecretStr(str(DOT_ENV['OPENAI_API_KEY']))
openai_client = RateLimitOpenAIClient(api_key=openai_api_key.get_secret_value())

prompt_prewarm = (prompts.system_test_prewarm, prompts.user_test_prewarm, 'prompt_prewarm')

prompts_messages = [
    prompt_prewarm,
]

# Define function to perform AI requirement reviews
async def run_reviewer_prompts(prompt_messages, requirements, ids, model: str = 'gpt-4o-mini'):
       
    req_tups = zip(requirements, ids)
    # Example of making multiple concurrent requests
    tasks = []
    for i, req_tup in enumerate(req_tups):
        tasks.append(
            openai_client.chat_completion_parse(
                model=model,
                messages=[
                    {"role": "system", "content": prompt_messages[0]},
                    {"role": "user", "content": prompt_messages[1].replace('{requirements}',f'{req_tup[1]}: {req_tup[0]}').replace('{enable_split}', 'True')}
                ],
                response_format={"type": "json_object"}
            )
        )
  
    # Wait for all requests to complete
    responses = await asyncio.gather(*tasks)
    
    # Process responses
    processed_responses = []
    for i, response in enumerate(responses):
        
        output = dict(response)
        message = response.choices[0].message
        print(message)
        if getattr(message, "content"):
            response_json = json.loads(message.content)
            if 'requirements_review' in  list(response_json.keys()):
                nested_dict = response_json['requirements_review']          
                print(f"Nested dict: {type(nested_dict)}, {len(nested_dict)}")
                # Only set to work if a single requirement is passed to prompt
                flat_dict = [flatdict.FlatDict(n, delimiter='.') for n in nested_dict]          
                print(f"Flat dict: {type(flat_dict)}, {len(flat_dict)}")
                print(flat_dict)
                for d in flat_dict:
                    output.update(
                        d
                    )
        
        if getattr(response, "usage"):
            
            output.update(
                dict(response.usage)
            )
            
            output.update(
                dict(response.usage.completion_tokens_details)
            )
            output.update(
                dict(response.usage.prompt_tokens_details)
            )

        if getattr(message, "parsed"):
            output.update(
                dict(message.parsed)
            )
        
        output.update(
            {'requirement_id': ids[i]}
        )

        output.update(
            {'prompt_type': prompt_messages[-1]}
        )
        
        print(f"\nResponse {i+1}:")
        print(output)
        processed_responses.append(output)
        
    #return processed_responses
    return processed_responses

# Call the evaluations on the dataframe 
import pandas as pd
iternum=1
id_col = 'requirement_id'
requirement_col_post_revision = 'proposed_rewrite'
rule_groups_to_evaluate = ['Accuracy']
MODEL = 'gpt-4o-mini'

for prompt in prompts_messages:
    for i in range(iternum):
        if i == 0:
            # Load requirements
            df = pd.read_excel('../src/data/software_requirements.xlsx')
            df = df.head(400)
            requirement_col = 'requirement'
            # Call the evaluations on the dataframe 
            df = pe.call_evals(df, col=requirement_col, eval_config=eval_config)
            # Get list of failed eval functions
            df = pe.get_failed_evals(df)
            # Map the failed eval functions to rule groups (as defined in the config.yaml file)
            df = pe.map_failed_eval_col_to_rule_group(df, eval_to_rule_map=config["SECTION_4_RULE_GROUPS"], failed_eval_col='failed_evals')   
            # Drop requirements which pass acceptance criteria
            # At present, the criteria is len(failed_evals_rule_ids) == 0
            df = df.loc[df['failed_evals_rule_ids'].str.len() > 0]
            df['iternum'] = 0
            df = pe.get_failed_eval_count_by_rule_group(
                df=df,
                rule_groups=rule_groups_to_evaluate,
                failed_eval_col='failed_evals_rule_ids'
                )
            df["has_failed_rule_groups"] = utils.check_failed_evals(df, rg=f"({'|'.join(rule_groups_to_evaluate)})", match_mode="regex")
            df = df.loc[df["has_failed_rule_groups"]==True]
            # Get score
            initial_weighted_values = pe.add_weighted_column(df, include_funcs, score_weights, "weighted_value_initial")
            initial_requirement_ids = list(df[id_col].values)
            initial_weighted_values_dict = dict(zip(initial_requirement_ids, initial_weighted_values))
            df = utils.drop_columns_by_regex(df, patterns=[r"^failed_evals_[A-Z]"], how="any")
            if df is not None:
                print(df.columns)
                df.to_excel(f"{output_directory}/eval_df_iter_{prompt[-1]}_0.xlsx")
        else:
            df = revisions_df.copy()
            requirement_col = requirement_col_post_revision

        if df is not None:
            requirements = list(df[requirement_col].values)
            ids = list(df[id_col].values)     
            revisions = asyncio.run(run_reviewer_prompts(prompt, requirements, ids, model=MODEL))
            revisions_df = pd.DataFrame(revisions)
            print(revisions_df)
            revisions_df.to_excel(f"{output_directory}/revisions_df_{i+1}_{prompt[-1]}.xlsx")
            # Call the evaluations on the dataframe (post revision)
            eval_df = pe.call_evals(revisions_df, col=requirement_col_post_revision, eval_config=eval_config)
            # Get list of failed eval functions
            eval_df = pe.get_failed_evals(eval_df)
            post_weighted_values = pe.add_weighted_column(eval_df, include_funcs, score_weights, "weighted_value")
            # Map the failed eval functions to rule groups (as defined in the config.yaml file)
            eval_df = pe.map_failed_eval_col_to_rule_group(eval_df, eval_to_rule_map=config["SECTION_4_RULE_GROUPS"], failed_eval_col='failed_evals')
            # Drop requirements which pass acceptance criteria
            # At present, the criteria is len(failed_evals_rule_ids) == 0
            eval_df = eval_df.loc[eval_df['failed_evals_rule_ids'].str.len() > 0]
            eval_df['iternum'] = i+1
            eval_df['initial_weighted_values'] = eval_df[id_col].map(initial_weighted_values_dict)
            eval_df.to_excel(f"{output_directory}/eval_df_iter_{prompt[-1]}_{i+1}.xlsx")

    # Compile results
    compiled_df = utils.concat_matching_dataframes(
        _path=output_directory,                     # base directory to scan
        _regex=rf"eval_df_iter_{prompt[-1]}_.*.xlsx$",             # regex applied to filenames
        recursive=False,
        case_sensitive=True,
        match_on="name",
        read_kwargs=None,
        check_list_like_columns=True)

    compiled_df = compiled_df.loc[compiled_df['iternum'] == iternum]
    compiled_df.to_excel(f'{output_directory}/compiled_df_{prompt[-1]}.xlsx')

# Compile all results
all_results_df = utils.concat_matching_dataframes(
    _path=output_directory,                     # base directory to scan
    _regex=rf"compiled_df.*.xlsx$",             # regex applied to filenames
    recursive=False,
    case_sensitive=True,
    match_on="name",
    read_kwargs=None,
    check_list_like_columns=True)
all_results_df.to_excel(f'{output_directory}/all_results_df.xlsx')