from dotenv import dotenv_values
from src import utils

# Load config settings
DOT_ENV = dotenv_values("../.env")
config = utils.load_config("../config.yaml")

# Create a unique run-id folder to store outputs
config["FILE_LOCATIONS"]["MAIN_DATA_FOLDER"] = "../src/data"
output_directory = utils.make_output_directory(config["FILE_LOCATIONS"])

from src.components import prompteval as pe
# Functions currently requiring remediation
exclude_funcs = [
    'eval_explicit_enumeration',
    'eval_follows_style_guide',
    'eval_has_correct_grammar',
    'eval_has_supporting_diagram_or_model_reference',
    'eval_is_structured_set',
    'eval_is_unique_expression',
    'eval_has_explicit_conditions_for_single_action',
    'eval_is_structured_statement'
]
# Make evaluation function config
eval_config = pe.make_eval_config(pe, exclude_funcs=exclude_funcs)


import ast
import asyncio
from pydantic import BaseModel, Field, SecretStr
from src.components.promptrunner import RateLimitOpenAIClient

# Run prompt for requirements containing failed evaluations (asychronously with retry and backoff to overcome rate limit errors)
openai_api_key = SecretStr(str(DOT_ENV['OPENAI_API_KEY']))
openai_client = RateLimitOpenAIClient(api_key=openai_api_key.get_secret_value())

# Define data validation model
class Revision(BaseModel):
    requirement_id: int = Field(description="The original requirement id provided by the user")
    requirement: str = Field(description="The original requirement provided by the user")
    revision: str = Field(description="The revised AI-generated output requirement")
    review: str = Field(description="A summary of the thought process used to generate the revision and why it was chosen")

# Define function to perform AI reviews
async def run_reviewer_prompts(requirements, ids, model: str = 'gpt-4o-mini', ):
       
    req_tups = zip(requirements, ids)
    # Example of making multiple concurrent requests
    tasks = []
    for i, req_tup in enumerate(req_tups):
        tasks.append(
            openai_client.chat_completion_parse(
                model=model,
                messages=[
                    {"role":"system", "content": 'You are a meticulous requirements analyst tasked with verifying the quality and clarity of a given requirement based on a detailed checklist.'},
                    {"role":"user", "content": f'Given the following id and requirement:\n\n"""\nrequirement_id:{req_tup[1]} requirement:{req_tup[0]}\n"""\n\nPlease systematically assess the requirement against each of the following criteria, providing a clear answer (Yes/No) and a concise explanation for each:\n\n1. Is the requirement clearly stated, avoiding ambiguous or vague terms?\n2. Can all readers (technical and non-technical stakeholders) understand the requirement without confusion?\n3. Is it written in plain, simple language with no jargon or undefined acronyms?\n4. Does the requirement use active voice and a positive statement (e.g., \'The system shall…\')?\n5. Does it avoid subjective words such as \'user-friendly\', \'fast\', or \'optimal\'?\n6. Is the requirement phrased as a single, atomic statement (not combining multiple requirements)?\n7. Does the requirement address a single capability or attribute?\n8. Avoid compound requirements that include more than one functionality or condition.\n9. Split complex requirements into multiple focused ones if necessary.\n10. Is the requirement stated in such a way that it can be verified through test, inspection, or analysis?\n11. Are the acceptance criteria or measurable parameters clearly indicated or implied?\n12. Could a tester or analyst objectively determine if the requirement is met or not?\n13. Does the requirement use terminology consistent with the rest of the documentation?\n14. Are key terms defined somewhere in a glossary or within the document?\n15. Is there any conflicting wording when compared to other requirements?\n16. Is the requirement traceable back to a stakeholder need, higher-level system requirement, or project objective?\n17. Does the requirement add value to the system, or is it redundant or unnecessary?\n18. Is the rationale for the requirement clear or documented (if applicable)?\n19. Does the requirement cover all relevant conditions and constraints (e.g., operating environment, performance parameters)?\n20. Is it realistically implementable within project constraints (time, cost, technology)?\n\nProvide your answer in a numbered list with each item corresponding to the checklist criteria.'}
                ],
                response_format=Revision
            )
        )
    
    # Wait for all requests to complete
    responses = await asyncio.gather(*tasks)
    
    # Process responses
    processed_responses = []
    for i, response in enumerate(responses):
        
        output = dict(response)
        message = response.choices[0].message

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
        
        print(f"\nResponse {i+1}:")
        print(output)
        processed_responses.append(output)
        
    #return processed_responses
    return processed_responses

# Call the evaluations on the dataframe 
import pandas as pd
iternum=3
id_col = 'requirement_id'

for i in range(iternum):
    if i == 0:
        # Load requirements
        df = pd.read_excel('../src/data/demo_dataset.xlsx')
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
        df.to_excel(f"{output_directory}/eval_df_initial.xlsx")
    else:
        df = revisions_df.copy()
        requirement_col = 'revision'

    requirements = list(df[requirement_col].values)
    ids = list(df[id_col].values)     
    revisions = asyncio.run(run_reviewer_prompts(requirements, ids))
    revisions_df = pd.DataFrame(revisions)
    # Call the evaluations on the dataframe 
    eval_df = pe.call_evals(revisions_df, col=requirement_col, eval_config=eval_config)
    # Get list of failed eval functions
    eval_df = pe.get_failed_evals(eval_df)
    # Map the failed eval functions to rule groups (as defined in the config.yaml file)
    eval_df = pe.map_failed_eval_col_to_rule_group(eval_df, eval_to_rule_map=config["SECTION_4_RULE_GROUPS"], failed_eval_col='failed_evals')
    # Drop requirements which pass acceptance criteria
    # At present, the criteria is len(failed_evals_rule_ids) == 0
    eval_df = eval_df.loc[eval_df['failed_evals_rule_ids'].str.len() > 0]
    eval_df.to_excel(f"{output_directory}/eval_df_iter_{i+1}.xlsx")