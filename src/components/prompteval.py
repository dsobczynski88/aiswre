import re
import logging
import asyncio
import nest_asyncio
from pathlib import Path
from typing import Union, List
import pandas as pd
from tqdm import tqdm
from langchain_core.prompts.chat import ChatPromptTemplate
from src import pd_utils
from src.prj_logger import get_logs

BASE_LOGGERNAME = "reviewer"
LOGGERNAME = f"{BASE_LOGGERNAME}.prompteval"
proj_logger = logging.getLogger(LOGGERNAME)

@get_logs(LOGGERNAME)
def run_eval_loop(df, runner, evals_config, output_data_folder, failed_eval_col='failed_evals', id_col='Requirement', max_iter=3):
    # run evaluation algorithm
    proj_logger.info('Entering: run_eval_loop')
    for iter in range(max_iter):
        proj_logger.info(f'Entering: iter num {iter} of run_eval_loop')
        if iter > 0:
            df = pd.read_excel(f"{output_data_folder}/df_{iter-1}.xlsx")
        df = df[[id_col]]
        proj_logger.info(f'Calling evaluations for iter num {iter} of run_eval_loop')
        # run evals on df
        df = call_evals(df, evals_config, id_col)
        df = get_failed_evals(df)
        proj_logger.info(f'Evaluations completed for iter num {iter} of run_eval_loop')
        if (df is not None):
            popped_cond = df[failed_eval_col].str.len() == 0
            popped_rows = df[popped_cond]
            if len(popped_rows) > 0:
                proj_logger.info(f'Requirements were found that passed all criteria during iter num {iter} of run_eval_loop')
            proj_logger.info(f'The Requirements df was originally {len(df)} rows')
            df = df[~popped_cond]
            proj_logger.info(f'The Requirements df is now {len(df)} rows')
            if len(df) > 0:
                # run prompts for requirements containing failed evals
                df = run_prompts_for_failed_evals(df, runner, evals_config, failed_eval_col, id_col)
                df = call_evals(df, evals_config, id_col)
                df = get_failed_evals(df)
                print(df.columns)
                df['revision'] = iter + 1
                df = df[[id_col,'revision']]
                pd_utils.to_excel(df, output_data_folder, str(iter), 'df')
            proj_logger.info(f'Exiting: iter num: {iter} of run_eval_loop')

@get_logs(LOGGERNAME)
def run_prompts_for_failed_evals(df, runner, evals_config, failed_eval_col='failed_evals', id_col='Requirement'):
    failed_evals = df[failed_eval_col].values
    _args=df[id_col].values
    chains=runner.assemble_eval_chain_list(failed_evals, evals_config)
    async_tasks = runner.run_multiple_chains(chains, _args)
    results = asyncio.run(async_tasks)
    results_df = pd.DataFrame(results)
    results_df = results_df.rename(columns={0:id_col})
    return results_df

@get_logs(LOGGERNAME)
def load_prompt_base_templates():
    '''define a set of base template descriptions and messages'''
    prompt_base_templates = {
        'req-reviewer-instruct-1': {
            'name': 'req-reviewer-instruct-1',
            'description': 'This template applies the standard text model practice of writing clear instructions by specifying steps. In this case, the prompt seeks to use OpenAI gpt models to perform a robust revision of software requirements based on INCOSE best practices.',
            'system': 'Step 1 - The user will hand over a Requirement, Criteria, and Examples. Your task is to revise the Requirement as per the provided Criteria and Examples, starting with the phrase "Initial Revision:".\nStep 2 - Compare the initial revision performed in Step 1 against the criteria to determine if any additional revisions are necessary. Let\'s think step-by-step.\nStep 3 - Return the final requirement revision based on Steps 1 and 2, starting with the phrase \"Final Revision:\".',
            'user': 'Requirement: {req}\nCriteria:\n{definition}\nExamples:\n{examples}'
        },
        'req-reviewer-instruct-2': {
            'name': 'req-reviewer-instruct-2'
        }  
    }
    return prompt_base_templates

@get_logs(LOGGERNAME)
def load_prompt_associations():
    '''define the associations between prompt evaluations and INCOSE rules'''
    prompt_associations = [
        ('eval_is_in_passive_voice',eval_is_in_passive_voice,'R2'),
        ('eval_if_vague_verb',eval_if_vague_verb,'R3'),
        #('eval_has_a_def_article',eval_has_a_def_article,'R5'),
        ('eval_has_vague_terms',eval_has_vague_terms,'R7'),
        ('eval_has_escape_clause',eval_has_escape_clause,'R8'),
    ]
    return prompt_associations

@get_logs(LOGGERNAME)
def load_evaluation_config(prompt_associations, prompt_templates):
    '''load evaluations configuration'''
    evaluation_config = {}
    for assoc in prompt_associations:
        evaluation_config[assoc[0]] = {}
        evaluation_config[assoc[0]]["func"] = assoc[1]
        evaluation_config[assoc[0]]["template"] = prompt_templates[assoc[2]]
    return evaluation_config

@get_logs(LOGGERNAME)
def assemble_prompt_template_from_messages(system_message: str, user_message: str) -> ChatPromptTemplate:
    '''Create a ChatPromptTemplate given an input dictionary containing keys: system, user'''
    return ChatPromptTemplate.from_messages([
            ("system",system_message),
            ("human", user_message)
        ])

@get_logs(LOGGERNAME) 
def assemble_prompt_templates_from_df(df, system_message_colname='system_message', user_message_colname='user_message'):
    '''Loop over dataframe to build a unique prompt template for each row'''
    prompt_templates_config = {}
    for index, row in tqdm(df.iterrows()):
        system_message=row[system_message_colname]
        user_message=row[user_message_colname]
        try:
            prompt_templates_config[f"R{index+1}"] = assemble_prompt_template_from_messages(system_message, user_message)
        except ValueError:
            system_message=system_message.replace(".} ",".]")
            user_message=user_message.replace(".} ",".]")
            prompt_templates_config[f"R{index+1}"] = assemble_prompt_template_from_messages(system_message, user_message)
    return prompt_templates_config
    
@get_logs(LOGGERNAME)
def generate_revisions_df(op: str, pat: str, requirement_col: str):
    directory = Path(dir)
    matching_files = list(directory.rglob(pat))
    dfs=[]
    for file in matching_files:
        temp_df = pd.read_excel(file, index_col=[0])
        temp_df = temp_df.reset_index().rename(columns={requirement_col:f'Revised_{requirement_col}', 'index':f'{requirement_col}_#'})
        dfs.append(temp_df)
    # concat dfs
    revisions_df = pd.concat(dfs, ignore_index=True, axis=0)[[f'Revised_{requirement_col}',f'{requirement_col}_#','revision']]
    revisions_df = revisions_df[revisions_df[f'Revised_{requirement_col}'].str.strip() != '']
    pd_utils.to_excel(revisions_df, dir, False, 'revisions_df')
    return revisions_df
    
@get_logs(LOGGERNAME)
def merge_revisions_df(reqs_df, revisions_df):
    #merge latest revisions to original requirements dataframe
    reqs_df = pd.merge(
        left=reqs_df, right=revisions_df[[f'Revised_{requirement_col}',f'{requirement_col}_#']], on=f'{requirement_col}_#', how='left'
    )
    pd_utils.to_excel(reqs_df, dir, False, 'reqs_df_with_revisions')
    return reqs_df

@get_logs(LOGGERNAME)
def add_spaces(x: str) -> str:
    #return f" {x} "
    return x

@get_logs(LOGGERNAME)
def convert_bool_to_ohe(bool_result: bool) -> int:
    if bool_result:
        return 1
    else:
        return 0

@get_logs(LOGGERNAME)
def call_evals(df: pd.DataFrame, eval_config: dict, col: str) -> None:
    # run evals for each row of the dataframe
    for _index, _row in df.iterrows():
        for key, value in eval_config.items():  # fix this line
            eval_func_to_call = eval_config[key]["func"] 
            eval_result = eval_func_to_call(_row[col])
            df.loc[_index, key] = convert_bool_to_ohe(
                eval_result
            )
    return df

@get_logs(LOGGERNAME)
def get_failed_evals(df: pd.DataFrame):
    eval_cols = [c for c in df.columns if c.startswith("eval")]
    df['failed_evals'] = df[eval_cols].apply(lambda _l: [eval_cols[e[0]] for e in enumerate(_l) if e[1]==1.0], axis=1)
    return df

@get_logs(LOGGERNAME)    
def eval_is_in_passive_voice(text: str) -> bool:
    """
    R2: Criteria from 4.1.2 INCOSE Guide to Writing Requirements:
        check if text is written in passive voice
    """
    'check for pattern shall + be + [main_verb]'
    pass

@get_logs(LOGGERNAME)
def eval_if_vague_verb(text: str) -> bool:
    """
    R3: Criteria from 4.1.3 INCOSE Guide to Writing Requirements:
        check if the requirements uses a vague verb 
    """
    vague_verbs = [
        "support", "process", "handle", "track", "manage", "flag"
    ]
    for verb in vague_verbs:
        if add_spaces(verb) in text:
            return True
    else:
        return False

@get_logs(LOGGERNAME)
def eval_has_a_def_article(text: str) -> bool:
    """
    R5: Criteria from 4.1.5 INCOSE Guide to Writing Requirements:
        check if text contains indefinite article \"a\" 
    """
    if add_spaces("a") in text:
        return True
    else:
        return False

@get_logs(LOGGERNAME)
def eval_has_vague_terms(text: str) -> bool:
    """
    R7: Criteria from 4.1.7 INCOSE Guide to Writing Requirements:
        check if text contains vague terms        
    """
    vague_terms = [
        "some", "any", "allowable", "several", "many", "a lot of", "a few", "almost always", 
        "very nearly", "nearly", "about", "close to", "almost","approximate","ancillary", "relevant", 
        "routine", "common", "generic", "significant","flexible", "expandable", "typical", "sufficient", 
        "adequate", "appropriate", "efficient", "effective", "proficient", "reasonable","customary",
        "usually", "approximately", "sufficiently","typically"
    ]
    for term in vague_terms:
        if add_spaces(term) in text:
            return True
    else:
        return False

@get_logs(LOGGERNAME)
def eval_has_escape_clause(text:str) -> bool:
    """
    R8: Criteria from 4.1.8 INCOSE Guide to Writing Requirements:
        check if text contains escape clauses which state vague
        conditions or possibilities        
    """
    clauses = [
         "so far as is possible", "as little as possible", "where possible", 
         "as much as possible", "if it should prove necessary", "if necessary", 
         "to the extent necessary", "as appropriate", "as required", "to the extent practical", 
         "if practicable"
    ]
    for clause in clauses:
        if add_spaces(clause) in text:
            return True
    else:
        return False