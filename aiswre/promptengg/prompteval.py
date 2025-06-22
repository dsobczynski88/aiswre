import re
import asyncio
import nest_asyncio
from pathlib import Path
from typing import Union, List
import pandas as pd
from aiswre.utils import pd_utils, prompt_utils
from aiswre.prj_logger import get_logs

BASE_LOGGERNAME = "reviewer"
LOGGERNAME = f"{BASE_LOGGERNAME}.prompteval"

@get_logs(LOGGERNAME)
def call_evals(df: pd.DataFrame, eval_config: dict, col: str) -> None:
    # run evals for each row of the dataframe
    for _index, _row in df.iterrows():
        for key, value in eval_config.items():  # fix this line
            eval_func_to_call = eval_config[key]["func"] 
            eval_result = eval_func_to_call(_row[col])
            df.loc[_index, key] = prompt_utils.convert_bool_to_ohe(
                eval_result
            )
    print(df.head(5))
    return df

@get_logs(LOGGERNAME)
def get_failed_evals(df: pd.DataFrame):
    eval_cols = [c for c in df.columns if c.startswith("eval")]
    df['failed_evals'] = df[eval_cols].apply(lambda _l: [eval_cols[e[0]] for e in enumerate(_l) if e[1]==1.0], axis=1)
    return df

@get_logs(LOGGERNAME)
def load_prompt_base_templates():
    '''define a set of base template descriptions and messages'''
    prompt_base_templates = {
        'req-reviewer-instruct-1': {
            'name': 'req-reviewer-instruct-1',
            'description': 'This template applies the standard text model practice of writing clear instructions by specifying steps. In this case, the prompt seeks to use OpenAI gpt models to perform a robust revision of software requirements based on INCOSE best practices.',
            'system': 'Step 1 - The user will hand over a Requirement, Criteria, and Examples. Your task is to revise the Requirement as per the provided Criteria and Examples, starting with the phrase "Initial Revision:".\nStep 2 - Compare the initial revision performed in Step 1 against the criteria to determine if any additional revisions are necessary. Let\'s think step-by-step.\nStep 3 - Return the final requirement revision based on Steps 1 and 2, starting with the phrase \"Final Revision:\".',
            'user': 'Requirement: {req}\nCriteria: {definition}\nExamples: {examples}'
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

def get_eval_funcs(prompt_associations):
    funcs = []
    for assoc in prompt_associations:
        funcs.append(assoc[1])
    return funcs
    

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
        if prompt_utils.add_spaces(verb) in text:
            return True
    else:
        return False

@get_logs(LOGGERNAME)
def eval_has_a_def_article(text: str) -> bool:
    """
    R5: Criteria from 4.1.5 INCOSE Guide to Writing Requirements:
        check if text contains indefinite article \"a\" 
    """
    if prompt_utils.add_spaces("a") in text:
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
        if prompt_utils.add_spaces(term) in text:
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
        if prompt_utils.add_spaces(clause) in text:
            return True
    else:
        return False