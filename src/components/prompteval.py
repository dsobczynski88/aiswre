import re
import logging
import asyncio
import nest_asyncio
from pathlib import Path
from typing import Union, List
import pandas as pd
from tqdm import tqdm
from langchain_core.prompts.chat import ChatPromptTemplate
import src
from src import pd_utils
from src.prj_logger import get_logs

LOGGERNAME = f"{src.BASE_LOGGERNAME}.prompteval"
proj_logger = logging.getLogger(LOGGERNAME)

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