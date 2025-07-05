import argparse
import re
from pathlib import Path
import logging
import pandas as pd
from functools import partial
from dotenv import dotenv_values
from tqdm import tqdm
import asyncio
import nest_asyncio
from pydantic import BaseModel, Field
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
    RunnableSequence
)
import src
from src.prj_logger import ProjectLogger, get_logs
from src import pd_utils
import src.components.preprocess as pp
from src.components.preprocess import PreprocessIncoseGuide, BuildIncoseTemplates
from src.components import prompteval as pe
from src.components.promptrunner import PromptRunner

@get_logs(src.BASE_LOGGERNAME)
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
        df = pe.call_evals(df, evals_config, id_col)
        df = pe.get_failed_evals(df)
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
                df = pe.call_evals(df, evals_config, id_col)
                df = pe.get_failed_evals(df)
                print(df.columns)
                df['revision'] = iter + 1
                df = df[[id_col,'revision']]
                pd_utils.to_excel(df, output_data_folder, str(iter), 'df')
            proj_logger.info(f'Exiting: iter num: {iter} of run_eval_loop')

@get_logs(src.BASE_LOGGERNAME)
def run_prompts_for_failed_evals(df, runner, evals_config, failed_eval_col='failed_evals', id_col='Requirement'):
    """Runs a prompt chain (RunnableSequence) for each row in the input dataframe (df). The RunnableSequence is constructed using the text from the id_col 
	as input and the failed_evals col (List[str]) where each element of the failed_evals_col refers to a specific evaluation function designed to revise
	a given input requirement against a specific criteria. The evals_config looks up the string name each evaluation function to fetch the associated function and template.
	Returns a dataframe containing only the prompt chain response for each row in the original input dataframe. 
	
    Arguments:
		df (pd.DataFrame): The dataframe containing the input text (id_col) and the list of function names to be called for evaluating the requirement (failed_eval_col)
		runner (PromptRunner): The PromptRunner instance used to run each Runnable Sequence (each element in df) asychronously
		evals_config (dict): A dictionary which holds the mapping between function name, INCOSE rule, and associated prompt template. 
		failed_eval_col (str): The column name in the dataframe containing the list of criteria which the input requirement did not satisfy.
		id_col (str): The column name in the dataframe containing the input requirement text which is to be evaluated using the runner.
	"""
    failed_evals = df[failed_eval_col].values
    _args=df[id_col].values
    chains=runner.assemble_eval_chain_list(failed_evals, evals_config)
    async_tasks = runner.run_multiple_chains(chains, _args)
    results = asyncio.run(async_tasks)
    results_df = pd.DataFrame(results)
    results_df = results_df.rename(columns={0:id_col})
    return results_df

@get_logs(src.BASE_LOGGERNAME)
def generate_revisions_df(op: str, pat: str, requirement_col: str):
    directory = Path(op)
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
    
@get_logs(src.BASE_LOGGERNAME)
def merge_revisions_df(reqs_df, revisions_df, requirement_col='Requirement'):
    #merge latest revisions to original requirements dataframe
    reqs_df = pd.merge(
        left=reqs_df, right=revisions_df[[f'Revised_{requirement_col}',f'{requirement_col}_#']], on=f'{requirement_col}_#', how='left'
    )
    pd_utils.to_excel(reqs_df, dir, False, 'reqs_df_with_revisions')
    return reqs_df

if __name__ == "__main__":

    # load constants and environment variables
    DOT_ENV = dotenv_values(".env")
    # instantiate openai client
    OPENAI_API_KEY = DOT_ENV['OPENAI_API_KEY']
    # define output data folder
    OUTPUT_DATA_FOLDER = './src/data'
    # define incose guide filepath
    INCOSE_GUIDE_FILEPATH = './src/data/incose_gtwr.pdf'
    # define regex pattern to split out rules in Section 4 of Guide
    INCOSE_SECTIONS_REGEX_PAT = r'([1-9]\.([0-9]+\.)?[0-9]?)[\s]+R\d'
    # define preprocessing settings
    REPLACE_TOKENS = ['INCOSE-TP-2010-006-04| VERS/REV:4  |  1 July 2023', "{", "}"]
    REPLACE_WITH = ' ' 
    # define selected base template name used to build INCOSE rules prompts
    SELECTED_BASE_TEMPLATE_NAME = 'req-reviewer-instruct-1'
    # define LLM model for evaluating requirements
    LLM_MODEL_NAME = 'gpt-4o-mini'

    # load logger
    proj_logger = ProjectLogger('reviewer',f"{OUTPUT_DATA_FOLDER}/reviewer.log").config()

    # load dataset
    reqs_df = pd.read_excel('./src/data/software_requirements_1.xlsx', index_col=[0])
    reqs_df = reqs_df.reset_index().rename(columns={'index':'Requirement_#'})

    # parse INCOSE guide using the PreprocessIncoseGuide class
    incose_preprocessor = PreprocessIncoseGuide(INCOSE_SECTIONS_REGEX_PAT).preprocess_rules_section_4(
        inpath=Path(INCOSE_GUIDE_FILEPATH),
        outpath=Path(OUTPUT_DATA_FOLDER),
        start_page=65,
        end_page=115,
        replace_tokens=REPLACE_TOKENS,
        replace_with=REPLACE_WITH
    )
    incose_guide_sections_df = incose_preprocessor.df

    # load selected base template messages
    base_template_messages = pp.BASE_TEMPLATES[SELECTED_BASE_TEMPLATE_NAME]

    # build INCOSE templates from base messages via class ProcessIncoseTemplates
    incose_template_builder = BuildIncoseTemplates(
        df=incose_guide_sections_df,
        base_messages=base_template_messages,
        output_data_folder_path=OUTPUT_DATA_FOLDER
    )
    incose_template_builder()
    # load evals_config which associates evaluation 
    # funcs with specific prompt templates (based on incose rules)
    # where each key value is a dictionary containing 
    # the eval func and associated template 
    evals_config = incose_template_builder.evals_config

    # instantiate prompt runner
    runner = PromptRunner(
        llm=ChatOpenAI(api_key=OPENAI_API_KEY, model=LLM_MODEL_NAME),
        use_structured_llm=False,
        pydantic_model=None
    )

    # run evaluation loop
    run_eval_loop(
        df=reqs_df,
        runner=runner,
        OUTPUT_DATA_FOLDER=OUTPUT_DATA_FOLDER,
        evals_config=evals_config,
    )

    # generate revisions df
    revisions_df = generate_revisions_df(
        op=OUTPUT_DATA_FOLDER,
        pat="df_*",
        requirement_col='Requirement'
    )

    # merge revisions df to original requirements
    reqs_df = merge_revisions_df(
        reqs_df, revisions_df
    )