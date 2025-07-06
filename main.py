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
from src import utils
import src.components.preprocess as pp
from src.components.preprocess import PreprocessIncoseGuide, BuildIncoseTemplates
from src.components.promptrunner import IncoseRequirementReviewer
from src.components import prompteval as pe


def run_eval_loop(df, runner, output_data_folder, failed_eval_col='failed_evals', max_iter=3):
    # run evaluation algorithm
    for iter in range(max_iter):
        proj_logger.info(f'Entering: iter num {iter} of run_eval_loop')
        if iter > 0:
            df = pd.read_excel(f"{output_data_folder}/revised_df_iter_{iter-1}.xlsx")
        df = df[[runner.id_col]]
        proj_logger.info(f'Calling evaluations for iter num {iter} of run_eval_loop')
        # run evals on df
        df = runner.call_evals(df, runner.id_col)
        df = runner.get_failed_evals(df)
        proj_logger.info(f'Evaluations completed for iter num {iter} of run_eval_loop')
        if (df is not None):
            pass_cond = df[failed_eval_col].str.len() == 0
            pass_rows = df[pass_cond]
            if len(pass_rows) > 0:
                proj_logger.info(f'{len(pass_rows)}/{len(df)} Requirements passed all criteria during iter num {iter} of run_eval_loop')
                df = df[~pass_rows]
            if len(df) > 0:
                proj_logger.info(f'{len(df)} Requirements still require evaluation')
                # run prompts for requirements containing failed evals
                evals_lists = list(df[failed_eval_col].values)
                print(f"Eval lists: {evals_lists}")
                args_lists = list(df[incose_reviewer.id_col].values)
                revised_df = incose_reviewer(evals_lists, args_lists)
                revised_df['revision'] = iter + 1
                revised_df = revised_df[[incose_reviewer.id_col,'revision']]
                utils.to_excel(df, output_data_folder, str(iter), 'revised_df_iter')
            proj_logger.info(f'Exiting: iter num: {iter} of run_eval_loop')


if __name__ == "__main__":

    # load constants and environment variables
    DOT_ENV = dotenv_values(".env")
    # instantiate openai client
    OPENAI_API_KEY = DOT_ENV['OPENAI_API_KEY']
    # define selected base template name used to build INCOSE rules prompts
    SELECTED_BASE_TEMPLATE_NAME = 'req-reviewer-instruct-1'
    # define folder to store run outputs
    RUN_NAME = f"run-{utils.get_current_date_time()}"
    # define output data folder
    OUTPUT_DATA_FOLDER = f'./src/data/{RUN_NAME}'
    # make run folder 
    Path(OUTPUT_DATA_FOLDER).mkdir(parents=True, exist_ok=True)    
    # define incose guide filepath
    INCOSE_GUIDE_FILEPATH = './src/data/incose_gtwr.pdf'
    # define regex pattern to split out rules in Section 4 of Guide
    INCOSE_SECTIONS_REGEX_PAT = r'([1-9]\.([0-9]+\.)?[0-9]?)[\s]+R\d'
    # define preprocessing settings
    REPLACE_TOKENS = ['INCOSE-TP-2010-006-04| VERS/REV:4  |  1 July 2023', "{", "}"]
    REPLACE_WITH = ' ' 
    # define LLM model for evaluating requirements
    LLM_MODEL_NAME = 'gpt-4o-mini'
    # define the column name in the dataset containing the requirements
    DATASET_REQ_COLNAME = 'Requirement'
    # load logger
    ProjectLogger(src.BASE_LOGGERNAME,f"{OUTPUT_DATA_FOLDER}/{src.BASE_LOGGERNAME}.log").config()
    proj_logger = logging.getLogger(src.BASE_LOGGERNAME)
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

    # build INCOSE templates from base messages via class BuildIncoseTemplates
    incose_template_builder = BuildIncoseTemplates(
        df=incose_guide_sections_df,
        base_messages=base_template_messages,
        output_data_folder_path=OUTPUT_DATA_FOLDER
    )

    # instantiate incose requirement reviewer
    incose_reviewer = IncoseRequirementReviewer(
        llm=ChatOpenAI(api_key=OPENAI_API_KEY, model=LLM_MODEL_NAME),
        use_structured_llm=False,
        pydantic_model=None,
        templates=incose_template_builder.templates,
        evals_config=incose_template_builder.evals_config,
        id_col=DATASET_REQ_COLNAME
    )

    # run evaluation loop
    run_eval_loop(
        df=reqs_df,
        runner=incose_reviewer,
        output_data_folder=OUTPUT_DATA_FOLDER,
        failed_eval_col='failed_evals',
        max_iter=3
    )

    # generate revisions df
    revisions_df = utils.generate_revisions_df(
        op=OUTPUT_DATA_FOLDER,
        pat="revised_df*",
        requirement_col='Requirement'
    )

    # merge revisions df to original requirements
    reqs_df = utils.merge_revisions_df(
        reqs_df, revisions_df
    )