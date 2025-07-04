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
from src.prj_logger import ProjectLogger
from src import pd_utils
#from src.components.sectionalize import Sectionalize
from src.components.preprocess import TextPreprocessor, Sectionalize, PreprocessIncoseGuide, ProcessIncoseTemplates
from src.components import prompteval as pe
from src.components.promptrunner import PromptRunner


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
REPLACE_TOKENS = ['INCOSE-TP-2010-006-04| VERS/REV:4  |  1 July 2023']
REPLACE_WITH = ' ' 
# define selected base template name used to build INCOSE rules prompts
SELECTED_BASE_TEMPLATE_NAME = 'req-reviewer-instruct-1'

# load logger
ProjectLogger('reviewer',f"{OUTPUT_DATA_FOLDER}/reviewer.log").config()

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
base_template_messages = pe.load_prompt_base_templates()[SELECTED_BASE_TEMPLATE_NAME]

# build INCOSE templates from base messages via class ProcessIncoseTemplates
incose_template_builder = ProcessIncoseTemplates(
    df=incose_guide_sections_df,
    base_messages=base_template_messages,
    output_data_folder_path=OUTPUT_DATA_FOLDER
)
prompt_templates= incose_template_builder().templates

# load prompt associations which associates evaluation 
# funcs with specific prompt templates (incose rules)
prompt_associations = pe.load_prompt_associations()
# has the associations into a config dictionary 
# where each key value is a dictionary containing 
# the eval func and associated template 
evals_config = pe.load_evaluation_config(prompt_associations, prompt_templates)

# instantiate prompt runner
runner = PromptRunner(
    llm=ChatOpenAI(api_key=OPENAI_API_KEY, model='gpt-4o-mini'),
    use_structured_llm=False,
    pydantic_model=None
)

# run evaluation loop
pe.run_eval_loop(
    df=reqs_df,
    runner=runner,
    OUTPUT_DATA_FOLDER=OUTPUT_DATA_FOLDER,
    evals_config=evals_config,
)

# generate revisions df
revisions_df = pe.generate_revisions_df(
    op=OUTPUT_DATA_FOLDER,
    pat="df_*",
    requirement_col='Requirement'
)

# merge revisions df to original requirements
reqs_df = pe.merge_revisions_df(
    reqs_df, revisions_df
)