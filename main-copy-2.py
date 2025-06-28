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
from src.components.sectionalize import Sectionalize
from src.components.clean import TextPreprocessor
from src.components import prompteval as pe
from src.components.promptrunner import PromptRunner

#
# load constants
# load environment variables
DOT_ENV = dotenv_values(".env")
# instantiate openai client
OPENAI_API_KEY = DOT_ENV['OPENAI_API_KEY']
# define output data folder
OUTPUT_DATA_FOLDER = './src/data'
# define incose guide filepath
INCOSE_GUIDE_FILEPATH = './src/data/incose_gtwr.pdf'
# define regex pattern to split out rules in Section 4 of Guide
INCOSE_SECTIONS_REGEX_PAT = r'([1-9]\.([0-9]+\.)?[0-9]?)[\s]+R\d'
#
# load logger
ProjectLogger('reviewer',f"{OUTPUT_DATA_FOLDER}/reviewer.log").config()
#
# load dataset
reqs_df = pd.read_excel('./src/data/software_requirements_1.xlsx', index_col=[0])
reqs_df = reqs_df.reset_index().rename(columns={'index':'Requirement_#'})
#
# parse INCOSE guide using the Sectionalize class
incose_guide = Sectionalize(regex=INCOSE_SECTIONS_REGEX_PAT)
incose_guide.get_pdf_text(
    fp=Path(INCOSE_GUIDE_FILEPATH), 
    start_page=65, 
    end_page=115
)
# save text pre cleaning
incose_guide.save_text(Path(f"{OUTPUT_DATA_FOLDER}/extract.txt"))
# run cleaning pipeline
incose_processor = TextPreprocessor()
incose_processor.pipeline = [
    partial(incose_processor.replace, replace_tokens=['INCOSE-TP-2010-006-04| VERS/REV:4  |  1 July 2023'], replace_with=' '),
    incose_processor.remove_multi_whitespace,
]
incose_guide.text = incose_processor.clean_text(text=incose_guide.text)
# save cleaned text
incose_guide.save_text(Path(f"{OUTPUT_DATA_FOLDER}/extract-post-clean.txt"))
incose_guide.parse_incose_guide()
incose_guide_sections_df = incose_guide.df

# add templates for each rule to incose_guide_sections_df using a specified base template
prompt_base_templates = pe.load_prompt_base_templates()
# select base template
base_template = prompt_base_templates['req-reviewer-instruct-1']
# assign base template to incose guide
incose_guide_sections_df['system_message'] = base_template['system']
incose_guide_sections_df['user_base_message'] = base_template['user']
# replace relevant template variables with INCOSE data (e.g., definition, examples)
incose_guide_sections_df['user_message'] = incose_guide_sections_df[['user_base_message','definition']].apply(lambda l: l[0].replace('{definition}',l[1]), axis=1)
incose_guide_sections_df['user_message'] = incose_guide_sections_df[['user_message','examples']].apply(lambda l: l[0].replace('{examples}',l[1]), axis=1)
# build prompt templates based on incose rules
prompt_templates= pe.assemble_prompt_templates_from_df(incose_guide_sections_df, system_message_colname='system_message', user_message_colname='user_message')
pd_utils.to_excel(incose_guide_sections_df, OUTPUT_DATA_FOLDER, False, 'incose_guide_sections_df')
# load prompt associations which associates evaluation funcs with specific prompt templates (incose rules)
prompt_associations = pe.load_prompt_associations()
# has the associations into a config dictionary where each key value is a dictionary containing the eval func and associated template 
evals_config = pe.load_evaluation_config(prompt_associations, prompt_templates) # might not need this --- could combine with above "prompt associations" variable
#
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

revisions_df = pe.generate_revisions_df(
    op=OUTPUT_DATA_FOLDER,
    pat="df_*",
    requirement_col='Requirement'
)

reqs_df = pe.merge_revisions_df(
    reqs_df, revisions_df
)