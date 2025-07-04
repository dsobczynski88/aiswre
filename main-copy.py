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
from src.components import prompteval as pe
from src.components.promptrunner import ParallelPromptRunner

# load environment variables
config = dotenv_values(".env")

# define output data folder
output_data_folder = './src/data'

# load logger
proj_log = ProjectLogger('reviewer',f"{output_data_folder}/reviewer.log").config()

#BASE_LOGGERNAME = "reviewer"
#LOGGERNAME = f"{BASE_LOGGERNAME}.main"
#proj_logger = logging.getLogger(LOGGERNAME)
proj_logger = logging.getLogger('reviewer.main')

# data ingestion (parse incose guide)
incose_guide_fp = './src/data/incose_gtwr.pdf'
sectionalizer = Sectionalize(regex=r'([1-9]\.([0-9]+\.)?[0-9]?)[\s]+R\d')
text = sectionalizer.get_pdf_text(fp=Path(incose_guide_fp), start_page=65, end_page=115)
sectionalizer.save_text(text=text, op=Path(f"{output_data_folder}/extract.txt"))
incose_guide_sections_df = sectionalizer.get_sections_df(text)
incose_guide_sections_df = sectionalizer.add_section_text(incose_guide_sections_df, text)
incose_guide_sections_df =sectionalizer.get_incose_definition(incose_guide_sections_df)
incose_guide_sections_df =sectionalizer.get_incose_elaboration(incose_guide_sections_df)
incose_guide_sections_df =sectionalizer.get_incose_examples(incose_guide_sections_df)
pd_utils.to_excel(incose_guide_sections_df, output_data_folder, False, 'incose_guide_sections_df')
# add templates for each rule to incose_guide_sections_df using a specified base template
prompt_base_templates = pe.load_prompt_base_templates()
# select base template
base_template = prompt_base_templates['req-reviewer-instruct-1']
base_template_system_message = base_template['system']
base_template_user_message = base_template['user']
# assign base template to incose guide
incose_guide_sections_df['system_base_message'] = base_template_system_message
incose_guide_sections_df['user_base_message'] = base_template_user_message
# replace relevant template variables with INCOSE data (e.g., definition, examples)
incose_guide_sections_df['system_message'] = incose_guide_sections_df['system_base_message']
incose_guide_sections_df['user_message'] = incose_guide_sections_df[['user_base_message','definition']].apply(lambda l: l[0].replace('{definition}',l[1]), axis=1)
incose_guide_sections_df['user_message'] = incose_guide_sections_df[['user_message','examples']].apply(lambda l: l[0].replace('{examples}',l[1]), axis=1)
pd_utils.to_excel(incose_guide_sections_df, output_data_folder, False, 'incose_guide_sections_df')
# build prompt templates based on incose rules
prompt_templates= pe.assemble_prompt_templates_from_df(incose_guide_sections_df, system_message_colname='system_message', user_message_colname='user_message')
# load requirements dataset
reqs_df = pd.read_excel('./src/data/software_requirements_1.xlsx', index_col=[0])
reqs_df = reqs_df.reset_index().rename(columns={'index':'Requirement_#'})

# load prompt associations which associates evaluation funcs with specific prompt templates (incose rules)
prompt_associations = pe.load_prompt_associations()
# has the associations into a config dictionary where each key value is a dictionary containing the eval func and associated template 
evals_config = pe.load_evaluation_config(prompt_associations, prompt_templates) # might not need this --- could combine with above "prompt associations" variable

print(evals_config)

# instantiate openai client
secret_key = config['OPENAI_API_KEY']

# define pydantic model
class Requirement:
    revision: str = Field("A revised software requirement")

# instantiate prompt runner
ppr = ParallelPromptRunner(
    llm=ChatOpenAI(api_key=secret_key, model='gpt-4o-mini'),
    use_structured_llm=False,
    pydantic_model=None
)

#reqs_df = pe.call_evals(reqs_df, evals_config, 'Requirement')
#reqs_df = pe.get_failed_evals(reqs_df)
'''
pe.run_eval_loop(
    df=reqs_df,
    runner=ppr,
    output_data_folder=output_data_folder,
    evals_config=evals_config,
)
'''


# Define the directory and naming string
directory = Path(f"{output_data_folder}")
naming_string = "df_*"

# Get all files matching the naming string
matching_files = list(directory.rglob(naming_string))

# Print the matching files
dfs=[]
for file in matching_files:
    temp_df = pd.read_excel(file, index_col=[0])
    temp_df = temp_df.reset_index().rename(columns={'Requirement':'Revised_Requirement', 'index':'Requirement_#'})
    dfs.append(temp_df)
# concat dfs
revisions_df = pd.concat(dfs, ignore_index=True, axis=0)[['Revised_Requirement','Requirement_#','revision']]
revisions_df = revisions_df[revisions_df['Revised_Requirement'].str.strip() != '']
pd_utils.to_excel(revisions_df, output_data_folder, False, 'revisions_df')

#merge latest revisions to original requirements dataframe
reqs_df = pd.merge(
    left=reqs_df, right=revisions_df[['Revised_Requirement','Requirement_#']], on='Requirement_#', how='left'
)
pd_utils.to_excel(reqs_df, output_data_folder, False, 'reqs_df_with_revisions')
