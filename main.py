import re
from pathlib import Path
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
from aiswre.prj_logger import ProjectLogger
from aiswre.utils import pd_utils, prompt_utils
from aiswre.preprocess.sectionalize import Sectionalize
from aiswre.promptengg import prompteval as pe
from aiswre.promptengg.promptrunner import ParallelPromptRunner

# load environment variables
config = dotenv_values(".env")

# define output data folder
output_data_folder = './aiswre/data'

# load logger
proj_log = ProjectLogger('reviewer',f"{output_data_folder}/reviewer.log").config()

# data ingestion (parse incose guide)
incose_guide_fp = './aiswre/data/incose_gtwr.pdf'
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
prompt_templates= prompt_utils.assemble_prompt_templates_from_df(incose_guide_sections_df, system_message_colname='system_message', user_message_colname='user_message')

# load requirements dataset
reqs_df = pd.read_excel('./aiswre/data/software_requirements.xlsx')

# load prompt associations which associates evaluation funcs with specific prompt templates (incose rules)
prompt_associations = pe.load_prompt_associations()
# has the associations into a config dictionary where each key value is a dictionary containing the eval func and associated template 
evals_config = pe.load_evaluation_config(prompt_associations, prompt_templates) # might not need this --- could combine with above "prompt associations" variable

# instantiate openai client
secret_key = config['OPENAI_API_KEY']

# define pydantic model
class Requirement:
    revision: str = Field("A revised software requirement")

# instantiate prompt runner
ppr = ParallelPromptRunner(
    llm=ChatOpenAI(api_key=secret_key, model='gpt-4o-mini'),
    pydantic_model=None
)

def run_prompts_for_failed_evals(df, runner, evals_config, failed_eval_col='failed_evals', id_col='Requirement'):
    failed_evals = df[failed_eval_col].values
    _args=df[id_col].values
    chains=[runner.assemble_eval_chain_list(fe, evals_config, runner.llm) for fe in failed_evals]
    async_tasks = runner.run_multiple_chains(chains, _args)
    results = asyncio.run(async_tasks)
    results_df = pd.DataFrame(results)
    results_df = results_df.rename(columns={0:'Requirement'})
    return results_df

def run_eval_loop(df, prompt_associations, runner, evals_config, failed_eval_col='failed_evals', id_col='Requirement', max_iter=3):
    # run evaluation algorithm
    eval_funcs = pe.get_eval_funcs(prompt_associations)
    for iter in range(max_iter):
        # run evals on df
        df = pe.call_evals(df, eval_funcs, 'Requirement')
        df = pe.get_failed_evals(df)
        # for reqs where all evals passed, pop off these rows into revised_df
        popped_rows = df.drop(df[df[failed_eval_col].str.len() == 0].index)
        if popped_rows:
            try:
                assert revised_df
            except NameError:
                revised_df = popped_rows
            else:
                revised_df = pd.concat([revised_df, popped_rows], axis=0, ignore_index=False)
                pd_utils.to_excel(revised_df, output_data_folder, iter, 'revised_df')
        # update df
        df = df[df[failed_eval_col].str.len() > 0]
        pd_utils.to_excel(df, output_data_folder, iter, 'df_pre_prompt')
        if not len(df):
            break
        else:
            # run prompts for requirements containing failed evals
            df = run_prompts_for_failed_evals(df, runner, evals_config, failed_eval_col, id_col)
            pd_utils.to_excel(df, output_data_folder, iter, 'df_post_prompt')
    else:
        #log that not all requirements fully passed
        pass