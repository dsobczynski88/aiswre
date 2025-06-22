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

#BASE_LOGGERNAME = "reviewer"
#LOGGERNAME = f"{BASE_LOGGERNAME}.main"
#proj_logger = logging.getLogger(LOGGERNAME)
proj_logger = logging.getLogger('reviewer.main')

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
reqs_df = pd.read_excel('./aiswre/data/software_requirements_1.xlsx')#.head(10)

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

def run_prompts_for_failed_evals(df, runner, evals_config, failed_eval_col='failed_evals', id_col='Requirement'):
    failed_evals = df[failed_eval_col].values
    _args=df[id_col].values
    chains=runner.assemble_eval_chain_list(failed_evals, evals_config)
    print(chains)
    print(type(chains), len(chains), type(chains[0]))
    async_tasks = runner.run_multiple_chains(chains, _args)
    results = asyncio.run(async_tasks)
    results_df = pd.DataFrame(results)
    results_df = results_df.rename(columns={0:id_col})
    return results_df

def run_eval_loop(df, runner, evals_config, failed_eval_col='failed_evals', id_col='Requirement', max_iter=3):
    # run evaluation algorithm
    proj_logger.info('Entering: run_eval_loop')
    for iter in range(max_iter):
        proj_logger.info(f'Entering: iter num {iter} of run_eval_loop')
        if iter > 0:
            df = pd.read_excel(f"{output_data_folder}/df_{iter-1}.xlsx")
        df = df[[id_col]]
        #df[failed_eval_col] = df[failed_eval_col].apply(lambda s: pd_utils.recast_eval(s),'')
        proj_logger.info(f'Calling evaluations for iter num {iter} of run_eval_loop')
        # run evals on df
        df = pe.call_evals(df, evals_config, id_col)
        df = pe.get_failed_evals(df)
        proj_logger.info(f'Evaluations completed for iter num {iter} of run_eval_loop')
        # for reqs where all evals passed, pop off these rows into revised_df
        if (df is not None):
            popped_cond = df[failed_eval_col].str.len() == 0
            popped_rows = df[popped_cond]#.index
            if len(popped_rows) > 0:
                proj_logger.info(f'Requirements were found that passed all criteria during iter num {iter} of run_eval_loop')
                pd_utils.to_excel(popped_rows, output_data_folder, str(iter), 'passed_reqs_df')
            proj_logger.info(f'The Requirements df was originally {len(df)} rows')
            df = df[~popped_cond]
            proj_logger.info(f'The Requirements df is now {len(df)} rows')
            if len(df) > 0:
                # run prompts for requirements containing failed evals
                df = run_prompts_for_failed_evals(df, runner, evals_config, failed_eval_col, id_col)
                df = pe.call_evals(df, evals_config, id_col)
                df = pe.get_failed_evals(df)
                pd_utils.to_excel(df, output_data_folder, str(iter), 'df')
            proj_logger.info(f'Exiting: iter num: {iter} of run_eval_loop')

reqs_df = pe.call_evals(reqs_df, evals_config, 'Requirement')
reqs_df = pe.get_failed_evals(reqs_df)

run_eval_loop(
    df=reqs_df,
    runner=ppr,
    evals_config=evals_config,
)