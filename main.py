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

# load requirements dataset
reqs_df = pd.read_excel('./aiswre/data/software_requirements.xlsx')

# load eval func to incose rules map (associations)
prompt_associations = pe.load_prompt_associations()
prompt_templates_config = prompt_utils.assemble_prompt_templates_from_df(incose_guide_sections_df, system_message_colname='system_message', user_message_colname='user_message')

# run evaluations on requirements dataset
pe.call_evals(reqs_df, pe.get_eval_funcs(prompt_associations), 'Requirement')
pe.get_failed_evals(reqs_df)
pd_utils.to_excel(reqs_df, output_data_folder, False, 'reqs_df_evaluation')  
print(reqs_df.head(5))

reqs_df_filt = reqs_df.loc[reqs_df['failed_evals'].str.len() > 0].head(3) 

# load evaluations configuration
evals_config = {
    'eval_is_in_passive_voice':{
        'func':pe.eval_is_in_passive_voice,
        'template': prompt_templates_config['R2']
    },
    'eval_if_vague_verb':{
        'func': pe.eval_if_vague_verb,
        'template': prompt_templates_config['R3']
    },
    'eval_has_a_def_article':{
        'func': pe.eval_has_a_def_article,
        'template': prompt_templates_config['R5']
    },      
    'eval_has_vague_terms': {
        'func': pe.eval_has_vague_terms,
        'template': prompt_templates_config['R7']
    },
    'eval_has_escape_clause':{
        'func':pe.eval_has_escape_clause,
        'template': prompt_templates_config['R8']
    }
}

# instantiate openai client
secret_key = config['OPENAI_API_KEY']
client = ChatOpenAI(api_key=secret_key, model='gpt-4o-mini')

def assemble_chain_from_template(template, llm):
    chain = RunnableLambda(lambda x: {"req":x}) | template | llm | (lambda x: x.content) | (lambda x: ''.join(re.findall('Final Revision:(.*)',x)))
    return chain

def assemble_eval_chain_list(ids, evals, llm):
    chains=[]
    for _id in ids:
        template = evals[_id]["template"]
        chains.append(assemble_chain_from_template(template, llm))
    composed_chain = RunnableSequence(*chains)
    return composed_chain

async def run_chain(chain, inputs):
    if inputs is not None:
        return await chain.ainvoke(inputs)
    else:
        raise ValueError

async def run_multiple_chains(chains, _args):
    tasks=[]
    if _args is not None:
        for i, args in enumerate(_args):
            tasks.append(run_chain(chains[i], args))
    print('awaiting results...')
    results = await asyncio.gather(*tasks)
    print('results fetched...')
    return results


# get failed evals for all requirements
failed_evals = reqs_df_filt['failed_evals'].values

class Requirement:
    revision: str = Field("A revised software requirement")

pydantic_model = Requirement
#llm = client.with_structured_output(pydantic_model, method="json_schema")
llm = client

chains=[]
for fe in failed_evals:
    chains.append(assemble_eval_chain_list(fe, evals_config, llm))

_args=reqs_df_filt['Requirement'].values

print(chains)
print(_args)

#result = chains[0].invoke(_args[0])
#print(result)

print(len(chains))
print(len(_args))

LOAD_DATA=True

if not LOAD_DATA:
    async_tasks = run_multiple_chains(chains, _args)
    results = asyncio.run(async_tasks)
    results_df = pd.DataFrame(results)
    results_df.rename(columns={0:'Requirement'}, inplace=True)
    pd_utils.to_excel(reqs_df_filt, output_data_folder, False, 'reqs_df_filt')
    pd_utils.to_excel(results_df, output_data_folder, False, 'results_df')

results_df = pd.read_excel(f"{output_data_folder}/results_df.xlsx")
try:
    results_df.rename(columns={0:'Requirement'}, inplace=True)
except Exception as e:
    print(results_df.columns)
# run evaluations on requirements dataset
pe.call_evals(results_df, eval_funcs, 'Requirement')
pe.get_failed_evals(results_df)
pd_utils.to_excel(results_df, output_data_folder, 1, 'results_df_evaluation')  
print(results_df.head(5))

'''
reqs_df = pd.DataFrame({'req': results})
pe.call_evals(reqs_df, eval_funcs, 'req')
# define prompts to revise requirements not meeting criteria
pe.get_failed_evals(reqs_df)  
print(reqs_df.head(5))
'''



# build chains for each requirement (based on failed evaluations)

'''
evals_config = {
    'eval_is_in_passive_voice':{
        'func':pe.eval_is_in_passive_voice,
        'template': ChatPromptTemplate.from_template("Revise the requirement to avoid use of passive voice: {req}")
    },
    'eval_if_vague_verb':{
        'func': pe.eval_if_vague_verb,
        'template': ChatPromptTemplate.from_template("Revise the requirement to avoid use of vague verbs: {req}")
    },
        'eval_has_a_def_article':{
            'func': pe.eval_has_a_def_article,
            'template': ChatPromptTemplate.from_template("Revise the requirement to avoid use of definite articles like \"a\": {req}")
    },      
    'eval_has_vague_terms': {
        'func': pe.eval_has_vague_terms,
        'template': ChatPromptTemplate.from_template("Revise the requirement to avoid use of vague terms: {req}")
    },
    'eval_has_escape_clause':{
        'func':pe.eval_has_escape_clause,
        'template': ChatPromptTemplate.from_template("Revise the requirement to avoid use of escape clauses: {req}")
    }
}
'''