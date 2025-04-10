import re
import asyncio
import nest_asyncio
from pathlib import Path
from typing import Union
import pandas as pd
#from langchain_ollama import ChatOllama, OllamaEmbeddings
from pydantic import BaseModel, Field
from langchain_core.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate
from aiswre.utils import pd_utils

#from prompts.promptlib import templates
#embeddings_model = OllamaEmbeddings(model="llama3")
#data_directory = Path.cwd() / 'data'
#raw_directory = data_directory / 'raw'
#input_directory = data_directory / 'input'
#output_directory = data_directory / 'output'
#logs_directory = data_directory / 'logs'

#load and parse word document

# Set up prompts
def assemble_prompt_template(row: dict) -> ChatPromptTemplate:

    return ChatPromptTemplate.from_messages([
            (
                "system", 
                row["system"],
            ),
            (
                "human",
                row["user"],
            )
        ])

def assemble_prompt_str(_template: ChatPromptTemplate) -> str:
    prompt_str = ''
    for message in _template.messages:
        #print(f"Message type: {type(message).__name__}")
        prompt_str += message.prompt.template + "\n" 
    return prompt_str

def check_for_prompt_vars(_template: ChatPromptTemplate) -> set:
    prompt_vars = set()
    for message in _template.messages:
        #print(f"Message type: {type(message).__name__}")
        prompt_vars.update(message.prompt.input_variables)
    return prompt_vars

def assemble_prompts_from_df(df):
    prompt_templates = []
    prompt_args_list = []
    prompt_args = {}
    for index, row in df.iterrows():
        _template = assemble_prompt_template(row)
        prompt_templates.append(_template)
        df.loc[index, 'prompt_template'] = assemble_prompt_str(_template)    
        # define template args
        prompt_vars = check_for_prompt_vars(_template)
        for var in prompt_vars:
            if f"var_{var}" not in df.columns:
                #log error
                df[f"var_{var}"] = ''
                df.to_csv("./aiswre/templates/templates.csv")
                raise KeyError(f'The column: var_{var} was not found. It has been added to the dataframe')
            else:
                prompt_args[var] = row[f"var_{var}"]
        prompt_args_list.append(prompt_args)
        # invoke template
        _prompt = _template.invoke(prompt_args)
        df.loc[index, 'prompt_value'] = _prompt.to_string()
    df.drop(columns=[c for c in df if "Unnamed: " in c], inplace=True)
    df.to_csv("./aiswre/templates/templates_invoked.csv")
    return prompt_templates, prompt_args_list

templates = [
    {
        "name": 'prompt-template-1',
        "description": 'description of prompt-template-1',
        "system": 'system message with {instructions} for prompt-template-1',
        "user": 'user message with {instructions} for prompt-template-1',
    },
    {
        "name": 'prompt-template-2',
        "description": 'description of prompt-template-2',
        "system": 'system message with {instructions} for prompt-template-2',
        "user": 'user message with {instructions} for prompt-template-2',
    }
]

#templates_df = pd.DataFrame(templates)
#templates_df['prompt_template'] = templates_df[['system','user']].apply(lambda s: "\n".join(*s), axis=1)
#print(templates_df)
#templates_df.to_csv("./aiswre/templates/templates.csv")

templates_df = pd.read_csv("./aiswre/templates/templates.csv")

for c in ['prompt_value','prompt_template']:
    if c in templates_df.columns:
        templates_df[c] = ''
    else:
        templates_df.insert(0, c, '')

prompt_templates, prompt_args_list = assemble_prompts_from_df(templates_df)

#print(prompt_templates)
print(prompt_args_list)
# define function here to populate vars...

