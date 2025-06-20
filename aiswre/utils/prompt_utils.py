import pandas as pd
from tqdm import tqdm
from langchain_core.prompts.chat import ChatPromptTemplate

# build templates
def assemble_prompt_template_from_messages(system_message: str, user_message: str) -> ChatPromptTemplate:
    '''Create a ChatPromptTemplate given an input dictionary containing keys: system, user'''
    return ChatPromptTemplate.from_messages([
            ("system",system_message),
            ("human", user_message)
        ])
        
def assemble_prompt_templates_from_df(df, system_message_colname='system_message', user_message_colname='user_message'):
    '''Loop over dataframe to build a unique prompt template for each row'''
    prompt_templates_config = {}
    for index, row in tqdm(df.iterrows()):
        system_message=row[system_message_colname]
        user_message=row[user_message_colname]
        try:
            prompt_templates_config[f"R{index+1}"] = assemble_prompt_template_from_messages(system_message, user_message)
        except ValueError:
            system_message=system_message.replace(".} ",".]")
            user_message=user_message.replace(".} ",".]")
            prompt_templates_config[f"R{index+1}"] = assemble_prompt_template_from_messages(system_message, user_message)
    return prompt_templates_config
    

def add_spaces(x: str) -> str:
    #return f" {x} "
    return x

def convert_bool_to_ohe(bool_result: bool) -> int:
    if bool_result:
        return 1
    else:
        return 0