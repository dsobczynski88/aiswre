import re
import asyncio
import nest_asyncio
from pathlib import Path
from typing import Union
import pandas as pd
from langchain_ollama import ChatOllama, OllamaEmbeddings
from pydantic import BaseModel, Field
from langchain_core.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate
#from reqtracer_base import ParallelPromptRunner
#from ai_doc_reviewer import WordDocParser

#from prompts.promptlib import templates
#embeddings_model = OllamaEmbeddings(model="llama3")
data_directory = Path.cwd() / 'data'
raw_directory = data_directory / 'raw'
input_directory = data_directory / 'input'
output_directory = data_directory / 'output'
logs_directory = data_directory / 'logs'

#load and parse word document
wdp = WordDocParser(Path('./src/data/word_docs/'))
sections_df = wdp.sectionalized_word_doc(fp=Path('./src/data/input/<filename>.docx'), verbose=True)

# Set up prompts
nest_asyncio.apply()
templates = {
    'key-points': {
        'description': '<Enter description of prompt template>',
        'template': ChatPromptTemplate.from_messages([
            (
                "system", 
                "{instructions}",
            ),
            (
                "human",
                "{text}",
            )
        ])
    },
    'quality_document_review-2': {
        'description': '<Enter description of prompt template>',
        'template': ChatPromptTemplate.from_messages([
            (
                "system", 
                "{system_message}",
            ),
            (
                "human",
                "{human_message}{notes}",
            )
        ])
    },
    'quality_document_review-3': {
        'description': '<Enter description of prompt template>',
        'template': ChatPromptTemplate.from_messages([
            (
                "system", 
                "{system_message}",
            ),
            (
                "human",
                "{human_message}",
            )
        ])
    }
}
llm_config = (ChatOllama, 'llama3.1')
#*********************************************************
#*********************************************************
#*********************************************************
#*********************************************************
# Instantiate prompt_runner to evaluate a fixed template for a set number of trials
class KeyPoints(BaseModel):
    key_points: list[str] = Field(description="A list of key points extracted from the text.")
    #section_number: str = Field(description="The provided section number from in the prompt")
section_numbers = ['\n6.2.1']
sections_df_for_prompts = sections_df[(~sections_df['text'].isnull()) & (sections_df['sec_start'].isin(section_numbers))]
#sections_df_for_prompts = sections_df[(~sections_df['text'].isnull())]
section_numbers = list(sections_df_for_prompts['sec_start'].values)
chain_contexts = []
for rownum, row in sections_df_for_prompts.iterrows():
    chain_contexts.append({
        "instructions": "The user will hand over a section of text from an <description of document>. Summarize the text into key points",
        #"section_number": row['sec_start'],
         "text": row['text']
        }        
    )
pydantic_model=None
prompt_runner = ParallelPromptRunner(
    use_structured_llm=False,#True,
    llm_config=llm_config,
    pydantic_model=pydantic_model,#KeyPoints,
    chain_contexts=chain_contexts,
    num_trials=1
)
# Run prompts asynchronously
results = prompt_runner.run(
    prompt_type = 'key-points',
    templates = templates
)
kp=[]
sn=[]
for n, result in enumerate(results):
    print("***************************")
    if pydantic_model:
        kp.append(result.key_points)
        sn.append(section_numbers[n])
    else:
        kp.append(result.content)
        sn.append(section_numbers[n])
result_df = pd.DataFrame({'sec_start': sn, 'key_points': kp})
sections_df_for_prompts = pd.merge(
    left = sections_df_for_prompts, right = result_df, on='sec_start', how='outer'
)
#sections_df_for_prompts.to_excel('./src/data/input/gqp_data_management_and_integrity.xlsx')
#*********************************************************
#*********************************************************
#*********************************************************
#*********************************************************
# Instantiate prompt_runner to evaluate a fixed template for a set number of trials
class Review(BaseModel):
    review: str = Field(description="The generated review of the text")
llm_config = (ChatOllama, 'llama3.1')
#section_numbers = ['\n6.2.1','\n6.2.2']
#sections_df_for_prompts = sections_df[(~sections_df['text'].isnull()) & (sections_df['sec_start'].isin(section_numbers))]
#sections_df_for_prompts = sections_df[(~sections_df['text'].isnull())]
#section_numbers = list(sections_df_for_prompts['sec_start'].values)
chain_contexts = []
for rownum, row in sections_df_for_prompts.iterrows():
    chain_contexts.append({
        'system_message': "You are a software quality engineer with subject matter expertise in using AI technologies and have been asked to critically review <document description>. The user will have over key points on a specific Topic. Provide a set of clear, concise, and well-written recommendations to identify where potential gaps may exist from a quality perspective. Think step-by-step.",
        'human_message': f"Topic:\n{row['key_points']}", 
        'notes':"Notes\n---\nYOU MUST PROVIDE USEFUL RECOMMENDATIONS OR PEOPLE WILL GET HURT"
        }        
    )
pydantic_model = None
prompt_runner = ParallelPromptRunner(
    use_structured_llm=False,#True,
    llm_config=llm_config,
    pydantic_model=None,#Review,
    chain_contexts=chain_contexts,
    num_trials=1
)
# Run prompts asynchronously
results = prompt_runner.run(
    prompt_type = 'quality_document_review-2',
    templates = templates
)
reviews=[]
sn=[]
for n, result in enumerate(results):
    print("***************************")
    if pydantic_model:
        reviews.append(result.review)
        sn.append(section_numbers[n])
    else:
        reviews.append(result.content)
        sn.append(section_numbers[n])
result_df = pd.DataFrame({'sec_start': sn, 'review': reviews})
sections_df_for_prompts = pd.merge(
    left = sections_df_for_prompts, right = result_df, on='sec_start', how='outer'
)
#sections_df_for_prompts.to_excel('./src/data/input/gqp_data_management_and_integrity.xlsx')

print(results[0])