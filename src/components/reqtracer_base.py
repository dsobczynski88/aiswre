import re
from typing import List, Union, Callable, Tuple, Optional
from functools import reduce
#import time
import ast
import asyncio
from pathlib import Path
#from dotenv import dotenv_values
import pandas as pd
from spire.doc import *
from spire.doc.common import *
from prompts.promptlib import templates
#from langchain_openai import ChatOpenAI
from preprocess import load_csv, ProcessedGuide
from utils import write_string, mk_dict_from_df
from generate_html import df_to_html
from tqdm import tqdm

class ParallelPromptRunner:

    def __init__(self, use_structured_llm: bool, llm_config: Tuple[Callable, str], pydantic_model: Optional, chain_contexts: List[dict], num_trials=1):
        
        self.llm_name = llm_config[1]
        self.llm = llm_config[0](model=self.llm_name)
        if use_structured_llm:
            #self.llm = self.llm.with_structured_output(pydantic_model, method="function_calling")
            self.llm = self.llm.with_structured_output(pydantic_model, method="json_schema")    
        self.chain_contexts = chain_contexts
        self.num_trials = num_trials
        self._data_directory = Path.cwd() / 'data'
        self._raw_directory = self._data_directory / 'raw'
        self._input_directory = self._data_directory / 'input'
        self._output_directory = self._data_directory / 'output'
        self._logs_directory = self._data_directory / 'logs'

    def show_directories(self):
        print(f'''
            data directory: {self._data_directory}
            raw directory : {self._raw_directory}
            input directory: {self._input_directory}
            output directory: {self._output_directory}
            logs directory: {self._logs_directory}
            '''
            )

    def run(self, prompt_type, templates):
        chain_outputs = self.assemble_chains(prompt_type, templates)
        if self.chain_contexts is not None:
            async_tasks = self.run_multiple_chains(chain_outputs[0], chain_outputs[1])
        else:
            async_tasks = self.run_multiple_chains(chain_outputs, None)
        return asyncio.run(async_tasks)
    
    def assemble_chains(self, prompt_type, templates):       
        template = templates[prompt_type]['template']
        chain = template | self.llm
        if self.chain_contexts is not None:
            return (chain, self.chain_contexts)
        else:
            return chain

    def assemble_faiss_chains(self, prompt_type, templates):
        pass
        
    @staticmethod
    async def run_chain(chain, inputs):
        if inputs is not None:
            return await chain.ainvoke(inputs)
        else:
            return await chain.ainvoke()

    async def run_multiple_chains(self, chain, _args):
        tasks=[]
        if _args is not None:
            for args in _args:
                tasks.append(__class__.run_chain(chain, args))
        else:
            tasks.append(__class__.run_chain(chain, None))
        print('Awaiting results...')
        results = await asyncio.gather(*tasks)
        print('Results fetched...')
        return results

    def process_responses(self, df, response_col, method):
        if method == 'json_schema':
            df[response_col] = df[response_col].apply(lambda s: re.findall(r'testcases=\[(.+)\]', str(s), flags=re.DOTALL))
            df[response_col] = df[response_col].apply(lambda s: ast.literal_eval(str(s)))
            df = df.explode(response_col)
            df[response_col] = df[response_col].apply(lambda s: ast.literal_eval(str(s)))
            df = df.explode(response_col)
            df = df.reset_index(drop=True)
            expand_df = df[response_col].apply(pd.Series)
            df = pd.concat([df, expand_df], axis=1)
            return df
    
    @staticmethod
    def create_comparisons(df1,df2,df1_cols, df2_cols, compare_func):
        df2 = df2[df2_cols].drop_duplicates()
        df2_dict = mk_dict_from_df(df2, df2_cols)
        df1_dict = mk_dict_from_df(df1, df1_cols)
        list_of_dict=[]
        for name2, des2 in df2_dict.items():
            compare_dict = {}
            for name1, des1 in df1_dict.items():
                compare_dict[name1] = compare_func([des2, des1])
            list_of_dict.append(compare_dict)
        df2.reset_index(drop=True, inplace=True)
        df2[f'result_{compare_func.__name__}'] = pd.Series(list_of_dict)
        expand_df = df2[f'result_{compare_func.__name__}'].apply(pd.Series)
        df2.rename(columns = {f'result_{compare_func.__name__}':'result_dict'}, inplace=True)
        df2 = pd.concat([df2, expand_df], axis=1)
        return df2