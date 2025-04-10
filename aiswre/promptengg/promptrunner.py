import re
from typing import List, Union, Callable, Tuple, Optional
from functools import reduce
#import time
import ast
import asyncio
from pathlib import Path
#from dotenv import dotenv_values
import pandas as pd
#from langchain_openai import ChatOpenAI
from tqdm import tqdm

class ParallelPromptRunner:

    def __init__(self, use_structured_llm: bool, llm, pydantic_model, chain_contexts: List[dict], num_trials=1):
        
        #self.llm_name = llm_config[1]
        #self.llm = llm_config[0](model=self.llm_name)
        self.llm = llm
        if use_structured_llm:
            #self.llm = self.llm.with_structured_output(pydantic_model, method="function_calling")
            self.llm = self.llm.with_structured_output(pydantic_model, method="json_schema")    
        self.chain_contexts = chain_contexts
        self.num_trials = num_trials
        self._data_directory = Path.cwd() / 'data'
        
    def show_directories(self):
        print(f"""
            data directory: {self._data_directory}
            """)

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