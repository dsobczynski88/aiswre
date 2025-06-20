import re
import logging
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
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
    RunnableSequence
)
from aiswre.prj_logger import get_logs

class ParallelPromptRunner:

    BASE_LOGGERNAME = "reviewer"
    LOGGERNAME = f"{BASE_LOGGERNAME}.promptrunner"

    def __init__(self, use_structured_llm: bool, llm, pydantic_model, chain_contexts: List[dict], num_trials=1):
        
        #self.llm_name = llm_config[1]
        #self.llm = llm_config[0](model=self.llm_name)
        self._logger = logging.getLogger(ParallelPromptRunner.LOGGERNAME)
        self.llm = llm
        if use_structured_llm:
            #self.llm = self.llm.with_structured_output(pydantic_model, method="function_calling")
            self.llm = self.llm.with_structured_output(pydantic_model, method="json_schema")    
        self.chain_contexts = chain_contexts
        self.num_trials = num_trials
        self._data_directory = Path.cwd() / 'data'

    @property
    def logger(self):
        return self._logger
    
    def show_directories(self):
        print(f"""
            data directory: {self._data_directory}
            """)

    @get_logs(LOGGERNAME)
    def run(self, prompt_type, templates):
        chain_outputs = self.assemble_chains(prompt_type, templates)
        if self.chain_contexts is not None:
            async_tasks = self.run_multiple_chains(chain_outputs[0], chain_outputs[1])
        else:
            async_tasks = self.run_multiple_chains(chain_outputs, None)
        return asyncio.run(async_tasks)
    
    @get_logs(LOGGERNAME)
    def assemble_chains(self, prompt_type, templates):       
        template = templates[prompt_type]['template']
        chain = template | self.llm
        if self.chain_contexts is not None:
            return (chain, self.chain_contexts)
        else:
            return chain
     
    @get_logs(LOGGERNAME)
    def assemble_chain_from_template(template, llm):
        chain = RunnableLambda(lambda x: {"req":x}) | template | llm | (lambda x: x.content) | (lambda x: ''.join(re.findall('Final Revision:(.*)',x)))
        return chain

    @get_logs(LOGGERNAME)
    def assemble_eval_chain_list(self, ids, evals, llm):
        chains=[]
        for _id in ids:
            template = evals[_id]["template"]
            chains.append(self.assemble_chain_from_template(template, llm))
        composed_chain = RunnableSequence(*chains)
        return composed_chain

    async def run_chain(chain, inputs):
        if inputs is not None:
            return await chain.ainvoke(inputs)
        else:
            raise ValueError

    async def run_multiple_chains(self, chains, _args):
        tasks=[]
        if _args is not None:
            for i, args in enumerate(_args):
                tasks.append(self.run_chain(chains[i], args))
        print('awaiting results...')
        results = await asyncio.gather(*tasks)
        print('results fetched...')
        return results
    
    async def run_chain(self, chain, inputs):
        if inputs is not None:
            return await chain.ainvoke(inputs)
        else:
            return await chain.ainvoke()
    
    @get_logs(LOGGERNAME)
    async def run_multiple_chains(self, chain, _args):
        tasks=[]
        if _args is not None:
            for args in _args:
                tasks.append(self.run_chain(chain, args))
        else:
            tasks.append(self.run_chain(chain, None))
        self._logger.info('awaiting results...')
        results = await asyncio.gather(*tasks)
        self._logger.info('results fetched...')
        return results