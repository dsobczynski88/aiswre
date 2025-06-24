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

    def __init__(self, use_structured_llm: bool, llm, pydantic_model):
        
        #self.llm_name = llm_config[1]
        #self.llm = llm_config[0](model=self.llm_name)
        self._logger = logging.getLogger(ParallelPromptRunner.LOGGERNAME)
        self.llm = llm
        if use_structured_llm:
            #self.llm = self.llm.with_structured_output(pydantic_model, method="function_calling")
            self.llm = self.llm.with_structured_output(pydantic_model, method="json_schema")    
        self._data_directory = Path.cwd() / 'data'

    @property
    def logger(self):
        return self._logger
    
    def show_directories(self):
        print(f"""
            data directory: {self._data_directory}
            """)
    
    @get_logs(LOGGERNAME)
    def assemble_chain_from_template(self, template):
        chain = RunnableLambda(lambda x: {"req":x}) | template | self.llm | (lambda x: x.content) | (lambda x: ''.join(re.findall('Final Revision:(.*)',x)))
        return chain

    @get_logs(LOGGERNAME)
    def assemble_eval_chain_list(self, ser_evals, evals):
        ser_chains=[]
        for list_eval in ser_evals:
            row_chain = []
            for eval_func in list_eval:
                template = evals[eval_func]["template"]
                row_chain.append(self.assemble_chain_from_template(template))
            composed_chain = RunnableSequence(*row_chain)
            ser_chains.append(composed_chain)
        return ser_chains

    async def run_chain(self, chain, inputs):
        if inputs is not None:
            return await chain.ainvoke(inputs)
        else:
            raise ValueError
    
    @get_logs(LOGGERNAME)
    async def run_multiple_chains(self, chains, _args):
        tasks=[]
        _inputs = zip(chains, _args)
        for input in _inputs:
            chain = input[0]
            _arg = input[1]
            if _arg is not None:
                tasks.append(self.run_chain(chain, _arg))
            else:
                tasks.append(self.run_chain(chain, None))
        self._logger.info('awaiting results...')
        results = await asyncio.gather(*tasks)
        self._logger.info('results fetched...')
        return results