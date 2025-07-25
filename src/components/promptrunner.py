"""
This module defines the PromptRunner class, which facilitates running 
prompts asynchronously using multiple chains of actions. It is designed 
to work with large-language models (LLMs) and follows a structured 
approach for input and output management.

Classes:
    PromptRunner: Manages the execution of multiple prompt chains 
    asynchronously, utilizing a given LLM for processing.

Methods:
    run_multiple_chains(chains: List[str], args_lists: dict) -> List[str]:
        Executes multiple chains, represented by RunnableSequence objects, 
        asynchronously using inputs derived from the provided arguments 
        list and returns the results.

    run_chain(chain: RunnableSequence, inputs: dict) -> dict:
        Invokes a single chain asynchronously with the given prompt input 
        variables, returning the asynchronous task result.

    assemble_eval_chain_list(ser_evals: List, evals: dict) -> List[RunnableSequence]:
        Constructs a list of RunnableSequences, where each sequence 
        integrates a requirement revision prompt and the evaluations specified.

    assemble_chain_from_template(template: ChatPromptTemplate) -> RunnableSequence:
        Generates a RunnableSequence that processes an input requirement 
        against an evaluation template, revises it according to a set rule, 
        and provides the final revised output.
"""


import re
import logging
from typing import List, Union, Callable, Tuple, Optional
import asyncio
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
    RunnableSequence
)
import src
from src.prj_logger import get_logs


class PromptRunner:
    """
	A class to run prompts asychronously
	
    Attributes:
		logger: python logging object
		llm (ChatOpenAI): the large-language-model (LLM) to be used for prompting
	
    Methods:
		run_multiple_chains(self, chains: List[str], args_lists: dict) -> List[str]:
			Returns a list of responses from LLM prompts via method self.run_chain. 
				The length of the results is equal to the number of input chains.
		
        run_chain(self, chain: RunnableSequence, inputs: dict) -> dict:
			Returns an asynchronous task via langchain's asynchronous method .ainvoke
				 for a given chain and prompt input variables (inputs).
	    
		assemble_eval_chain_list(self, ser_evals: List, evals: dict) -> List[RunnableSequence]:
			Returns a list of chains where each chain is a RunnableSequence. Each sequence
				is composed of an requirement revision prompt which is in itself a RunnableSequence
		
        assemble_chain_from_template(self, template: ChatPromptTemplate) -> RunnableSequence:
			Returns a RunnableSequence which takes in an input requirement, a evaluation template 
				to revise against a specific rule, and returns a final revision.
	"""

    LOGGERNAME = f"{src.BASE_LOGGERNAME}.PromptRunner"

    def __init__(self, use_structured_llm: bool, llm, pydantic_model):
        
        self._logger = logging.getLogger(PromptRunner.LOGGERNAME)
        self.llm = llm
        if use_structured_llm:
            self.llm = self.llm.with_structured_output(pydantic_model, method="json_schema")    

    @property
    def logger(self):
        return self._logger
    
    @get_logs(LOGGERNAME)
    async def run_multiple_chains(self, chains, args_lists):
        tasks=[]
        _inputs = zip(chains, args_lists)
        for input in _inputs:
            chain = input[0]
            arg_list = input[1]
            if arg_list is not None:
                tasks.append(self.run_chain(chain, arg_list))
            else:
                tasks.append(self.run_chain(chain, None))
        self._logger.info('awaiting results...')
        results = await asyncio.gather(*tasks)
        self._logger.info('results fetched...')
        return results
    
    @get_logs(LOGGERNAME)
    async def run_chain(self, chain, arg_list):
        if arg_list is not None:
            return await chain.ainvoke(arg_list)
        else:
            raise ValueError