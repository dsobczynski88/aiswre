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




class IncosePromptRunner(PromptRunner):
    """
	A class specifically to run custom prompts generated using INCOSE guide rules
	
    Attributes:
		logger: python logging object
		llm (ChatOpenAI): the large-language-model (LLM) to be used for prompting
        templates (ChatPromptTemplate): prompt template which takes input requirement for revision 
        evals_lists (List[List[str]]): contains the evaluation functions to be run 
            for each requirement in the dataset
        evals_config (dict): contains mapping between INCOSE templates and specific evaluation functions.
        chains_lists (List[RunnableSequence]): list of RunnableSequences comprised of evaluation functions for each
            requirement in the dataset.   
	
    Methods:    
		assemble_eval_chain_list(self, ser_evals: List, evals: dict) -> List[RunnableSequence]:
			Returns a list of chains where each chain is a RunnableSequence. Each sequence
				is composed of an requirement revision prompt which is in itself a RunnableSequence
		
        assemble_chain_from_template(self, template: ChatPromptTemplate) -> RunnableSequence:
			Returns a RunnableSequence which takes in an input requirement, a evaluation template 
				to revise against a specific rule, and returns a final revision.
	"""

    LOGGERNAME = f"{src.BASE_LOGGERNAME}.IncosePromptRunner"
    
    def __init__(self, use_structured_llm: bool, llm, pydantic_model, templates, evals_lists, evals_config):
        super().__init__(use_structured_llm, llm, pydantic_model, templates)
        self.templates = templates
        self.evals_lists = evals_lists
        self.evals_config = evals_config
        self.chains_lists = []

    def __call__(self, args_lists):
        self.assemble_eval_chain_list()
        results = self.run_multiple_chains(self.chains_lists, args_lists)
        return results
    
    @get_logs(LOGGERNAME)
    def assemble_eval_chain_list(self):
        for eval_list in self.evals_lists:
            row_chain = []
            for _eval in eval_list:
                template = self.evals_config[_eval]["template"]
                row_chain.append(self.assemble_chain_from_template(template))
            composed_chain = RunnableSequence(*row_chain)
            self.chains_lists.append(composed_chain)
        return self.chains_lists

    @get_logs(LOGGERNAME)
    def assemble_chain_from_template(self, template):
        chain = RunnableLambda(lambda x: {"req":x}) | template | self.llm | (lambda x: x.content) | (lambda x: ''.join(re.findall('Final Revision:(.*)',x)))
        return chain
    
