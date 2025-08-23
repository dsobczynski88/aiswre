"""
This module defines the PromptRunner class, which facilitates running 
prompts asynchronously using multiple chains of actions. It is designed 
to work with large-language models (LLMs) and follows a structured 
approach for input and output management.
"""

import logging
from typing import List, Dict, Any, Optional
import nest_asyncio
import asyncio
from tqdm.asyncio import tqdm_asyncio
from langchain_core.runnables import RunnableSequence

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

class PromptRunner:
    """
    A class to run LLM prompts asynchronously through processing chains.
    
    This class manages the execution of multiple prompt chains in parallel,
    handling structured output when needed.
    
    Attributes:
        llm: The large-language-model to be used for prompting
        logger: Logger instance for tracking execution
    """

    LOGGER_NAME = "projectlog.PromptRunner"

    def __init__(self, llm, pydantic_model=None, use_structured_output=False):
        """
        Initialize the PromptRunner with an LLM and optional structured output configuration.
        
        Args:
            llm: The language model to use for processing
            pydantic_model: Optional Pydantic model for structured output
            use_structured_output: Whether to use structured output formatting
        """
        self._logger = logging.getLogger(self.LOGGER_NAME)
        self.llm = llm
        
        if use_structured_output and pydantic_model:
            self.llm = self.llm.with_structured_output(
                pydantic_model, 
                method="json_schema"
            )

    @property
    def logger(self):
        """Access the logger instance."""
        return self._logger
    
    async def run_multiple_chains(
        self, 
        chains: List[RunnableSequence], 
        args_lists: List[Dict[str, Any]]
    ) -> List[Any]:
        """
        Execute multiple chains asynchronously with their respective inputs.
        
        Args:
            chains: List of RunnableSequence objects to execute
            args_lists: List of input dictionaries for each chain
            
        Returns:
            List of results from each chain execution
        
        Raises:
            ValueError: If the lengths of chains and args_lists don't match
        """
        if len(chains) != len(args_lists):
            raise ValueError("Number of chains must match number of argument lists")
            
        self.logger.info(f"Preparing to run {len(chains)} chains asynchronously")
        
        tasks = [
            self.run_chain(chain, args)
            for chain, args in zip(chains, args_lists)
        ]
        
        self.logger.info('Awaiting results...')
        results = await tqdm_asyncio.gather(*tasks)
        self.logger.info('Results fetched successfully')
        
        return results

    async def run_chain(
        self, 
        chain: RunnableSequence, 
        inputs: Optional[Dict[str, Any]]
    ) -> Any:
        """
        Execute a single chain asynchronously with the given inputs.
        
        Args:
            chain: The RunnableSequence to execute
            inputs: Dictionary of input variables for the chain
            
        Returns:
            The result of the chain execution
            
        Raises:
            ValueError: If inputs is None
        """
        if inputs is None:
            raise ValueError("Input arguments cannot be None")
            
        return await chain.ainvoke(inputs)