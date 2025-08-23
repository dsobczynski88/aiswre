"""
This module includes functions specifically designed to preprocess the INCOSE guide, build
requirement revision prompt templates, and run prompts for requirement revision.
"""
from typing import List, Callable, Union, Dict, Optional, Any
import re
from functools import partial
from pathlib import Path
import pandas as pd
from pprint import pformat
import asyncio
import nest_asyncio
from tqdm import tqdm
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
    RunnableSequence
)
import src
from src.prj_logger import get_logs
from src import utils
from src.utils import map_A_to_B
import src.components.prompteval as pe
from src.components.promptrunner import PromptRunner
from src.components.preprocess import TextPreprocessor, Sectionalize, BuildTemplates

nest_asyncio.apply()


def preprocess_incose_guide(
    input_path: Path, 
    output_path: Path, 
    start_page: int, 
    end_page: int, 
    regex: str,
    replace_tokens: List[str], 
    subpatterns: List[str], 
    replace_with: str
) -> pd.DataFrame:
    """
    Preprocess the INCOSE guide PDF to extract and structure rule information.
    
    Args:
        input_path: Path to the INCOSE guide PDF
        output_path: Directory to save processed files
        start_page: First page to process
        end_page: Last page to process
        regex: Regular expression pattern to identify sections
        replace_tokens: List of tokens to replace
        subpatterns: List of regex patterns to substitute
        replace_with: Replacement string for tokens and patterns
        
    Returns:
        DataFrame containing structured INCOSE guide rules
    """
    # Create base objects using composition
    sectionalize = Sectionalize(regex)
    preprocessor = TextPreprocessor()
    
    # Extract text from PDF
    text = sectionalize.get_pdf_text(input_path, start_page, end_page)
    
    # Save raw extracted text
    sectionalize.save_text(output_path / "extract.txt")
    
    # Set up preprocessing pipeline
    preprocessor.pipeline = [
        partial(TextPreprocessor.replace, replace_tokens=replace_tokens, replace_with=replace_with),
        partial(TextPreprocessor.resub, patterns=subpatterns, replace_with=replace_with),
        TextPreprocessor.remove_multi_whitespace,
    ]
    
    # Apply preprocessing to text
    sectionalize.text = preprocessor.clean_text(sectionalize.text)
    
    # Save cleaned text
    sectionalize.save_text(output_path / "extract-post-clean.txt")
    
    # Extract sections
    df = sectionalize.get_sections_df()
    df = sectionalize.add_section_text()
    
    # Extract rule information
    df = extract_incose_rule_info(df)
    
    return df


def extract_incose_rule_info(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract structured information from INCOSE guide sections.
    
    Args:
        df: DataFrame with raw section text
        
    Returns:
        DataFrame with extracted rule components
    """
    # Extract rule number
    df['rule_number'] = df['extract'].apply(
        lambda s: re.search(r'^ (R\d+) –', s, flags=re.DOTALL).group(1) if re.search(r'^ (R\d+) –', s, flags=re.DOTALL) else None
    )
    
    # Extract rule title
    df['rule_title'] = df['extract'].apply(
        lambda s: re.search(r'^ R\d+ – ([A-Z\W]+) Definition', s, flags=re.DOTALL).group(1) 
        if re.search(r'^ R\d+ – ([A-Z\W]+) Definition', s, flags=re.DOTALL) else None
    )
    
    # Extract definition
    df['definition'] = df['extract'].apply(
        lambda s: ''.join(re.findall(r'Definition:([\s\W\w]+)(?=Elaboration:)', s, flags=re.DOTALL))
    )
    
    # Extract elaboration
    df['elaboration'] = df['extract'].apply(
        lambda s: ''.join(re.findall(r'Elaboration:([\s\W\w]+)(?=Examples:)', s, flags=re.DOTALL))
    )
    
    # Extract examples
    df['examples'] = df['extract'].apply(
        lambda s: ''.join(re.findall(r'Examples:(.*)$', s, flags=re.DOTALL))
    )
    
    # Clean examples - remove bracketed text
    df['examples'] = df['examples'].apply(
        lambda s: re.sub(r'\[[^\]]+\]', '', s, flags=re.DOTALL)
    )
    
    # Clean examples - remove exceptions and relationships
    df['examples'] = df['examples'].apply(
        lambda s: re.sub(r'(Exceptions and relationships:.*)$', '', s, flags=re.DOTALL)
    )
    
    return df


def build_incose_templates(
    incose_df: pd.DataFrame, 
    base_messages: Dict[str, str], 
    output_folder_path: Path
) -> Dict[str, Any]:
    """
    Build prompt templates based on INCOSE guide rules.
    
    Args:
        incose_df: DataFrame containing INCOSE guide rules
        base_messages: Dictionary with base system and user message templates
        output_folder_path: Path to save output files
        
    Returns:
        Dictionary of templates keyed by rule number
    """
    # Create template builder
    template_builder = BuildTemplates(incose_df, base_messages)
    
    # Add message columns to DataFrame
    template_builder.add_message_col_to_frame("system")
    template_builder.add_message_col_to_frame("user")
    
    # Replace template variables with INCOSE data
    template_builder.replace_prompt_variables_from_frame("user_message", "definition")
    template_builder.replace_prompt_variables_from_frame("user_message", "examples")
    
    # Build templates
    templates = template_builder.assemble_templates_from_df(
        system_message_col='system_message',
        user_message_col='user_message',
        template_name_prefix='R'
    )
    
    return templates


def build_incose_eval_config(
    incose_df: pd.DataFrame, 
    output_folder_path: Path, 
    templates: Dict[str, Any], 
    rule_to_eval_map: Dict[str, str],
    rule_num_col: str = 'rule_number'
) -> Dict[str, Dict]:
    """
    Create mapping between evaluation functions and INCOSE rules.
    
    Args:
        incose_df: DataFrame containing INCOSE guide rules
        output_folder_path: Path to save output files
        templates: Dictionary of templates keyed by rule number
        rule_to_eval_map: Mapping from rule IDs to evaluation function names
        rule_num_col: Column name containing rule numbers
        
    Returns:
        Dictionary mapping evaluation functions to templates and functions
    """
    # Create evaluation config
    evals_config = {}
    
    # Map evaluation functions to templates
    for rule, eval_name in rule_to_eval_map.items():
        evals_config[eval_name] = {
            "func": getattr(pe, eval_name),
            "template": templates[rule]
        }
    
    # Save config to file
    with open(Path(output_folder_path) / "evals_config.txt", "w") as f:
        f.write(pformat(evals_config))
    
    return evals_config


class IncoseRequirementReviewer:
    """
    A class to run custom prompts generated using INCOSE guide rules.
    
    This class uses composition to leverage the PromptRunner functionality.
    """
    LOGGER_NAME = f"{src.BASE_LOGGERNAME}.IncoseRequirementReviewer"
        
    def __init__(
        self, 
        use_structured_llm: bool, 
        llm, 
        pydantic_model, 
        templates: Dict[str, Any], 
        evals_config: Dict[str, Dict], 
        id_col: str = 'Requirement'
    ):
        """
        Initialize the IncoseRequirementReviewer.
        
        Args:
            use_structured_llm: Whether to use structured output from LLM
            llm: Language model to use
            pydantic_model: Model for structured output
            templates: Dictionary of templates
            evals_config: Configuration mapping evaluations to templates
            id_col: Column name for requirement IDs
        """
        # Use composition instead of inheritance
        self._prompt_runner = PromptRunner(use_structured_llm, llm, pydantic_model)
        self.templates = templates
        self.evals_config = evals_config
        self.evals_chains = []
        self.id_col = id_col
        self._logger = logging.getLogger(self.LOGGER_NAME)

    def revise(
        self, 
        evals_lists: List[List[str]], 
        args_lists: List[str], 
        capture_func: Optional[Callable] = None
    ) -> pd.DataFrame:
        """
        Revise requirements based on evaluation results.
        
        Args:
            evals_lists: Lists of evaluation functions to run for each requirement
            args_lists: Lists of requirement statements
            capture_func: Optional function to extract specific output from LLM response
            
        Returns:
            DataFrame with revision results
        """
        self.assemble_eval_chain_list(evals_lists, capture_func)
        results = asyncio.run(self._prompt_runner.run_multiple_chains(self.evals_chains, args_lists))
        results_df = self.cast_results_to_frame(results)
        return results_df
    
    def assemble_eval_chain_list(
        self, 
        evals_lists: List[List[str]], 
        capture_func: Optional[Callable] = None
    ) -> List[RunnableSequence]:
        """
        Assemble chains of evaluation functions for each requirement.
        
        Args:
            evals_lists: Lists of evaluation functions to run for each requirement
            capture_func: Optional function to extract specific output from LLM response
            
        Returns:
            List of runnable sequences
        """
        self.evals_chains = []
        
        for eval_list in evals_lists:
            self._logger.info(f"Assembling chain for evaluations: {eval_list}")
            row_chain = []
            
            for eval_name in eval_list:
                template = self.evals_config[eval_name]["template"]
                row_chain.append(self.assemble_chain_from_template(template, capture_func))
            
            composed_chain = RunnableSequence(*row_chain)
            self.evals_chains.append(composed_chain)
            
        return self.evals_chains

    def assemble_chain_from_template(
        self, 
        template, 
        capture_func: Optional[Callable] = None
    ) -> RunnableSequence:
        """
        Create a chain from a template.
        
        Args:
            template: Prompt template
            capture_func: Optional function to extract specific output from LLM response
            
        Returns:
            Runnable sequence
        """
        if capture_func is not None:
            chain = (
                RunnableLambda(lambda x: {"req": x}) | 
                template | 
                self._prompt_runner.llm | 
                (lambda x: x.content) | 
                capture_func
            )
        else:
            chain = (
                RunnableLambda(lambda x: {"req": x}) | 
                template | 
                self._prompt_runner.llm | 
                (lambda x: x.content)
            )
        return chain
    
    def cast_results_to_frame(self, results: List[str]) -> pd.DataFrame:
        """
        Convert LLM results to a DataFrame.
        
        Args:
            results: Results returned by the LLM
            
        Returns:
            DataFrame with results
        """
        results_df = pd.DataFrame(results)
        results_df = results_df.rename(columns={0: self.id_col})
        return results_df
    
    def run_eval_sequence(
        self, 
        df: pd.DataFrame, 
        col: str, 
        failed_eval_col: str, 
        col_suffix: Optional[str] = None, 
        eval_to_rule_map: Optional[Dict[str, str]] = None
    ) -> pd.DataFrame:
        """
        Run evaluation sequence on requirements in a DataFrame.
        
        Args:
            df: DataFrame containing requirements
            col: Column containing requirement text
            failed_eval_col: Column to store failed evaluations
            col_suffix: Optional suffix for output columns
            eval_to_rule_map: Optional mapping from evaluations to rules
            
        Returns:
            DataFrame with evaluation results
        """
        # Call evaluations
        df = self.call_evals(df, col)
        
        # Get failed evaluations
        df = self.get_failed_evals(df)
        
        # Map failed evaluations to rule IDs if mapping provided
        if eval_to_rule_map is not None:
            df = self.map_failed_evals_to_rule_ids(df, eval_to_rule_map)
        
        # Drop intermediate evaluation columns
        df = df.drop(columns=[c for c in df.columns if c.startswith('eval')])
        
        # Add suffix to output columns if provided
        if col_suffix is not None:
            failed_eval_cols = [c for c in df.columns if c.startswith('failed_evals')]
            
            for c in failed_eval_cols:
                # Rename column with suffix
                df.rename(columns={c: f"{col_suffix}_{c}"}, inplace=True)
                
                # Add resolution indicator column
                df[f"if_resolved_{col_suffix}_{c}"] = df[f"{col_suffix}_{c}"].apply(
                    lambda _l: 1 - int(bool(len(_l)))
                )
                
                # Add resolution percentage column
                df[f"%_resolved_{col_suffix}_{c}"] = round(
                    sum(df[f"if_resolved_{col_suffix}_{c}"]) / 
                    len(df[f"if_resolved_{col_suffix}_{c}"]) * 100, 
                    0
                )
        
        return df
    
    def call_evals(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """
        Run evaluations for each row in the DataFrame.
        
        Args:
            df: DataFrame containing requirements
            col: Column containing requirement text
            
        Returns:
            DataFrame with evaluation results
        """
        result_df = df.copy()
        
        # Run evaluations for each row
        for idx, row in result_df.iterrows():
            for eval_name, eval_config in self.evals_config.items():
                eval_func = eval_config["func"]
                eval_result = eval_func(row[col])
                result_df.loc[idx, eval_name] = pe.convert_bool_to_ohe(eval_result)
        
        return result_df
    
    def get_failed_evals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify failed evaluations for each requirement.
        
        Args:
            df: DataFrame with evaluation results
            
        Returns:
            DataFrame with added 'failed_evals' column
        """
        result_df = df.copy()
        eval_cols = [c for c in result_df.columns if c.startswith("eval")]
        
        result_df['failed_evals'] = result_df[eval_cols].apply(
            lambda row: [eval_cols[i] for i, val in enumerate(row) if val == 1.0], 
            axis=1
        )
        
        return result_df

    def map_failed_evals_to_rule_ids(
        self, 
        df: pd.DataFrame, 
        eval_to_rule_map: Dict[str, str]
    ) -> pd.DataFrame:
        """
        Map failed evaluations to rule IDs.
        
        Args:
            df: DataFrame with failed evaluations
            eval_to_rule_map: Mapping from evaluation names to rule IDs
            
        Returns:
            DataFrame with added 'failed_evals_rule_ids' column
        """
        result_df = df.copy()
        
        result_df['failed_evals_rule_ids'] = result_df['failed_evals'].apply(
            lambda eval_list: map_A_to_B(eval_list, eval_to_rule_map) if isinstance(eval_list, list) else None
        )
        
        return result_df