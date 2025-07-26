from typing import List, Callable, Union
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

class PreprocessIncoseGuide(TextPreprocessor, Sectionalize):

    LOGGERNAME = f"{src.BASE_LOGGERNAME}.PreprocessIncoseGuide"

    def __init__(self, regex):
        Sectionalize.__init__(self, regex)
        TextPreprocessor.__init__(self)

    
    def get_incose_definition(self, pat=r'Definition:([\s\W\w]+)(?=Elaboration:)', _flags=re.DOTALL):
        self.df['definition'] = self.df['extract'].apply(lambda s: ''.join(re.findall(pat, s, flags=_flags)))
        return self.df

    
    def get_incose_elaboration(self, pat=r'Elaboration:([\s\W\w]+)(?=Examples:)', _flags=re.DOTALL):
        self.df['elaboration'] = self.df['extract'].apply(lambda s: ''.join(re.findall(pat, s, flags=_flags)))
        return self.df

    
    def get_incose_rule_number(self, pat=r'^ (R\d+) –', _flags=re.DOTALL):
        self.df['rule_number'] = self.df['extract'].apply(lambda s: re.search(pat, s, flags=_flags).group(1))
        return self.df
    
    
    def get_incose_examples(self, pat=r'Examples:(.*)$', _flags=re.DOTALL):
        self.df['examples'] = self.df['extract'].apply(lambda s: ''.join(re.findall(pat, s, flags=_flags)))
        return self.df

    
    def remove_bracketed_text(self, col='examples', pat=r'\[[^\]]+\]', _flags=re.DOTALL):
        self.df[col] = self.df[col].apply(lambda s: ''.join(re.sub(pat, '', s, flags=_flags)))
        return self.df
    
    
    def clean_incose_examples(self, col='examples', pat=r'(Exceptions and relationships:.*)$', _flags=re.DOTALL):
        self.df[col] = self.df[col].apply(lambda s: ''.join(re.sub(pat, '', s, flags=_flags)))
        return self.df
    
    
    def preprocess_rules_section_4(self, inpath, outpath, start_page, end_page, replace_tokens, subpatterns, replace_with):
        self.get_pdf_text(inpath, start_page, end_page)
        self.save_text(outpath / "extract.txt")
        self._pipeline = [
            partial(self.replace, replace_tokens=replace_tokens, replace_with=replace_with),
            partial(self.resub, patterns=subpatterns, replace_with=replace_with),
            self.remove_multi_whitespace,
        ]
        self.text = self.clean_text(self.text)
        self.save_text(outpath / "extract-post-clean.txt")
        self.get_sections_df()
        self.add_section_text()
        self.get_incose_rule_number()
        self.get_incose_definition()
        self.get_incose_elaboration()
        self.get_incose_examples()
        self.remove_bracketed_text()
        self.clean_incose_examples()
        return self


class BuildIncoseTemplates(BuildTemplates):

    LOGGERNAME = f"{src.BASE_LOGGERNAME}.BuildIncoseTemplates"
 
    def __init__(self, df, base_messages, output_data_folder_path):
        super().__init__(df, base_messages)
        self.output_data_folder_path = output_data_folder_path
        self.add_message_col_to_frame("system")
        self.add_message_col_to_frame("user")
        # replace relevant template variables with INCOSE data (e.g., definition, examples)
        self.replace_prompt_variable_from_frame("user_message", "definition")
        self.replace_prompt_variable_from_frame("user_message", "examples")
        # build prompt templates based on incose rules
        self.assemble_templates_from_df(
            system_message_colname='system_message',
            user_message_colname='user_message',
            template_name_prefix='R'
        )

class BuildIncoseEvalConfig:
    
    LOGGERNAME = f"{src.BASE_LOGGERNAME}.BuildIncoseEvalConfig"
    
    def __init__(self, incose_guide_df: pd.DataFrame, output_data_folder_path: str, templates, rule_to_eval_map: dict, rule_num_col='rule_number'): #base_messages: dict):
        
        self.incose_guide_df = incose_guide_df
        self.rule_num_col = rule_num_col
        self.rule_to_eval_map = rule_to_eval_map
        self.templates = templates
        self.evals_config={}
        self.load_evals_config()
        self.output_data_folder = output_data_folder_path
        #self.output_func = base_messages['func']
        BuildIncoseEvalConfig.write_text(Path(self.output_data_folder)/"evals_config.txt", "w", pformat(self.evals_config))
        #incose_guide_sections_df = self.incose_guide_df
        #utils.to_excel(incose_guide_sections_df, self.output_data_folder, False, 'incose_guide_sections_df')
    
    @staticmethod
    def write_text(fp: Path, mode: str, data: dict):
        with open(fp, mode) as f:
            f.write(data)
    
    def load_evals_config(self):
        '''load evaluations configuration'''
        for _rule, _evalname in self.rule_to_eval_map.items():
            self.evals_config[_evalname] = {}
            self.evals_config[_evalname]["func"] = getattr(pe, _evalname) 
            self.evals_config[_evalname]["template"] = self.templates[_rule]
        return self.evals_config

class IncoseRequirementReviewer(PromptRunner):
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

    LOGGERNAME = f"{src.BASE_LOGGERNAME}.IncoseRequirementReviewer"
        
    def __init__(self, use_structured_llm: bool, llm, pydantic_model, templates, evals_config, id_col='Requirement'):
        super().__init__(use_structured_llm, llm, pydantic_model)
        self.templates = templates
        self.evals_config = evals_config
        self.evals_chains = []
        self.id_col = id_col

    def revise(self, evals_lists, args_lists, capture_func):
        self.assemble_eval_chain_list(evals_lists, capture_func)
        results = asyncio.run(self.run_multiple_chains(self.evals_chains, args_lists))
        results_df = self.cast_results_to_frame(results)
        return results_df
    
    
    def assemble_eval_chain_list(self, evals_lists, capture_func=None):
        print(evals_lists)
        for eval_list in evals_lists:
            print(f"Eval list: {eval_list}")
            row_chain = []
            for _eval in eval_list:
                template = self.evals_config[_eval]["template"]
                row_chain.append(self.assemble_chain_from_template(template, capture_func))
            composed_chain = RunnableSequence(*row_chain)
            self.evals_chains.append(composed_chain)
        return self.evals_chains

    def assemble_chain_from_template(self, template, capture_func=None):

        if capture_func is not None:
            chain = RunnableLambda(lambda x: {"req":x}) | template | self.llm | (lambda x: x.content) | (capture_func)
        else:
            chain = RunnableLambda(lambda x: {"req":x}) | template | self.llm | (lambda x: x.content)
        return chain
    
    def cast_results_to_frame(self, results):
        """Casts the results returned by the LLM (e.g., via self.run_multiple_chains) to a dataframe
        
        Arguments:
            results (List[str]): Results returned by the LLM 
            id_col (str): The column name in the dataframe containing the input requirement text which is to be evaluated using the runner.
        """
        results_df = pd.DataFrame(results)
        results_df = results_df.rename(columns={0:self.id_col})
        return results_df
    
    def run_eval_sequence(self, df: pd.DataFrame, col: str, failed_eval_col: str, col_suffix: Union[str, None], eval_to_rule_map: Union[dict, None] = None):
        self.call_evals(df, col)
        self.get_failed_evals(df)
        if eval_to_rule_map is not None:
            self.map_failed_evals_to_rule_ids(df, eval_to_rule_map)
        df = df.drop(columns=[c for c in df.columns if c.startswith('eval')])
        if col_suffix is not None:
            failed_eval_cols = [c for c in df.columns if c.startswith('failed_evals')]
            for c in failed_eval_cols:
                df.rename(columns={c:f"{col_suffix}_{c}"}, inplace=True)
                df[f"if_resolved_{col_suffix}_{c}"] = df[f"{col_suffix}_{c}"].apply(lambda _l: 1 - int(bool(len(_l))))
                df[f"%_resolved_{col_suffix}_{c}"] = round(sum(df[f"if_resolved_{col_suffix}_{c}"]) / len(df[f"if_resolved_{col_suffix}_{c}"]) * 100, 0)
        return df
    
    
    def call_evals(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        # run evals for each row of the dataframe
        for _index, _row in df.iterrows():
            for key, value in self.evals_config.items():  # fix this line
                eval_func_to_call = self.evals_config[key]["func"] 
                eval_result = eval_func_to_call(_row[col])
                df.loc[_index, key] = pe.convert_bool_to_ohe(
                    eval_result
                )
        return df

    
    def get_failed_evals(self, df: pd.DataFrame) -> pd.DataFrame:
        eval_cols = [c for c in df.columns if c.startswith("eval")]
        df['failed_evals'] = df[eval_cols].apply(lambda _l: [eval_cols[e[0]] for e in enumerate(_l) if e[1]==1.0], axis=1)
        return df
    
    
    def map_failed_evals_to_rule_ids(self, df: pd.DataFrame, eval_to_rule_map: dict) -> pd.DataFrame:
        df['failed_evals_rule_ids'] = df['failed_evals'].apply(lambda _l: map_A_to_B(_l, eval_to_rule_map) if type(_l) == list else None)
        return df