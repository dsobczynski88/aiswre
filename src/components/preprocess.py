import re
import logging
from typing import List, Callable, Union
from functools import partial, reduce
from pathlib import Path
import numpy as np
import pandas as pd
import pymupdf
from tqdm import tqdm
from langchain_core.prompts.chat import ChatPromptTemplate
import src
from src.prj_logger import get_logs
from src import pd_utils


class Sectionalize:
    """
    A general class to load, split and process a pdf into specific sections as per a set regex pattern.

    Attributes:
        regex (str): The regex pattern used to separate sections within a corpus of text
        logger: The logger object which is initially configured through the ProjectLogger class
        text (str): The corpus desired to be split into specific sections
        df (pd.DataFrame): A dataframe containing the splitted text corpus  
    
    Methods:
        get_pdf_text(self, fp: Path, start_page: int, end_page: int) -> str:
           Leverages pymupdf to convert PDFs to text for a given start/end page number.
               Returns the PDF text separated as defined by the regex
        save_text(self, op: Path) -> None:
            Saves the text attribute to a txt file as per the specified output path (op).   
        print_text(self, start_index: int, end_index: int) -> None:
            Prints a slice of the text attribute (from start_index to end_index). 
                Note the end index is included in the returned slice. 
        get_sections_df(self):
                         
        add_section_text()
               
    NOTES: Methods are decorated with @get_logs which logs the timing associated with method execution and 
        any incurred exceptions.   
    """

    LOGGERNAME = f"{src.BASE_LOGGERNAME}.Sectionalize"

    def __init__(self, regex):
        self.regex = regex
        self._logger = logging.getLogger(Sectionalize.LOGGERNAME)
        self.text = None
        self.df = None

    @property
    def logger(self):
        return self._logger
        
    @get_logs(LOGGERNAME)
    def get_pdf_text(self, fp: Path, start_page=0, end_page=-1):
        doc = pymupdf.open(fp)
        if (start_page == 0) and (end_page == -1):
            adj_doc = doc
        elif (start_page == 0) and (end_page != -1):
            adj_doc = doc[:(end_page)]
        elif (start_page != 0) and (end_page == -1):
            adj_doc = doc[(start_page-1):]
        else:
            adj_doc = doc[(start_page-1):(end_page)]
        self.text = chr(12).join([page.get_text() for page in adj_doc])
        return self.text

    @get_logs(LOGGERNAME)
    def get_sections_df(self):
        section_numbers = [{'start': m.start(1), 'end': m.end(1), 'match': m.group(1), 'full_match':m.group(0)} for m in list(re.finditer(self.regex,self.text))]
        self.df = pd.DataFrame(section_numbers)
        return self.df
    
    @get_logs(LOGGERNAME)
    def add_section_text(self, match_start_col='start', match_end_col='end'):
        self.df['start_idx']=self.df[match_end_col].astype(int)
        self.df['end_idx']=self.df[match_start_col].astype(int).shift(-1).fillna(0)  
        # update the last row such that the end_idx is the end of the corpus
        self.df.iloc[-1, self.df.columns.get_loc('end_idx')] = len(self.text) - 1
        self.df['extract'] = self.df[['start_idx','end_idx']].apply(lambda i: self.text[int(i[0]):int(i[1]+1)], axis=1)
        return self.df
    
    @get_logs(LOGGERNAME)
    def save_text(self, op: Path):
        with open(op, "w", encoding="utf-8") as f:
            f.write(self.text)

    @get_logs(LOGGERNAME)
    def print_text(self, start_index=0, end_index=-1):
        if end_index != -1:
            print(self.text[start_index:(end_index+1)])
        else:
            print(self.text[start_index:])

class TextPreprocessor:
    """
    A general class to be used to preprocess text for downstream NLP/ML/AI applications.

    Attributes:
        pipeline: (List[Callable]):  A list of functions to be applied in a cumulative fashion using reduce     
    
    Instance Methods:
        clean_text(self, text: str) -> str:
            Applies a list of text preprocessing functions (self.pipeline) to an input string (text)
                Returns the preprocessed input text. 
        clean_frame(self, df: pd.DataFrame, apply_to_cols: List[str]) -> pd.DataFrame:
            Applies a list of text preprocessing functions (self.pipeline) to each row of the dataframe 
                (df) for each columnan specified in apply_to_cols. Returns the preprocessed df.
    
    Static Methods:
        replace(_str: str, replace_tokens: List[str], replace_with: str) -> str:
            Replaces specific tokens (replace_tokens) in a given string with a fixed string (replace_with).
                Returns the preprocessed string.
        remove_multi_whitespace(_str: str):
            Replaces combinations of multiple whitespace characters (e.g., newline tab, space) with a single
                space. Returns the preprocessed string.

        make_lower(_str: str) -> str:
            Converts all text for a given input string to lowercase. Returns the preprocessed string.

        remove_stopwords(_str: str, STOP_WORDS: List[str]) -> str:
            Replaces specific tokens (STOP_WORDS) in a given string with a single space.
                Assumes input string tokens are denoted by a single space. Returns the preprocessed string.
    
    NOTES: Methods are decorated with @get_logs which logs the timing associated with method execution and 
        any incurred exceptions.   
    """

    LOGGERNAME = f"{src.BASE_LOGGERNAME}.TextPreprocessor"

    def __init__(self):
        self._pipeline = []
        self._logger = logging.getLogger(TextPreprocessor.LOGGERNAME)

    @property
    def pipeline(self):
        return self._pipeline

    @pipeline.setter
    def pipeline(self, new_pipeline: List[Callable]):
        self._pipeline = new_pipeline

    @get_logs(LOGGERNAME)
    def clean_text(self, text: str) -> str:
        return reduce(lambda x, y: y(x), self._pipeline, text)
    
    @get_logs(LOGGERNAME)
    def clean_frame(self, df: pd.DataFrame, apply_to_cols: List[str]) -> pd.DataFrame:
        for acol in apply_to_cols:
            df = pd_utils.replace_null(df, acol, ' ')
            df[acol] = df[acol].astype(str)
            df[acol] = df[acol].apply(lambda s: self.clean_text(self._pipeline, s))
        return df

    @staticmethod
    @get_logs(LOGGERNAME)
    def replace(_str: str, replace_tokens: List[str], replace_with: str) -> str:
        for tok in replace_tokens:
            _str = _str.replace(tok, replace_with)
        return _str

    @staticmethod
    @get_logs(LOGGERNAME)
    def remove_multi_whitespace(_str: str) -> str:
        """Replace multiple spaces with a single space
        Args:
            _str (str): the corpus (str) on which to apply the function
        """
        return re.sub(r'\s{1,}', ' ', _str)
 
    @staticmethod
    @get_logs(LOGGERNAME)
    def make_lower(_str: str) -> str:
        """Make input text lowercase
        Args:
            _str (str): the corpus (str) on which to apply the function
        """
        return _str.lower()
    
    @staticmethod
    @get_logs(LOGGERNAME)
    def remove_stopwords(_str:str, STOP_WORDS: List[str]) -> str:
        """Remove stopwords from a given input string
        Args:
            _str (str): the corpus (str) on which to apply the function
        """   
        lst_str = _str.split()
        if STOP_WORDS is not None:
            lst_str = [word for word in lst_str if word not in STOP_WORDS]
        return ' '.join(lst_str)
    
class ProcessTemplates:
    """
    A class which generates prompt template strings from input system and user messages. These
    messages contain variables denoted with curly braces. This class is specifically designed
    to take in a dataframe containing column names which match the prompt variable names to 
    populate the prompts which specific contents from each dataframe row.

    Attributes:
        df (pd.DataFrame): The dataframe which contains the values of prompt content variables.
            These values within specific dataframe columns are used in creating the final templates.
        base_messages (dict): A dictionary containing system and user messages. Additionally,the
            messages dictionary contains fields name and description which describe the prompt.

    Methods:
    """

    LOGGERNAME = f"{src.BASE_LOGGERNAME}.ProcessTemplates"

    def __init__(self, df, base_messages):
        self.df = df
        self.base_messages = base_messages
        self.templates = {}
    
    @get_logs(LOGGERNAME)
    def add_message_col_to_frame(self, name):
        self.df[f"{name}_message"] = self.base_messages[name]
    
    @get_logs(LOGGERNAME)
    def replace_prompt_variable_from_frame(self, message_col, replace_col):
        self.df[message_col] = self.df[[message_col, replace_col]].apply(lambda l: l[0].replace("{"+replace_col+"}",l[1]), axis=1)

    @get_logs(LOGGERNAME)
    def assemble_templates_from_df(self, system_message_colname='system_message', user_message_colname='user_message', template_name_prefix='template'):
        '''Loop over dataframe to build a unique prompt template for each dataframe row'''
        for index, row in tqdm(self.df.iterrows()):
            system_message=row[system_message_colname]
            user_message=row[user_message_colname]
            self.templates[f"{template_name_prefix}{index+1}"] = ProcessTemplates.get_template_from_messages(system_message, user_message)
        return self.templates

    @staticmethod
    @get_logs(LOGGERNAME)
    def get_template_from_messages(system_message: str, user_message: str) -> ChatPromptTemplate:
        '''Create a ChatPromptTemplate given an input dictionary containing keys: system, user'''
        return ChatPromptTemplate.from_messages([
                ("system",system_message),
                ("human", user_message)
            ])

    
class ProcessIncoseTemplates(ProcessTemplates):

    LOGGERNAME = f"{src.BASE_LOGGERNAME}.ProcessIncoseTemplates"

    def __init__(self, df, base_messages, output_data_folder_path):
        super().__init__(df, base_messages)
        self.output_data_folder_path = output_data_folder_path
    
    @get_logs(LOGGERNAME)
    def __call__(self):
        # add system and user messages as columns to each row of dataframe
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
        incose_guide_sections_df = self.df
        pd_utils.to_excel(incose_guide_sections_df, self.output_data_folder_path, False, 'incose_guide_sections_df')
        return self

class PreprocessIncoseGuide(TextPreprocessor, Sectionalize):

    LOGGERNAME = f"{src.BASE_LOGGERNAME}.PreprocessIncoseGuide"
    INCOSE_SECTIONS_REGEX_PAT = r'([1-9]\.([0-9]+\.)?[0-9]?)[\s]+R\d'
    INCOSE_GUIDE_FILEPATH = Path('./src/data/incose_gtwr.pdf')
    OUTPUT_DATA_FOLDER = Path('./src/data')
    REPLACE_TOKENS = ['INCOSE-TP-2010-006-04| VERS/REV:4  |  1 July 2023', '{', '}']
    REPLACE_WITH = ' '

    def __init__(self, regex=INCOSE_SECTIONS_REGEX_PAT):
        Sectionalize.__init__(self, regex)
        TextPreprocessor.__init__(self)

    @get_logs(LOGGERNAME)
    def get_incose_definition(self, pat=r'Definition:([\s\W\w]+)(?=Elaboration:)'):
        self.df['definition'] = self.df['extract'].apply(lambda s: ''.join(re.findall(pat, s)))
        return self.df

    @get_logs(LOGGERNAME)
    def get_incose_elaboration(self, pat=r'Elaboration:([\s\W\w]+)(?=Examples:)'):
        self.df['elaboration'] = self.df['extract'].apply(lambda s: ''.join(re.findall(pat, s)))
        return self.df

    @get_logs(LOGGERNAME)
    def get_incose_examples(self, pat=r'Examples:(.*)$', flags=re.DOTALL):
        self.df['examples'] = self.df['extract'].apply(lambda s: ''.join(re.findall(pat, s, flags=flags)))
        return self.df
    
    @get_logs(LOGGERNAME)
    def clean_incose_examples(self, pat='(Exceptions and relationships:.*)$', flags=re.DOTALL):
        self.df['examples'] = self.df['examples'].apply(lambda s: ''.join(re.sub(pat, '', s)))
        return self.df
    
    @get_logs(LOGGERNAME)
    def preprocess_rules_section_4(self, inpath=INCOSE_GUIDE_FILEPATH, outpath=OUTPUT_DATA_FOLDER, start_page=65, end_page=115, replace_tokens=REPLACE_TOKENS, replace_with=REPLACE_WITH):
        self.get_pdf_text(inpath, start_page, end_page)
        self.save_text(outpath / "extract.txt")
        self._pipeline = [
            partial(self.replace, replace_tokens=replace_tokens, replace_with=replace_with),
            self.remove_multi_whitespace,
        ]
        self.text = self.clean_text(self.text)
        self.save_text(outpath / "extract-post-clean.txt")
        self.get_sections_df()
        self.add_section_text()
        self.get_incose_definition()
        self.get_incose_examples()
        self.clean_incose_examples()
        return self