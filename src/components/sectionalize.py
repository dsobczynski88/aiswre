from pathlib import Path
import re
import logging
import numpy as np
import pandas as pd
import pymupdf
from src.prj_logger import get_logs


class Sectionalize:

    BASE_LOGGERNAME = "reviewer"
    LOGGERNAME = f"{BASE_LOGGERNAME}.sectionalize"

    def __init__(self, regex=r'([1-9][0-9]?\.([0-9]\.)?[0-9]?)[\s]+[A-Z]+'):
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
    def parse_incose_guide(self):
        self.get_sections_df()
        self.add_section_text()
        self.get_incose_definition()
        self.get_incose_examples()
        self.clean_incose_examples()
        return self
        
    @get_logs(LOGGERNAME)
    def add_section_text(self, match_start_col='start', match_end_col='end'):
        self.df['start_idx']=self.df[match_end_col].astype(int)
        self.df['end_idx']=self.df[match_start_col].astype(int).shift(-1).fillna(0)  
        # update the last row such that the end_idx is the end of the corpus
        self.df.iloc[-1, self.df.columns.get_loc('end_idx')] = len(self.text) - 1
        self.df['extract'] = self.df[['start_idx','end_idx']].apply(lambda i: self.text[int(i[0]):int(i[1]+1)], axis=1)
        return self.df
    
    @get_logs(LOGGERNAME)
    def get_incose_definition(self, pat='Definition:([\s\W\w]+)(?=Elaboration:)'):
        self.df['definition'] = self.df['extract'].apply(lambda s: ''.join(re.findall(pat, s)))
        return self.df

    @get_logs(LOGGERNAME)
    def get_incose_elaboration(self, pat='Elaboration:([\s\W\w]+)(?=Examples:)'):
        self.df['elaboration'] = self.df['extract'].apply(lambda s: ''.join(re.findall(pat, s)))
        return self.df

    @get_logs(LOGGERNAME)
    def get_incose_examples(self, pat='Examples:(.*)$', flags=re.DOTALL):
        self.df['examples'] = self.df['extract'].apply(lambda s: ''.join(re.findall(pat, s, flags=flags)))
        return self.df
    
    @get_logs(LOGGERNAME)
    def clean_incose_examples(self, pat='(Exceptions and relationships:.*)$', flags=re.DOTALL):
        self.df['examples'] = self.df['examples'].apply(lambda s: ''.join(re.sub(pat, '', s)))
        return self.df

    @staticmethod
    @get_logs(LOGGERNAME)
    def get_pdf_texttrace(fp: Path, page_num):
        doc = pymupdf.open(fp)
        adj_doc = doc[(page_num-1)]
        return adj_doc.get_texttrace()
    
    @get_logs(LOGGERNAME)
    def save_text(self, op: Path):
        with open(op, "w", encoding="utf-8") as f:
            f.write(self.text)

    @get_logs(LOGGERNAME)
    def print_text(self, start_index=0, end_index=-1):
        print(self.text[start_index:(end_index+1)])