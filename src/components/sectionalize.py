from pathlib import Path
import re
import logging
import numpy as np
import pandas as pd
import pymupdf
from aiswre.prj_logger import get_logs

class Sectionalize:

    BASE_LOGGERNAME = "reviewer"
    LOGGERNAME = f"{BASE_LOGGERNAME}.sectionalize"

    def __init__(self, regex=r'([1-9][0-9]?\.([0-9]\.)?[0-9]?)[\s]+[A-Z]+'):
        self.regex = regex
        self._logger = logging.getLogger(Sectionalize.LOGGERNAME)

    @property
    def logger(self):
        return self._logger

    @get_logs(LOGGERNAME)
    def get_sections_df(self, text: str):
        section_numbers = [{'start': m.start(1), 'end': m.end(1), 'match': m.group(1), 'full_match':m.group(0)} for m in list(re.finditer(self.regex,text))]
        df = pd.DataFrame(section_numbers)
        return df
    
    @staticmethod
    @get_logs(LOGGERNAME)
    def add_section_text(df, corpus, match_start_col='start', match_end_col='end'):
        df['start_idx']=df[match_end_col].astype(int)
        df['end_idx']=df[match_start_col].astype(int).shift(-1).fillna(0)  
        # update the last row such that the end_idx is the end of the corpus
        df.iloc[-1, df.columns.get_loc('end_idx')] = len(corpus) - 1
        df['extract'] = df[['start_idx','end_idx']].apply(lambda i: corpus[int(i[0]):int(i[1]+1)], axis=1)
        return df
    
    @staticmethod
    @get_logs(LOGGERNAME)
    def get_incose_definition(df, pat='Definition:([\s\W\w]+)(?=Elaboration:)'):
        df['definition'] = df['extract'].apply(lambda s: ''.join(re.findall(pat, s)))
        return df

    @staticmethod
    @get_logs(LOGGERNAME)
    def get_incose_elaboration(df, pat='Elaboration:([\s\W\w]+)(?=Examples:)'):
        df['elaboration'] = df['extract'].apply(lambda s: ''.join(re.findall(pat, s)))
        return df

    @staticmethod
    @get_logs(LOGGERNAME)
    def get_incose_examples(df, pat='Examples:([\s\W\w]+)(?=Exceptions and relationships:|$)'):
        df['examples'] = df['extract'].apply(lambda s: ''.join(re.findall(pat, s)))
        return df

    @staticmethod
    @get_logs(LOGGERNAME)
    def get_pdf_texttrace(fp: Path, page_num):
        doc = pymupdf.open(fp)
        adj_doc = doc[(page_num-1)]
        return adj_doc.get_texttrace()
    
    @staticmethod
    @get_logs(LOGGERNAME)
    def get_pdf_text(fp: Path, start_page=0, end_page=-1):
        doc = pymupdf.open(fp)

        if (start_page == 0) and (end_page == -1):
            adj_doc = doc
        elif (start_page == 0) and (end_page != -1):
            adj_doc = doc[:(end_page)]
        elif (start_page != 0) and (end_page == -1):
            adj_doc = doc[(start_page-1):]
        else:
            adj_doc = doc[(start_page-1):(end_page)]

        all_text = chr(12).join([page.get_text() for page in adj_doc])
        return all_text

    @staticmethod
    @get_logs(LOGGERNAME)
    def save_text(text: str, op: Path):
        with open(op, "w", encoding="utf-8") as f:
            f.write(text)

    @staticmethod
    @get_logs(LOGGERNAME)
    def print_text(text: str, start_index=0, end_index=-1):
        print(text[start_index:(end_index+1)])