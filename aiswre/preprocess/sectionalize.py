from typing import List
from pathlib import Path
import re
import numpy as np
import pandas as pd
import pymupdf
from spire.doc import *
from spire.doc.common import *

# [(m.start(0), m.end(0), m.group(0)) for m in list(re.finditer('2.0|3.0',s))]
# [(m.start(0), m.end(0), m.group(0)) for m in list(re.finditer(rf'{x}.0[\s]+[A-Z]',s))]
# r'([1-9][0-9]?\.([0-9]\.)?[0-9]?)[\s]+[A-Z]+'

class Sectionalize:

    def __init__(self, regex):
        self.regex = regex

    def get_sections(self, text: str):
        section_numbers = [{'start': m.start(1), 'end': m.end(1), 'match': m.group(1)} for m in list(re.finditer(self.regex,text))]
        df = pd.DataFrame(section_numbers)
        return df

    @staticmethod
    def get_pdf_texttrace(fp: Path, page_num):
        doc = pymupdf.open(fp)
        adj_doc = doc[(page_num-1)]
        return adj_doc.get_texttrace()
    
    @staticmethod
    def get_pdf_text(fp: Path, start_page=0, end_page=-1):
        doc = pymupdf.open(fp)

        adj_doc = doc[(start_page-1):(end_page+1)]

        all_text = chr(12).join([page.get_text() for page in adj_doc])
        return all_text

    @staticmethod
    def save_text(text: str, op: Path):
        with open(op, "w", encoding="utf-8") as f:
            f.write(text)

    @staticmethod
    def print_text(text: str, start_index=0, end_index=-1):
        print(text[start_index:(end_index+1)])

    @staticmethod
    def process_sectionalized_df(df: pd.DataFrame):
        df['if_valid'] = df['match'].apply(lambda s: None if s[-1] == "." else True)
        df.dropna(subset=['if_valid'], inplace=True)
        df.reset_index(drop=True, inplace=True)
        section_ids = pd.DataFrame(df['match'].apply(lambda s: s.split(".")).tolist())
        section_ids_cols = [f"s_{c}" for c in section_ids.columns]
        section_ids.columns = section_ids_cols
        section_ids.fillna(0, inplace=True)
        df = pd.merge(left=df, right=section_ids, right_index=True, left_index=True)
        df[section_ids_cols] = df[section_ids_cols].astype(int)
        df.sort_values(by=section_ids_cols, inplace=True)
        #df['curr']=df[section_ids_cols].map(str).apply(lambda i: int("".join([_i for _i in i])), axis=1)
        df['curr']=df['start'].astype(int)
        df['next']=df['curr'].shift(-1).fillna(0)
        df['prev']=df['curr'].shift(1).fillna(0)
        df['curr>prev']=df[['curr','prev']].apply(lambda i: int(i[0] >= i[1]), axis=1)
        df['curr<next']=df[['curr','next']].apply(lambda i: int(i[0] <= i[1]), axis=1)
        error_df = df.copy()
        error_sections = set(error_df[(error_df['curr>prev']==0)|(error_df['curr<next']==0)]['match'].values)
        print(error_sections)
        df = df.drop_duplicates(subset='match', keep='first').reset_index()
        df.loc[len(df)-1,'next'] = len(text)
        df['extract'] = df[['curr','next']].apply(lambda i: text[int(i[0]):int(i[1]+1)], axis=1)
        for err_sec in error_sections:
            if err_sec != df['match'].values[-1]:
                df.loc[df['match']==err_sec,'start']=None
                df.loc[df['match']==err_sec,'end']=None
                df.loc[df['match']==err_sec,'extract']=None
        return df
    
    @staticmethod
    def section_key(section_id):
        parts = section_id.split('.')
        major_section = int(parts[0])
        subsections = tuple(map(int, parts[1:]))
        return (major_section, subsections)

#regex_pattern = r'([\s]+[1-9][\s]+[A-Z]+)'\n1 \nS'
regex_pattern = r'\n(\d)\s+(\w|\*)+'
parser = Sectionalize(regex_pattern)
text = parser.get_pdf_text(fp='./src/data/pdfs/IEC_82304.pdf', start_page=8, end_page=27)
#parser.save_text(text, op='./src/data/pdfs/IEC_82304.txt')
print(text)
df = parser.get_sections(text)
print(set(df['match'].values))
df.to_excel('./src/data/pdfs/df_IEC_82304.xlsx')
cleaned_df = parser.process_sectionalized_df(df)
cleaned_df.to_excel('./src/data/pdfs/df_IEC_82304_cleaned.xlsx')