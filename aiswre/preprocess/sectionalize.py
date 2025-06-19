from pathlib import Path
import re
import numpy as np
import pandas as pd
import pymupdf

class Sectionalize:

    def __init__(self, regex=r'([1-9][0-9]?\.([0-9]\.)?[0-9]?)[\s]+[A-Z]+'):
        self.regex = regex

    def get_sections_df(self, text: str):
        section_numbers = [{'start': m.start(1), 'end': m.end(1), 'match': m.group(1), 'full_match':m.group(0)} for m in list(re.finditer(self.regex,text))]
        df = pd.DataFrame(section_numbers)
        return df
    
    @staticmethod
    def add_section_text(df, corpus, match_start_col='start', match_end_col='end'):
        df['start_idx']=df[match_end_col].astype(int)
        df['end_idx']=df[match_start_col].astype(int).shift(-1).fillna(0)  
        # update the last row such that the end_idx is the end of the corpus
        df.iloc[-1, df.columns.get_loc('end_idx')] = len(corpus) - 1
        df['extract'] = df[['start_idx','end_idx']].apply(lambda i: corpus[int(i[0]):int(i[1]+1)], axis=1)
        return df
    
    @staticmethod
    def get_incose_definition(df, pat='Definition:([\s\W\w]+)(?=Elaboration:)'):
        df['definition'] = df['extract'].apply(lambda s: ''.join(re.findall(pat, s)))
        return df

    @staticmethod
    def get_incose_elaboration(df, pat='Elaboration:([\s\W\w]+)(?=Examples:)'):
        df['elaboration'] = df['extract'].apply(lambda s: ''.join(re.findall(pat, s)))
        return df

    @staticmethod
    def get_incose_examples(df, pat='Examples:([\s\W\w]+)(?=Exceptions and relationships:|$)'):
        df['examples'] = df['extract'].apply(lambda s: ''.join(re.findall(pat, s)))
        return df

    @staticmethod
    def get_pdf_texttrace(fp: Path, page_num):
        doc = pymupdf.open(fp)
        adj_doc = doc[(page_num-1)]
        return adj_doc.get_texttrace()
    
    @staticmethod
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
    def save_text(text: str, op: Path):
        with open(op, "w", encoding="utf-8") as f:
            f.write(text)

    @staticmethod
    def print_text(text: str, start_index=0, end_index=-1):
        print(text[start_index:(end_index+1)])

    @staticmethod
    def section_key(section_id):
        parts = section_id.split('.')
        major_section = int(parts[0])
        subsections = tuple(map(int, parts[1:]))
        return (major_section, subsections)

    @staticmethod
    def sort_sections_df(df: pd.DataFrame, text):
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
        return df
    
    @staticmethod
    def get_section_text(df: pd.DataFrame, text):
        df['curr']=df['start'].astype(int)
        df['prev_full_match']=df['full_match'].shift(1).fillna(0)
        #df['next_adj'] = df[''] 
        df['next']=df['curr'].shift(-1).fillna(0)
        df['prev']=df['curr'].shift(1).fillna(0)
        df['next_curr_idx']=1
        df.loc[df.index[:-1],'next_curr_idx'] = np.diff(df.index)
        df['section_count'] = df.groupby('match').transform('count')['start']
        df['prev_match']=df['match'].shift(1)
        df['prev_<_curr']=df[['curr','prev']].apply(lambda i: int(i[0] >= i[1]), axis=1)
        df['next_>_curr']=df[['curr','next']].apply(lambda i: int(i[0] <= i[1]), axis=1)
        error_df = df.copy()
        error_sections = set(error_df[(error_df['prev_<_curr']==0)|(error_df['next_>_curr']==0)]['match'].values)
        df = df.drop_duplicates(subset='match', keep='first').reset_index()
        df.loc[len(df)-1,'next'] = len(text)
        df['extract'] = df[['curr','next']].apply(lambda i: text[int(i[0]):int(i[1]+1)], axis=1)
        return df
    
    @staticmethod
    def clean_index(df: pd.DataFrame):
        to_drop = []
        for index, row in df.iterrows():           
            if row['next_curr_idx'] != 1:
                if row['section_count'] > 1:
                    if (row['prev_match'] == row['match']):
                        #df.drop(index=index, inplace=True)
                        to_drop.append(True)
                    elif ((row['prev_<_curr'] == 0) or (row['next_>_curr'] == 0)):
                        #df.drop(index=index, inplace=True)
                        #df.loc[df.index[index], 'drop'] = True
                        to_drop.append(True)
                    else:
                        to_drop.append(False)
                else:
                    to_drop.append(False)
            else:
                to_drop.append(False)
        df['to_drop'] = to_drop
        return df