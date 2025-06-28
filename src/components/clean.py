import re
from functools import partial, reduce
import pandas as pd
from src import pd_utils


class TextPreprocessor:

    def __init__(self):
        self._pipeline = []

    @property
    def pipeline(self):
        return self._pipeline

    @pipeline.setter
    def pipeline(self, new_pipeline):
        self._pipeline = new_pipeline

    def clean_text(self, text):
        return reduce(lambda x, y: y(x), self._pipeline, text)
    
    def clean_frame(self, df, apply_to_cols):
        for acol in apply_to_cols:
            df = pd_utils.replace_null(df, acol, ' ')
            df[acol] = df[acol].astype(str)
            df[acol] = df[acol].apply(lambda s: self.clean_text(self._pipeline, s))
        return df

    @staticmethod
    def replace(_str, replace_tokens, replace_with):
        for tok in replace_tokens:
            _str = _str.replace(tok, replace_with)
        return _str

    @staticmethod
    def remove_multi_whitespace(_str:str) -> str:
        """Replace multiple spaces with a single space
        Args:
            _str (str): the corpus (str) on which to apply the function
        """
        return re.sub(r'\s{1,}', ' ', _str)
 
    @staticmethod
    def make_lower(_str:str) -> str:
        """Make input text lowercase
        Args:
            _str (str): the corpus (str) on which to apply the function
        """
        return _str.lower()
    
    @staticmethod
    def remove_stopwords(_str:str, STOP_WORDS) -> str:
        """Remove stopwords from a given input string
        Args:
            _str (str): the corpus (str) on which to apply the function
        """   
        lst_str = _str.split()
        if STOP_WORDS is not None:
            lst_str = [word for word in lst_str if word not in STOP_WORDS]
        return ' '.join(lst_str)