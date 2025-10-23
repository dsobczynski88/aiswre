import re
import logging
from typing import List, Callable, Optional, Union
from functools import partial, reduce
from pathlib import Path
import json
import numpy as np
import pandas as pd
import fitz
from tqdm import tqdm
from langchain_core.prompts.chat import ChatPromptTemplate


class Sectionalize:
    """
    A class to load, split and process a PDF into specific sections based on regex patterns.
    
    This class handles PDF text extraction and segmentation into logical sections based on
    provided regular expression patterns.
    """
    LOGGER_NAME = "projectlog.Sectionalize"

    def __init__(self, regex: str):
        """
        Initialize the Sectionalize object.
        
        Args:
            regex: Regular expression pattern used to identify section boundaries
        """
        self.regex = regex
        self._logger = logging.getLogger(self.LOGGER_NAME)
        self.text = None
        self.df = None

    def get_pdf_text(self, fp: Path, start_page: int = 0, end_page: int = -1) -> str:
        """
        Extract text from a PDF file with optional page range.
        
        Args:
            fp: Path to the PDF file
            start_page: First page to extract (0-indexed)
            end_page: Last page to extract (-1 for all pages)
            
        Returns:
            Extracted text from the PDF
        """
        doc = fitz.open(fp)
        n_pages = doc.page_count

        if end_page == -1 or end_page > n_pages:
            end_page = n_pages

        texts = [
            doc.load_page(i).get_text() for i in range(start_page, end_page)
        ]
        self.text = chr(12).join(texts)
        return self.text

    def get_sections_df(self) -> pd.DataFrame:
        """
        Create a DataFrame containing section boundaries based on regex pattern.
        
        Returns:
            DataFrame with columns for section boundaries and matched text
        """
        if self.text is None:
            self._logger.error("No text loaded. Call get_pdf_text() first.")
            raise ValueError("No text loaded. Call get_pdf_text() first.")
            
        section_matches = [
            {
                'start': m.start(1), 
                'end': m.end(1), 
                'match': m.group(1), 
                'full_match': m.group(0)
            } 
            for m in re.finditer(self.regex, self.text)
        ]
        
        self.df = pd.DataFrame(section_matches)
        return self.df   

    def add_section_text(self, match_start_col: str = 'start', match_end_col: str = 'end') -> Union[pd.DataFrame, None]:
        """
        Add extracted text between section boundaries to the DataFrame.
        
        Args:
            match_start_col: Column name containing start positions
            match_end_col: Column name containing end positions
            
        Returns:
            DataFrame with added 'extract' column containing section text
        """
        if self.df is None:
            self._logger.error("No sections DataFrame. Call get_sections_df() first.")
            raise ValueError("No sections DataFrame. Call get_sections_df() first.")

        if self.text is not None:    
            # Create index columns for extraction boundaries
            start_idx_col = f'{match_start_col}_idx'
            end_idx_col = f'{match_end_col}_idx'
            
            # Set extraction boundaries
            self.df[start_idx_col] = self.df[match_end_col].astype(int)
            self.df[end_idx_col] = self.df[match_start_col].astype(int).shift(-1).fillna(len(self.text))
            
            # Extract text between boundaries
            self.df['extract'] = self.df[[start_idx_col, end_idx_col]].apply(
                lambda i: self.text[int(i[0]):int(i[1])], axis=1
            )
            return self.df
    
    def save_text(self, output_path: Path) -> None:
        """
        Save extracted text to a file.
        
        Args:
            output_path: Path where text will be saved
        """
        if self.text is None:
            self._logger.error("No text to save. Call get_pdf_text() first.")
            raise ValueError("No text to save. Call get_pdf_text() first.")
            
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(self.text)
        
        self._logger.info(f"Text saved to {output_path}")

    def print_text(self, start_index: int = 0, end_index: int = -1) -> None:
        """
        Print a slice of the extracted text.
        
        Args:
            start_index: Starting index of text slice
            end_index: Ending index of text slice (-1 for end of text)
        """
        if self.text is None:
            self._logger.error("No text to print. Call get_pdf_text() first.")
            raise ValueError("No text to print. Call get_pdf_text() first.")
            
        if end_index != -1:
            print(self.text[start_index:end_index])
        else:
            print(self.text[start_index:])


class TextPreprocessor:
    """
    A class for preprocessing text using a pipeline of transformation functions.
    
    This class applies a sequence of text cleaning operations for NLP/ML/AI applications.
    """
    LOGGER_NAME = "projectlog.TextPreprocessor"

    def __init__(self, pipeline: Optional[List[Callable]] = None):
        """
        Initialize the TextPreprocessor with an optional pipeline.
        
        Args:
            pipeline: List of text processing functions to apply in sequence
        """
        self._pipeline = pipeline or []
        self._logger = logging.getLogger(self.LOGGER_NAME)

    @property
    def pipeline(self) -> List[Callable]:
        """Get the current preprocessing pipeline."""
        return self._pipeline

    @pipeline.setter
    def pipeline(self, new_pipeline: List[Callable]) -> None:
        """Set a new preprocessing pipeline."""
        self._pipeline = new_pipeline
    
    def add_processor(self, processor_func: Callable) -> None:
        """
        Add a processing function to the pipeline.
        
        Args:
            processor_func: Function that takes a string and returns a processed string
        """
        self._pipeline.append(processor_func)
    
    def clean_text(self, text: str) -> str:
        """
        Apply the preprocessing pipeline to a text string.
        
        Args:
            text: Input text to process
            
        Returns:
            Processed text after applying all pipeline functions
        """
        if not self._pipeline:
            self._logger.warning("Empty pipeline - text will be returned unchanged")
            return text
            
        return reduce(lambda x, func: func(x), self._pipeline, text)
    
    def clean_frame(self, df: pd.DataFrame, apply_to_cols: List[str]) -> pd.DataFrame:
        """
        Apply the preprocessing pipeline to specified columns in a DataFrame.
        
        Args:
            df: Input DataFrame
            apply_to_cols: List of column names to process
            
        Returns:
            DataFrame with processed text columns
        """
        from src import utils  # Import here to avoid circular imports
        
        result_df = df.copy()
        
        for col in apply_to_cols:
            if col not in result_df.columns:
                self._logger.warning(f"Column '{col}' not found in DataFrame - skipping")
                continue
                
            # Handle null values and ensure string type
            result_df = utils.replace_null(result_df, col, ' ')
            result_df[col] = result_df[col].astype(str)
            
            # Apply the cleaning pipeline
            result_df[col] = result_df[col].apply(self.clean_text)
            
        return result_df

    # Static utility methods for common text processing operations
    @staticmethod
    def replace(text: str, replace_tokens: List[str], replace_with: str) -> str:
        """
        Replace specific tokens in text with a replacement string.
        
        Args:
            text: Input text
            replace_tokens: List of strings to replace
            replace_with: Replacement string
            
        Returns:
            Text with replacements applied
        """
        for token in replace_tokens:
            text = text.replace(token, replace_with)
        return text

    @staticmethod
    def resub(text: str, patterns: List[str], replace_with: str, flags: int = re.DOTALL) -> str:
        """
        Apply regex substitutions to text.
        
        Args:
            text: Input text
            patterns: List of regex patterns to match
            replace_with: Replacement string
            flags: Regex flags
            
        Returns:
            Text with regex substitutions applied
        """
        for pattern in patterns:
            text = re.sub(pattern, replace_with, text, flags=flags)
        return text

    @staticmethod
    def remove_multi_whitespace(text: str) -> str:
        """
        Replace multiple whitespace characters with a single space.
        
        Args:
            text: Input text
            
        Returns:
            Text with normalized whitespace
        """
        return re.sub(r'\s+', ' ', text)
 
    @staticmethod
    def make_lower(text: str) -> str:
        """
        Convert text to lowercase.
        
        Args:
            text: Input text
            
        Returns:
            Lowercase text
        """
        return text.lower()
    
    @staticmethod
    def remove_stopwords(text: str, stop_words: List[str]) -> str:
        """
        Remove stopwords from text.
        
        Args:
            text: Input text
            stop_words: List of stopwords to remove
            
        Returns:
            Text with stopwords removed
        """
        if not stop_words:
            return text
            
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in stop_words]
        return ' '.join(filtered_words)


class BuildTemplates:
    """
    A class for generating prompt templates from a DataFrame and base messages.
    
    This class creates LangChain prompt templates by substituting variables in message templates
    with values from DataFrame columns.
    """
    LOGGER_NAME = "projectlog.BuildTemplates"

    def __init__(self, df: Union[pd.DataFrame, None], base_messages: Union[dict[str, str], None]):
        """
        Initialize the BuildTemplates object.
        
        Args:
            df: DataFrame containing values for template variables
            base_messages: Dictionary with system and user message templates
        """
        self.df = None if df is None else df.copy()
        self.base_messages = base_messages
        self.templates = {}
        self._logger = logging.getLogger(self.LOGGER_NAME)
    
    def add_message_col_to_frame(self, message_name: str) -> None:
        """
        Add a message template as a new column in the DataFrame.
        
        Args:
            message_name: Key in base_messages to add as a column
        """
        if (self.df is not None) and (self.base_messages is not None):
            if message_name not in list(self.base_messages.keys()):
                self._logger.error(f"Message '{message_name}' not found in base_messages")
                raise KeyError(f"Message '{message_name}' not found in base_messages")
            self.df[f"{message_name}_message"] = self.base_messages[message_name]
    
    def replace_prompt_variable(self, message_col: str, variable_name: str, replace_col: str) -> None:
        """
        Replace a variable in a message template with values from a DataFrame column.
        
        Args:
            message_col: Column containing the message template
            variable_name: Variable name to replace (without braces)
            replace_col: Column containing replacement values
        """
        if (self.df is not None):
            if message_col not in self.df.columns:
                self._logger.error(f"Message column '{message_col}' not found in DataFrame")
                raise KeyError(f"Message column '{message_col}' not found in DataFrame")
                
            if replace_col not in self.df.columns:
                self._logger.error(f"Replacement column '{replace_col}' not found in DataFrame")
                raise KeyError(f"Replacement column '{replace_col}' not found in DataFrame")
                
            self.df[message_col] = self.df.apply(
                lambda row: row[message_col].replace(f"{{{variable_name}}}", str(row[replace_col])), 
                axis=1
            )

    def replace_prompt_variables_from_frame(self, message_col: str, replace_col: str) -> None:
        """
        Replace variables in a message template with values from a DataFrame column.
        
        Args:
            message_col: Column containing the message template
            replace_col: Column containing replacement values
        """
        if (self.df is not None):
            if message_col not in self.df.columns:
                self._logger.error(f"Message column '{message_col}' not found in DataFrame")
                raise KeyError(f"Message column '{message_col}' not found in DataFrame")
                
            if replace_col not in self.df.columns:
                self._logger.error(f"Replacement column '{replace_col}' not found in DataFrame")
                raise KeyError(f"Replacement column '{replace_col}' not found in DataFrame")
                
            self.df[message_col] = self.df.apply(
                lambda row: row[message_col].replace(f"{{{replace_col}}}", str(row[replace_col])), 
                axis=1
            )

    def assemble_templates_from_df(
        self, 
        system_message_col: str = 'system_message', 
        user_message_col: str = 'user_message', 
        template_name_prefix: str = 'template'
    ) -> Union[dict[str, ChatPromptTemplate], None]:
        """
        Create prompt templates for each row in the DataFrame.
        
        Args:
            system_message_col: Column containing system messages
            user_message_col: Column containing user messages
            template_name_prefix: Prefix for template names
            
        Returns:
            Dictionary of ChatPromptTemplate objects
        """
        if (self.df is not None):
            if system_message_col not in self.df.columns:
                self._logger.error(f"System message column '{system_message_col}' not found in DataFrame")
                raise KeyError(f"System message column '{system_message_col}' not found in DataFrame")
                
            if user_message_col not in self.df.columns:
                self._logger.error(f"User message column '{user_message_col}' not found in DataFrame")
                raise KeyError(f"User message column '{user_message_col}' not found in DataFrame")
            
            templates = {}
            for index, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Building templates"):
                system_message = row[system_message_col]
                user_message = row[user_message_col]
                template_name = f"{template_name_prefix}{index}"               
                templates[template_name] = self.get_template_from_messages(
                    system_message=system_message,
                    user_message=user_message
                )
                
            self.templates = templates
            return templates

    @staticmethod
    def get_template_from_messages(system_message: str, user_message: str) -> ChatPromptTemplate:
        """
        Create a ChatPromptTemplate from system and user messages.
        
        Args:
            system_message: System message for the template
            user_message: User message for the template
            
        Returns:
            ChatPromptTemplate object
        """
        return ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", user_message)
        ])