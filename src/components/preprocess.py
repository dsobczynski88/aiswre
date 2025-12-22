import re
import logging
from typing import List, Callable, Optional, Union, Dict
from functools import reduce
from pathlib import Path

import pandas as pd
import fitz  # PyMuPDF
from tqdm import tqdm
from langchain_core.prompts.chat import ChatPromptTemplate


class Sectionalize:
    """
    Class to load, extract, and segment a PDF into text sections based on regex patterns.

    Handles PDF text extraction and segmentation into structured sections for further processing.
    """

    LOGGER_NAME = "projectlog.Sectionalize"

    def __init__(self, regex: str) -> None:
        """
        Initialize Sectionalize.

        Args:
            regex: Regular expression pattern used to identify section boundaries.
        """
        self.regex: str = regex
        self.text: Optional[str] = None
        self.df: Optional[pd.DataFrame] = None
        self._logger = logging.getLogger(self.LOGGER_NAME)

    def get_pdf_text(self, fp: Path, start_page: int = 0, end_page: int = -1) -> str:
        """
        Extract text from a PDF file with an optional page range.

        Args:
            fp: Path to the PDF file.
            start_page: First page (0-indexed) to extract.
            end_page: Last page to extract (-1 for all pages).

        Returns:
            Extracted text from the PDF.
        """
        doc = fitz.open(fp)
        n_pages = doc.page_count
        if end_page == -1 or end_page > n_pages:
            end_page = n_pages

        texts: List[str] = []
        for i in range(start_page, end_page):
            page = doc.load_page(i)
            # Explicitly specify text extraction type for type-checkers
            texts.append(page.get_text("text"))

        self.text = chr(12).join(texts)
        self._logger.info(f"Extracted text from pages {start_page}–{end_page}")
        return self.text

    def get_sections_df(self) -> pd.DataFrame:
        """
        Build a DataFrame with section indices and matches according to regex.

        Returns:
            DataFrame containing section boundaries and matched text.

        Raises:
            ValueError: If no text is loaded prior to calling this method.
        """
        if not self.text:
            self._logger.error("No text loaded. Call get_pdf_text() first.")
            raise ValueError("No text loaded. Call get_pdf_text() first.")

        matches = [
            {"start": m.start(1), "end": m.end(1), "match": m.group(1), "full_match": m.group(0)}
            for m in re.finditer(self.regex, self.text)
        ]

        self.df = pd.DataFrame(matches)
        self._logger.info(f"Found {len(matches)} sections based on regex.")
        return self.df

    def add_section_text(
        self,
        match_start_col: str = "start",
        match_end_col: str = "end",
        ) -> pd.DataFrame:
        """
        Add the extracted section text to the DataFrame between boundaries.

        Args:
            match_start_col: Column name for section start indices.
            match_end_col: Column name for section end indices.

        Returns:
            Updated DataFrame with an added 'extract' column.

        Raises:
            ValueError: If no DataFrame or text is available.
        """
        if self.df is None:
            raise ValueError("No sections DataFrame. Call get_sections_df() first.")
        if not self.text:
            raise ValueError("No text loaded. Call get_pdf_text() first.")

        start_idx_col = f"{match_start_col}_idx"
        end_idx_col = f"{match_end_col}_idx"

        self.df[start_idx_col] = self.df[match_end_col].astype(int)
        self.df[end_idx_col] = (
            self.df[match_start_col].astype(int).shift(-1).fillna(len(self.text))
        )

        def safe_extract(row) -> str:
            try:
                start_i, end_i = int(row[start_idx_col]), int(row[end_idx_col])
                return self.text[start_i:end_i]
            except Exception as e:
                self._logger.warning(f"Section extraction failed: {e}")
                return ""

        self.df["extract"] = self.df.apply(safe_extract, axis=1)
        self._logger.info("Added 'extract' column to section DataFrame.")
        return self.df

    def save_text(self, output_path: Path) -> None:
        """
        Save extracted text to a file.

        Args:
            output_path: Path where text will be saved.

        Raises:
            ValueError: If no text has been extracted yet.
        """
        if not self.text:
            raise ValueError("No text to save. Call get_pdf_text() first.")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(self.text)

        self._logger.info(f"Text saved to {output_path}")

    def print_text(self, start_index: int = 0, end_index: int = -1) -> None:
        """
        Print a slice of the extracted text for inspection.

        Args:
            start_index: Start index of slice.
            end_index: End index (-1 for full text end).
        """
        if not self.text:
            raise ValueError("No text to print. Call get_pdf_text() first.")
        content = self.text[start_index:end_index] if end_index != -1 else self.text[start_index:]
        print(content)


class TextPreprocessor:
    """
    Preprocess text using configurable cleaning pipeline of transformation functions.
    """

    LOGGER_NAME = "projectlog.TextPreprocessor"

    def __init__(self, pipeline: Optional[List[Callable[[str], str]]] = None) -> None:
        """
        Initialize TextPreprocessor.

        Args:
            pipeline: Optional list of callables for preprocessing.
        """
        self._pipeline: List[Callable[[str], str]] = pipeline or []
        self._logger = logging.getLogger(self.LOGGER_NAME)

    @property
    def pipeline(self) -> List[Callable[[str], str]]:
        """Return current text processing pipeline."""
        return self._pipeline

    @pipeline.setter
    def pipeline(self, new_pipeline: List[Callable[[str], str]]) -> None:
        """Replace the current text processing pipeline."""
        self._pipeline = new_pipeline

    def add_processor(self, processor_func: Callable[[str], str]) -> None:
        """Add a processor function to the pipeline."""
        self._pipeline.append(processor_func)

    def clean_text(self, text: str) -> str:
        """
        Apply all pipeline functions to input text sequentially.

        Args:
            text: Input text.

        Returns:
            Processed text.
        """
        if not self._pipeline:
            self._logger.warning("Empty pipeline — returning text unchanged.")
            return text
        return reduce(lambda x, func: func(x), self._pipeline, text)

    def clean_frame(self, df: pd.DataFrame, apply_to_cols: List[str]) -> pd.DataFrame:
        """
        Apply cleaning pipeline to specific DataFrame columns.

        Args:
            df: Input DataFrame.
            apply_to_cols: Columns to process.

        Returns:
            DataFrame with cleaned text columns.
        """
        from src import utils  # Avoid circular import

        result_df = df.copy()
        for col in apply_to_cols:
            if col not in result_df.columns:
                self._logger.warning(f"Column '{col}' not in DataFrame — skipping.")
                continue
            result_df = utils.replace_null(result_df, col, " ")
            result_df[col] = result_df[col].astype(str).apply(self.clean_text)
        return result_df

    # Common static text utilities

    @staticmethod
    def replace(text: str, replace_tokens: List[str], replace_with: str) -> str:
        for token in replace_tokens:
            text = text.replace(token, replace_with)
        return text

    @staticmethod
    def resub(text: str, patterns: List[str], replace_with: str, flags: int = re.DOTALL) -> str:
        for pattern in patterns:
            text = re.sub(pattern, replace_with, text, flags=flags)
        return text

    @staticmethod
    def remove_multi_whitespace(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def make_lower(text: str) -> str:
        return text.lower()

    @staticmethod
    def remove_stopwords(text: str, stop_words: List[str]) -> str:
        if not stop_words:
            return text
        return " ".join(word for word in text.split() if word.lower() not in stop_words)


class BuildTemplates:
    """
    Generate LangChain ChatPromptTemplates from a DataFrame and base messages.
    """

    LOGGER_NAME = "projectlog.BuildTemplates"

    def __init__(self, df: Optional[pd.DataFrame], base_messages: Optional[Dict[str, str]]) -> None:
        """
        Initialize BuildTemplates.

        Args:
            df: DataFrame with data for template variables.
            base_messages: Dict containing system/user base message strings.
        """
        self.df = df.copy() if df is not None else None
        self.base_messages = base_messages or {}
        self.templates: Dict[str, ChatPromptTemplate] = {}
        self._logger = logging.getLogger(self.LOGGER_NAME)

    def add_message_col_to_frame(self, message_name: str) -> None:
        """
        Add a message column to the DataFrame using a base template.
        """
        if self.df is None:
            return
        if message_name not in self.base_messages:
            raise KeyError(f"Message '{message_name}' not found in base_messages.")
        self.df[f"{message_name}_message"] = self.base_messages[message_name]

    def replace_prompt_variables_from_frame(self, message_col: str, replace_col: str) -> None:
        """
        Replace placeholders in message templates with DataFrame values.
        """
        if self.df is None:
            return
        if message_col not in self.df or replace_col not in self.df:
            raise KeyError(f"Missing columns '{message_col}' or '{replace_col}'.")
        self.df[message_col] = self.df.apply(
            lambda row: row[message_col].replace(f"{{{replace_col}}}", str(row[replace_col])),
            axis=1,
        )

    def assemble_templates_from_df(
        self,
        system_message_col: str = "system_message",
        user_message_col: str = "user_message",
        template_name_prefix: str = "template",
        ) -> Optional[Dict[str, ChatPromptTemplate]]:
        """
        Build ChatPromptTemplate objects for each row.
        """
        if self.df is None:
            return None
        if system_message_col not in self.df.columns or user_message_col not in self.df.columns:
            raise KeyError("Missing required message columns in DataFrame.")

        templates: Dict[str, ChatPromptTemplate] = {}
        for index, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Building templates"):
            template_name = f"{template_name_prefix}{index}"
            templates[template_name] = self.get_template_from_messages(
                system_message=row[system_message_col],
                user_message=row[user_message_col],
            )
        self.templates = templates
        self._logger.info(f"Built {len(templates)} templates successfully.")
        return templates

    @staticmethod
    def get_template_from_messages(system_message: str, user_message: str) -> ChatPromptTemplate:
        """Return a ChatPromptTemplate from system and user messages."""
        return ChatPromptTemplate.from_messages([("system", system_message), ("human", user_message)])