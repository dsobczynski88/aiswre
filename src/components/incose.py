"""
This module includes functions specifically designed to preprocess the INCOSE guide,
build requirement revision prompt templates, and run prompts for requirement revision.
"""

from __future__ import annotations

import re
import logging
import asyncio
from pathlib import Path
from functools import partial
from typing import List, Dict, Optional, Any, Callable

import pandas as pd
import nest_asyncio
from pprint import pformat
from tqdm import tqdm
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
    RunnableSequence,
)

import src
from src.prj_logger import get_logs
from src import utils
from src.utils import map_A_to_B
import src.components.prompteval as pe
from src.components.promptrunner import PromptRunner
from src.components.preprocess import TextPreprocessor, Sectionalize, BuildTemplates

nest_asyncio.apply()


# -------------------------------------------------------------------------
# -------------------------- PREPROCESSING --------------------------------
# -------------------------------------------------------------------------
def preprocess_incose_guide(
    input_path: Path,
    output_path: Path,
    start_page: int,
    end_page: int,
    regex: str,
    replace_tokens: List[str],
    subpatterns: List[str],
    replace_with: str,
    ) -> pd.DataFrame:
    """
    Preprocess the INCOSE guide PDF to extract and structure rule information.
    """
    # Create base objects using composition
    sectionalize = Sectionalize(regex)
    preprocessor = TextPreprocessor()

    # Extract text from PDF
    text = sectionalize.get_pdf_text(input_path, start_page, end_page)
    sectionalize.text = text or ""  # Ensure non‑None string

    # Save raw extracted text
    sectionalize.save_text(output_path / "extract.txt")

    # Set up preprocessing pipeline
    preprocessor.pipeline = [
        partial(TextPreprocessor.replace, replace_tokens=replace_tokens, replace_with=replace_with),
        partial(TextPreprocessor.resub, patterns=subpatterns, replace_with=replace_with),
        TextPreprocessor.remove_multi_whitespace,
    ]

    # Apply preprocessing to text safely
    sectionalize.text = preprocessor.clean_text(sectionalize.text or "")

    # Save cleaned text
    sectionalize.save_text(output_path / "extract-post-clean.txt")

    # Extract sections
    df = sectionalize.get_sections_df() or pd.DataFrame()
    df = sectionalize.add_section_text() or df

    # Extract rule information
    df = extract_incose_rule_info(df)
    return df


# -------------------------------------------------------------------------
# ----------------------- EXTRACT RULE INFO -------------------------------
# -------------------------------------------------------------------------
def extract_incose_rule_info(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract structured information from INCOSE guide sections.
    """
    # Extract rule number
    df["rule_number"] = df["extract"].apply(
        lambda s: (m.group(1) if (m := re.search(r"^\s*(R\d+)\s*–", s)) else None)
    )

    # Extract rule title
    df["rule_title"] = df["extract"].apply(
        lambda s: (m.group(1) if (m := re.search(r"^\s*R\d+\s*–\s*([A-Z\W]+)\s+Definition", s)) else None)
    )

    # Extract definition, elaboration, examples
    df["definition"] = df["extract"].apply(
        lambda s: "".join(re.findall(r"Definition:([\s\S]+?)(?=Elaboration:)", s))
    )
    df["elaboration"] = df["extract"].apply(
        lambda s: "".join(re.findall(r"Elaboration:([\s\S]+?)(?=Examples:)", s))
    )
    df["examples"] = df["extract"].apply(
        lambda s: "".join(re.findall(r"Examples:(.*)$", s, flags=re.DOTALL))
    )

    # Clean examples
    df["examples"] = (
        df["examples"]
        .apply(lambda s: re.sub(r"\[[^\]]+\]", "", s, flags=re.DOTALL))
        .apply(lambda s: re.sub(r"(Exceptions and relationships:.*)$", "", s, flags=re.DOTALL))
    )
    return df


# -------------------------------------------------------------------------
# --------------------- BUILD PROMPT TEMPLATES ----------------------------
# -------------------------------------------------------------------------
def build_incose_templates(
    incose_df: pd.DataFrame,
    base_messages: Dict[str, str],
    output_folder_path: Path,
    ) -> Dict[str, Any]:
    """
    Build prompt templates based on INCOSE guide rules.
    """
    template_builder = BuildTemplates(incose_df, base_messages)

    # Add message columns
    template_builder.add_message_col_to_frame("system")
    template_builder.add_message_col_to_frame("user")

    # Replace template variables with INCOSE data
    template_builder.replace_prompt_variables_from_frame("user_message", "definition")
    template_builder.replace_prompt_variables_from_frame("user_message", "examples")

    # Build templates
    templates = (
        template_builder.assemble_templates_from_df(
            system_message_col="system_message",
            user_message_col="user_message",
            template_name_prefix="R",
        )
        or {}
    )

    return templates


# -------------------------------------------------------------------------
# --------------------------- EVAL CONFIG ---------------------------------
# -------------------------------------------------------------------------
def build_incose_eval_config(
    incose_df: pd.DataFrame,
    output_folder_path: Path,
    templates: Dict[str, Any],
    rule_to_eval_map: Dict[str, str],
    rule_num_col: str = "rule_number",
    ) -> Dict[str, Dict[str, Any]]:
    """
    Create mapping between evaluation functions and INCOSE rules.
    """
    evals_config: Dict[str, Dict[str, Any]] = {}

    for rule, eval_name in rule_to_eval_map.items():
        if rule not in templates:
            continue
        eval_func = getattr(pe, eval_name, None)
        if eval_func is None:
            continue
        evals_config[eval_name] = {"func": eval_func, "template": templates[rule]}

    # Save config for traceability
    output_file = Path(output_folder_path) / "evals_config.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(pformat(evals_config))

    return evals_config