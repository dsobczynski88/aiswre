import sys
import argparse
import re
from pathlib import Path
import logging
import pandas as pd
from dotenv import dotenv_values
from tqdm import tqdm
import asyncio
import nest_asyncio
from langchain_openai import ChatOpenAI
import src
from src.prj_logger import ProjectLogger, get_logs
from src import utils
import src.components.workflow as wf
import src.components.preprocess as pp
from src.components.incose import PreprocessIncoseGuide, BuildIncoseTemplates, IncoseRequirementReviewer
from src.components import prompteval as pe


if __name__ == "__main__":

    # load yaml config
    config = utils.load_yaml('config.yaml')
    if config is not None:
        globals().update(config)
    else:
        sys.exit(1)

    # load api key
    DOT_ENV = dotenv_values(".env")
    OPENAI_API_KEY = DOT_ENV['OPENAI_API_KEY']
    if not all([DOT_ENV, OPENAI_API_KEY]):
        sys.exit(1)
        
    # command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', required=True, type=str, help='The requirements dataset file name')
    parser.add_argument('--model', '-m', required=True, type=str, help='The string name of the LLM model to be used for revising the requirements')
    parser.add_argument('--template', '-t', required=True, type=str, help='The string name of the base template to be used, as defined in `preprocess.py`')
    parser.add_argument('--iternum', '-i', required=True, type=int, help='The maximum number of iterations to be run during the evaluation loop function (`run_eval_loop`)')
    args = parser.parse_args()
    
    # define constants and environment variables
    DATASET_FILE_PATH = args.data
    LLM_MODEL_NAME = args.model
    SELECTED_BASE_TEMPLATE_NAME = args.template
    MAX_NUM_ITER = args.iternum
    RUN_NAME = f"run-{utils.get_current_date_time()}"
    OUTPUT_DATA_FOLDER = f'{FILE_LOCATIONS['MAIN_DATA_FOLDER']}/{RUN_NAME}'
    Path(OUTPUT_DATA_FOLDER).mkdir(parents=True, exist_ok=True)
    
    # setup logging
    ProjectLogger(src.BASE_LOGGERNAME,f"{OUTPUT_DATA_FOLDER}/{src.BASE_LOGGERNAME}.log").config()
    proj_logger = logging.getLogger(src.BASE_LOGGERNAME)
    
    # load requirements dataset
    try:
        reqs_df = pd.read_excel(DATASET_FILE_PATH)    
    except FileNotFoundError:
        sys.exit(1)

    # load master results (or create if not exists)
    try:
        results_df = pd.read_excel(Path(FILE_LOCATIONS['MAIN_DATA_FOLDER']) / "results.xlsx")
    except FileNotFoundError:
        results_df_columns = [
            'run_id','dataset','model','template','iternum',
            f'%_resolved_initial_{REQUIREMENTS_DATASET_SETTINGS['FAILED_EVAL_COL']}',f'%_resolved_final_{REQUIREMENTS_DATASET_SETTINGS['FAILED_EVAL_COL']}',
        ]
        results_df = pd.DataFrame(columns=results_df_columns, index=[0])

    # parse INCOSE guide using the PreprocessIncoseGuide class
    incose_preprocessor = PreprocessIncoseGuide(INCOSE_GUIDE_SETTINGS['SECTIONS_REGEX_PAT']).preprocess_rules_section_4(
        inpath=Path(FILE_LOCATIONS['INCOSE_GUIDE']),
        outpath=Path(OUTPUT_DATA_FOLDER),
        start_page=65,
        end_page=115,
        replace_tokens=INCOSE_GUIDE_SETTINGS['REPLACE_TOKENS'],
        replace_with=INCOSE_GUIDE_SETTINGS['REPLACE_WITH']
    )
    incose_guide_sections_df = incose_preprocessor.df
    # load selected base template messages
    base_template_messages = pp.BASE_TEMPLATES[SELECTED_BASE_TEMPLATE_NAME]

    # build INCOSE templates from base messages via class BuildIncoseTemplates
    incose_template_builder = BuildIncoseTemplates(
        df=incose_guide_sections_df,
        base_messages=base_template_messages,
        output_data_folder_path=OUTPUT_DATA_FOLDER
    )
    # instantiate incose requirement reviewer
    incose_reviewer = IncoseRequirementReviewer(
        llm=ChatOpenAI(api_key=OPENAI_API_KEY, model=LLM_MODEL_NAME),
        use_structured_llm=False,
        pydantic_model=None,
        templates=incose_template_builder.templates,
        evals_config=incose_template_builder.evals_config,
        id_col=REQUIREMENTS_DATASET_SETTINGS['REQ_COLNAME']
    )

    # run initial evaluations on the requirements dataset
    reqs_df = incose_reviewer.run_eval_sequence(reqs_df, incose_reviewer.id_col, REQUIREMENTS_DATASET_SETTINGS['FAILED_EVAL_COL'], 'initial', None)

    # run evaluation loop
    wf.run_eval_loop(
        df=reqs_df,
        runner=incose_reviewer,
        output_data_folder=OUTPUT_DATA_FOLDER,
        eval_func_to_rule_id_map=None,#incose_template_builder.EVAL_TO_RULE_MAPPING,
        failed_eval_col=REQUIREMENTS_DATASET_SETTINGS['FAILED_EVAL_COL'],
        max_iter=MAX_NUM_ITER,
        capture_func=incose_template_builder.output_func
    )

    # generate revisions df
    revisions_df = wf.generate_revisions_df(
        op=OUTPUT_DATA_FOLDER,
        pat="revised_df*",
        requirement_col=REQUIREMENTS_DATASET_SETTINGS['REQ_COLNAME'],
        revision_number_col = 'revision'
    )

    # merge revisions df to original requirements
    reqs_df =wf.merge_revisions_df(
        op=OUTPUT_DATA_FOLDER,
        reqs_df=reqs_df, 
        revisions_df=revisions_df,
        requirement_col=REQUIREMENTS_DATASET_SETTINGS['REQ_COLNAME'],
        revision_number_col = 'revision'
    )

    # run final evaluations on the requirements dataset
    reqs_df = incose_reviewer.run_eval_sequence(reqs_df, f"Revised_{incose_reviewer.id_col}", REQUIREMENTS_DATASET_SETTINGS['REQ_COLNAME'], 'final', None)
    utils.to_excel(reqs_df, OUTPUT_DATA_FOLDER, False, 'reqs_df_with_revisions')
    wf.append_results(results_df, FILE_LOCATIONS['MAIN_DATA_FOLDER'], RUN_NAME, DATASET_FILE_PATH, LLM_MODEL_NAME, SELECTED_BASE_TEMPLATE_NAME, MAX_NUM_ITER, REQUIREMENTS_DATASET_SETTINGS['FAILED_EVAL_COL'], reqs_df)
