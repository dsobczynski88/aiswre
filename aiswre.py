import sys
import argparse
import re
from pprint import pformat
from pathlib import Path
import logging
import pandas as pd
from dotenv import dotenv_values
from tqdm import tqdm
import asyncio
import nest_asyncio
#from langchain_openai import ChatOpenAI
import src
from src.prj_logger import ProjectLogger, get_logs
from src import utils
from src.components.workflow import BasicWorkflow

if __name__ == "__main__":

    config = utils.load_config()
    print(pformat(config))
    output_directory = utils.make_output_directory(config["FILE_LOCATIONS"])
    ProjectLogger(src.BASE_LOGGERNAME,f"{output_directory}/{src.BASE_LOGGERNAME}.log").config()
    
    wf = BasicWorkflow(
        config=config,
        data='src/data/demo_dataset.xlsx',
        model='llama3.1',
        template='req-reviewer-instruct-2',
        iternum=2,
    )
    wf.preprocess()
    wf.load_requirements()
    wf.revise_requirements()
    wf.save_output()

    # print head of preprocessed incose guide
    #wf.incose_preprocessor.df.to_excel('df.xlsx')
    #print(wf.incose_preprocessor.df.head(5))
    #print(wf.incose_preprocessor.df.columns)
    #print the R3 system message template
    #print(pformat(wf.incose_template_builder.templates['R3'].messages[0].prompt.template))
    #print(wf.incose_eval_config.evals_config)