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
import src
from src.prj_logger import ProjectLogger, get_logs
from src import utils
from src.components.workflow import BasicWorkflow

if __name__ == "__main__":

    config = utils.load_config()
    output_directory = utils.make_output_directory(config["FILE_LOCATIONS"])
    ProjectLogger(src.BASE_LOGGERNAME,f"{output_directory}/{src.BASE_LOGGERNAME}.log").config()
    
    wf = BasicWorkflow(
        config=config,
        data=config['FILE_LOCATIONS']['DATASET_FILE_PATH'],
        model=config['MODEL'],
        template=config['SELECTED_BASE_TEMPLATE'],
        iternum=config['ITERNUM'],
    )
    wf.preprocess_data()
    wf.load_requirements()
    wf.revise_requirements()
    wf.save_output()