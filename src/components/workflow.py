import logging
from pathlib import Path
import pandas as pd
from langchain_ollama import ChatOllama, OllamaEmbeddings
import src
from src import utils
from src.prj_logger import get_logs
from src.components.incose import (
    PreprocessIncoseGuide, 
    BuildIncoseEvalConfig, 
    BuildIncoseTemplates, 
    IncoseRequirementReviewer
)


class BasicWorkflow:
    
    LOGGERNAME = f"{src.BASE_LOGGERNAME}.workflow"
    
    def __init__(self,
                 config_file: str,
                 data: str,
                 model: str,
                 template: str,
                 iternum: int,
                 ):
        """
        ENTER DOCSTRING HERE
        """
        self.config_file = config_file
        self.data = data
        self.model = model
        self.template = template
        self.iternum = iternum
        self.incose_preprocessor = None
        self.incose_template_builder = None
        self.incose_reviewer = None
        self.base_template_messages = None
        self.run_name = None
        self.output_data_folder = None
        self.eval_func_to_rule_id_map = None
        self.reqs_df = None
        self.results_df = None
        self.proj_logger = logging.getLogger(BasicWorkflow.LOGGERNAME)

    def load_config(self):
        # load config
        config = utils.load_yaml(self.config_file)
        if config is not None:
            globals().update(config)
        else:
            raise
    
    def preprocess(self):
        # preprocess the incose guide section 4
        self.run_name = f"run-{utils.get_current_date_time()}"
        self.output_data_folder = f"{FILE_LOCATIONS['MAIN_DATA_FOLDER']}/{self.run_name}"
        Path(self.output_data_folder).mkdir(parents=True, exist_ok=True) 

        self.incose_preprocessor = PreprocessIncoseGuide(
            INCOSE_GUIDE_SETTINGS['SECTIONS_REGEX_PAT']).preprocess_rules_section_4(
            inpath=Path(FILE_LOCATIONS['INCOSE_GUIDE']),
            outpath=Path(self.output_data_folder),
            start_page=65,
            end_page=115,
            replace_tokens=INCOSE_GUIDE_SETTINGS['REPLACE_TOKENS'],
            subpatterns=INCOSE_GUIDE_SETTINGS['SUBPATTERNS'],
            replace_with=INCOSE_GUIDE_SETTINGS['REPLACE_WITH']
        )

        self.base_template_messages = BASE_PROMPT_TEMPLATES[self.template]
        self.incose_template_builder = BuildIncoseTemplates(
            df=self.incose_preprocessor.df,
            base_messages=self.base_template_messages,
            output_data_folder_path=self.output_data_folder
        )
        self.incose_eval_config = BuildIncoseEvalConfig(
            incose_guide_df=self.incose_preprocessor.df,
            output_data_folder_path=self.output_data_folder,
            templates=self.incose_template_builder.templates,
            rule_to_eval_map=PROMPT_EVALUTION_CONFIG,
            rule_num_col='rule_number'
        )

    def load_requirements(self):
        # load requirements dataset
        try:
            self.reqs_df = pd.read_excel(self.data)    
        except FileNotFoundError:
            raise

        # load master results (or create if not exists)
        try:
            self.results_df = pd.read_excel(Path(FILE_LOCATIONS['MAIN_DATA_FOLDER']) / "results.xlsx")
        except FileNotFoundError:
            results_df_columns = [
                'run_id','dataset','model','template','iternum',
                f"%_resolved_initial_{REQUIREMENTS_DATASET_SETTINGS['FAILED_EVAL_COL']}",f"%_resolved_final_{REQUIREMENTS_DATASET_SETTINGS['FAILED_EVAL_COL']}",
            ]
            self.results_df = pd.DataFrame(columns=results_df_columns, index=[0])


    def revise_requirements(self):
        
        self.incose_reviewer = IncoseRequirementReviewer(
            llm=ChatOllama(model=self.model),
            use_structured_llm=False,
            pydantic_model=None,
            templates=self.incose_template_builder.templates,
            evals_config=self.incose_eval_config.evals_config,
            id_col=REQUIREMENTS_DATASET_SETTINGS['REQ_COLNAME']
        )
        self.run_eval_loop()

    def run_eval_loop(self):
        # run evaluation algorithm
        df = self.reqs_df.copy()
        for iter in range(self.iternum):
            self.proj_logger.info(f'Entering: iter num {iter} of run_eval_loop')
            if iter > 0:
                df = pd.read_excel(f"{self.output_data_folder}/revised_df_iter_{iter-1}.xlsx")
                df = df.dropna(subset=[self.incose_reviewer.id_col])
            df = df[[self.incose_reviewer.id_col, f"{self.incose_reviewer.id_col}_#"]]
            self.proj_logger.info(f'Calling evaluations for iter num {iter} of run_eval_loop')
            # run evals on df
            df = self.incose_reviewer.run_eval_sequence(df, self.incose_reviewer.id_col, REQUIREMENTS_DATASET_SETTINGS['FAILED_EVAL_COL'], None, self.eval_func_to_rule_id_map)
            self.proj_logger.info(f'Evaluations completed for iter num {iter} of run_eval_loop')
            if (df is not None):
                df = df.fillna('')
                pass_cond = df[REQUIREMENTS_DATASET_SETTINGS['FAILED_EVAL_COL']].str.len() == 0
                pass_rows = df[pass_cond]
                prior_revision_df = df.copy()[[f"{self.incose_reviewer.id_col}_#", self.incose_reviewer.id_col, REQUIREMENTS_DATASET_SETTINGS['FAILED_EVAL_COL']]]#, f"{REQUIREMENTS_DATASET_SETTINGS['FAILED_EVAL_COL']}_rule_ids"]]
                renamed_columns = [f"{c}_prior_revision" for c in prior_revision_df.columns if c != f"{self.incose_reviewer.id_col}_#"]
                prior_revision_df.columns = [f"{self.incose_reviewer.id_col}_#"] + renamed_columns
                if len(pass_rows) > 0:
                    self.proj_logger.info(f'{len(pass_rows)}/{len(df)} Requirements passed all criteria during iter num {iter} of run_eval_loop')
                    df = df[~pass_cond]
                if len(df) > 0:
                    self.proj_logger.info(f'{len(df)} Requirements still require evaluation')
                    # run prompts for requirements containing failed evals
                    evals_lists = list(df[REQUIREMENTS_DATASET_SETTINGS['FAILED_EVAL_COL']].values)
                    args_lists = list(df[self.incose_reviewer.id_col].values)
                    # run revision prompts
                    revised_df = self.incose_reviewer.revise(evals_lists, args_lists, None)#BASE_PROMPT_TEMPLATES[self.template]["func"])
                    revised_df['revision'] = iter + 1
                    revised_df.index = df.index
                    revised_df = revised_df.reset_index().rename(columns={'index':f"{self.incose_reviewer.id_col}_#"})
                    revised_df = pd.merge(
                        left=revised_df, right=prior_revision_df, on=f"{self.incose_reviewer.id_col}_#", how='inner'
                    )
                    # if any output from ai is blank, then use the previous revision
                    revised_df[self.incose_reviewer.id_col] = revised_df[self.incose_reviewer.id_col].fillna(prior_revision_df[f"{self.incose_reviewer.id_col}_prior_revision"]) 
                    utils.to_excel(revised_df, self.output_data_folder, str(iter), 'revised_df_iter')
                self.proj_logger.info(f'Exiting: iter num: {iter} of run_eval_loop')

    def save_output(self):
        pass

"""
def append_results(results_df, output_fp, run_id, dataset, model, template, iternum, REQUIREMENTS_DATASET_SETTINGS['FAILED_EVAL_COL'],reqs_df):
    new_result_df = pd.DataFrame(data={
        'run_id': run_id,
        'dataset': dataset,
        'model': model,
        'template': template,
        'iternum': iternum,
        f'%_resolved_initial_{REQUIREMENTS_DATASET_SETTINGS['FAILED_EVAL_COL']}': reqs_df[f'%_resolved_initial_{REQUIREMENTS_DATASET_SETTINGS['FAILED_EVAL_COL']}'].iloc[0],
        f'%_resolved_final_{REQUIREMENTS_DATASET_SETTINGS['FAILED_EVAL_COL']}': reqs_df[f'%_resolved_final_{REQUIREMENTS_DATASET_SETTINGS['FAILED_EVAL_COL']}'].iloc[0]
    }, index=[0]
    )
    results_df = pd.concat([results_df, new_result_df], axis=0, ignore_index=False).dropna(subset=['run_id']).reset_index(drop=True)
    results_df.drop(columns=[c for c in results_df.columns if "Unnamed" in c], inplace=True)
    utils.to_excel(results_df, output_fp, False, 'results')     


def run_eval_loop(df, self.incose_reviewer, output_data_folder, eval_func_to_rule_id_map, REQUIREMENTS_DATASET_SETTINGS['FAILED_EVAL_COL']='failed_evals', max_iter=3, capture_func=None):
    # run evaluation algorithm
    for iter in range(max_iter):
        self.proj_logger.info(f'Entering: iter num {iter} of run_eval_loop')
        if iter > 0:
            df = pd.read_excel(f"{output_data_folder}/revised_df_iter_{iter-1}.xlsx")
            df = df.dropna(subset=[self.incose_reviewer.id_col])
        df = df[[self.incose_reviewer.id_col, f"{self.incose_reviewer.id_col}_#"]]
        self.proj_logger.info(f'Calling evaluations for iter num {iter} of run_eval_loop')
        # run evals on df
        df = self.incose_reviewer.run_eval_sequence(df, self.incose_reviewer.id_col, REQUIREMENTS_DATASET_SETTINGS['FAILED_EVAL_COL'], None, eval_func_to_rule_id_map)
        self.proj_logger.info(f'Evaluations completed for iter num {iter} of run_eval_loop')
        if (df is not None):
            df = df.fillna('')
            pass_cond = df[REQUIREMENTS_DATASET_SETTINGS['FAILED_EVAL_COL']].str.len() == 0
            pass_rows = df[pass_cond]
            prior_revision_df = df.copy()[[f"{self.incose_reviewer.id_col}_#", self.incose_reviewer.id_col, REQUIREMENTS_DATASET_SETTINGS['FAILED_EVAL_COL']]]#, f"{REQUIREMENTS_DATASET_SETTINGS['FAILED_EVAL_COL']}_rule_ids"]]
            renamed_columns = [f"{c}_prior_revision" for c in prior_revision_df.columns if c != f"{self.incose_reviewer.id_col}_#"]
            prior_revision_df.columns = [f"{self.incose_reviewer.id_col}_#"] + renamed_columns
            if len(pass_rows) > 0:
                self.proj_logger.info(f'{len(pass_rows)}/{len(df)} Requirements passed all criteria during iter num {iter} of run_eval_loop')
                df = df[~pass_cond]
            if len(df) > 0:
                self.proj_logger.info(f'{len(df)} Requirements still require evaluation')
                # run prompts for requirements containing failed evals
                evals_lists = list(df[REQUIREMENTS_DATASET_SETTINGS['FAILED_EVAL_COL']].values)
                args_lists = list(df[self.incose_reviewer.id_col].values)
                # run revision prompts
                revised_df = self.incose_reviewer.revise(evals_lists, args_lists, capture_func)
                revised_df['revision'] = iter + 1
                revised_df.index = df.index
                revised_df = revised_df.reset_index().rename(columns={'index':f"{self.incose_reviewer.id_col}_#"})
                revised_df = pd.merge(
                    left=revised_df, right=prior_revision_df, on=f"{self.incose_reviewer.id_col}_#", how='inner'
                )
                # if any output from ai is blank, then use the previous revision
                revised_df[self.incose_reviewer.id_col] = revised_df[self.incose_reviewer.id_col].fillna(prior_revision_df[f"{self.incose_reviewer.id_col}_prior_revision"]) 
                utils.to_excel(revised_df, output_data_folder, str(iter), 'revised_df_iter')
            self.proj_logger.info(f'Exiting: iter num: {iter} of run_eval_loop')


@get_logs(src.BASE_LOGGERNAME)
def generate_revisions_df(op: str, pat: str, requirement_col: str = 'Requirement', revision_number_col: str = 'revision'):
    directory = Path(op)
    matching_files = list(directory.rglob(pat))
    dfs=[]
    for file in matching_files:
        temp_df = pd.read_excel(file)
        temp_df = temp_df.rename(columns={requirement_col:f'Revised_{requirement_col}'}).drop(columns=['Unnamed: 0'])
        dfs.append(temp_df)
    # concat dfs
    revisions_df = pd.concat(dfs, ignore_index=True, axis=0)#[[f'Revised_{requirement_col}',f'{requirement_col}_#','revision']]
    revisions_df = revisions_df[revisions_df[f'Revised_{requirement_col}'].str.strip() != '']
    utils.to_excel(revisions_df, op, False, 'revisions_df')
    return revisions_df

@get_logs(src.BASE_LOGGERNAME)
def merge_revisions_df(op, reqs_df, revisions_df, requirement_col='Requirement', revision_number_col='revision'):
    #merge latest revisions to original requirements dataframe
    revisions_df = revisions_df.sort_values(by=[f'{requirement_col}_#', revision_number_col], ascending=True).drop_duplicates(subset=[f'{requirement_col}_#'], keep='last').reset_index()
    reqs_df = pd.merge(
        left=reqs_df, right=revisions_df[[f'Revised_{requirement_col}',f'{requirement_col}_#']], on=f'{requirement_col}_#', how='left'
    )
    return reqs_df
"""