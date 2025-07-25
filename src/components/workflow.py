import logging
from pathlib import Path
import pandas as pd
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
    #proj_logger = logging.getLogger(LOGGERNAME)

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

    def revise(self):
        pass
    
    def save_output(self):
        pass

"""
def append_results(results_df, output_fp, run_id, dataset, model, template, iternum, failed_eval_col,reqs_df):
    new_result_df = pd.DataFrame(data={
        'run_id': run_id,
        'dataset': dataset,
        'model': model,
        'template': template,
        'iternum': iternum,
        f'%_resolved_initial_{failed_eval_col}': reqs_df[f'%_resolved_initial_{failed_eval_col}'].iloc[0],
        f'%_resolved_final_{failed_eval_col}': reqs_df[f'%_resolved_final_{failed_eval_col}'].iloc[0]
    }, index=[0]
    )
    results_df = pd.concat([results_df, new_result_df], axis=0, ignore_index=False).dropna(subset=['run_id']).reset_index(drop=True)
    results_df.drop(columns=[c for c in results_df.columns if "Unnamed" in c], inplace=True)
    utils.to_excel(results_df, output_fp, False, 'results')     


def run_eval_loop(df, runner, output_data_folder, eval_func_to_rule_id_map, failed_eval_col='failed_evals', max_iter=3, capture_func=None):
    # run evaluation algorithm
    for iter in range(max_iter):
        proj_logger.info(f'Entering: iter num {iter} of run_eval_loop')
        if iter > 0:
            df = pd.read_excel(f"{output_data_folder}/revised_df_iter_{iter-1}.xlsx")
            df = df.dropna(subset=[runner.id_col])
        df = df[[runner.id_col, f"{runner.id_col}_#"]]
        proj_logger.info(f'Calling evaluations for iter num {iter} of run_eval_loop')
        # run evals on df
        df = runner.run_eval_sequence(df, runner.id_col, failed_eval_col, None, eval_func_to_rule_id_map)
        proj_logger.info(f'Evaluations completed for iter num {iter} of run_eval_loop')
        if (df is not None):
            df = df.fillna('')
            pass_cond = df[failed_eval_col].str.len() == 0
            pass_rows = df[pass_cond]
            prior_revision_df = df.copy()[[f"{runner.id_col}_#", runner.id_col, failed_eval_col]]#, f"{failed_eval_col}_rule_ids"]]
            renamed_columns = [f"{c}_prior_revision" for c in prior_revision_df.columns if c != f"{runner.id_col}_#"]
            prior_revision_df.columns = [f"{runner.id_col}_#"] + renamed_columns
            if len(pass_rows) > 0:
                proj_logger.info(f'{len(pass_rows)}/{len(df)} Requirements passed all criteria during iter num {iter} of run_eval_loop')
                df = df[~pass_cond]
            if len(df) > 0:
                proj_logger.info(f'{len(df)} Requirements still require evaluation')
                # run prompts for requirements containing failed evals
                evals_lists = list(df[failed_eval_col].values)
                args_lists = list(df[runner.id_col].values)
                # run revision prompts
                revised_df = runner.revise(evals_lists, args_lists, capture_func)
                revised_df['revision'] = iter + 1
                revised_df.index = df.index
                revised_df = revised_df.reset_index().rename(columns={'index':f"{runner.id_col}_#"})
                revised_df = pd.merge(
                    left=revised_df, right=prior_revision_df, on=f"{runner.id_col}_#", how='inner'
                )
                # if any output from ai is blank, then use the previous revision
                revised_df[runner.id_col] = revised_df[runner.id_col].fillna(prior_revision_df[f"{runner.id_col}_prior_revision"]) 
                utils.to_excel(revised_df, output_data_folder, str(iter), 'revised_df_iter')
            proj_logger.info(f'Exiting: iter num: {iter} of run_eval_loop')


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