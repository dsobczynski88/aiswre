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
                 config: dict,
                 data: str,
                 model: str,
                 template: str,
                 iternum: int,
                 ):
        """
        ENTER DOCSTRING HERE
        """
        self.config = config
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
        self.revisions_df = None
        self.results_df = None
        self.proj_logger = logging.getLogger(BasicWorkflow.LOGGERNAME)
        
    def preprocess(self):
        # preprocess the incose guide section 4
        self.run_name = f"run-{utils.get_current_date_time()}"
        self.output_data_folder = f"{self.config['FILE_LOCATIONS']['MAIN_DATA_FOLDER']}/{self.run_name}"
        Path(self.output_data_folder).mkdir(parents=True, exist_ok=True) 

        self.incose_preprocessor = PreprocessIncoseGuide(
            self.config['INCOSE_GUIDE_SETTINGS']['SECTIONS_REGEX_PAT']).preprocess_rules_section_4(
            inpath=Path(self.config['FILE_LOCATIONS']['INCOSE_GUIDE']),
            outpath=Path(self.output_data_folder),
            start_page=65,
            end_page=115,
            replace_tokens=self.config['INCOSE_GUIDE_SETTINGS']['REPLACE_TOKENS'],
            subpatterns=self.config['INCOSE_GUIDE_SETTINGS']['SUBPATTERNS'],
            replace_with=self.config['INCOSE_GUIDE_SETTINGS']['REPLACE_WITH']
        )

        self.base_template_messages = self.config['BASE_PROMPT_TEMPLATES'][self.template]
        self.incose_template_builder = BuildIncoseTemplates(
            df=self.incose_preprocessor.df,
            base_messages=self.base_template_messages,
            output_data_folder_path=self.output_data_folder
        )
        self.incose_eval_config = BuildIncoseEvalConfig(
            incose_guide_df=self.incose_preprocessor.df,
            output_data_folder_path=self.output_data_folder,
            templates=self.incose_template_builder.templates,
            rule_to_eval_map=self.config['PROMPT_EVALUTION_CONFIG'],
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
            self.results_df = pd.read_excel(Path(self.config['FILE_LOCATIONS']['MAIN_DATA_FOLDER']) / "results.xlsx")
        except FileNotFoundError:
            results_df_columns = [
                'run_id','dataset','model','template','iternum',
                f"%_resolved_initial_{self.config['REQUIREMENTS_DATASET_SETTINGS']['FAILED_EVAL_COL']}",f"%_resolved_final_{self.config['REQUIREMENTS_DATASET_SETTINGS']['FAILED_EVAL_COL']}",
            ]
            self.results_df = pd.DataFrame(columns=results_df_columns, index=[0])

    def revise_requirements(self):
        self.incose_reviewer = IncoseRequirementReviewer(
            llm=ChatOllama(model=self.model),
            use_structured_llm=False,
            pydantic_model=None,
            templates=self.incose_template_builder.templates,
            evals_config=self.incose_eval_config.evals_config,
            id_col=self.config['REQUIREMENTS_DATASET_SETTINGS']['REQ_COLNAME']
        )
        self.incose_reviewer.run_eval_sequence(self.reqs_df, 
                                               f"{self.config['REQUIREMENTS_DATASET_SETTINGS']['REQ_COLNAME']}", 
                                               self.config['REQUIREMENTS_DATASET_SETTINGS']['FAILED_EVAL_COL'], 
                                               'initial', 
                                               self.eval_func_to_rule_id_map
                                               )
        self.run_eval_loop()
        self.generate_revisions_df()
        self.merge_revisions_df()
        self.reqs_df = self.incose_reviewer.run_eval_sequence(self.reqs_df, 
                                               f"Revised_{self.config['REQUIREMENTS_DATASET_SETTINGS']['REQ_COLNAME']}", 
                                               self.config['REQUIREMENTS_DATASET_SETTINGS']['FAILED_EVAL_COL'], 
                                               'final', 
                                               self.eval_func_to_rule_id_map
                                               )

    def save_output(self):
        utils.to_excel(self.reqs_df, self.output_data_folder, False, 'reqs_df_with_revisions')
        new_result_df = pd.DataFrame(data={
        'run_id': self.run_name,
        'dataset': self.data,
        'model': self.model,
        'template': self.template,
        'iternum': self.iternum,
        f"%_resolved_initial_{self.config['REQUIREMENTS_DATASET_SETTINGS']['FAILED_EVAL_COL']}": self.reqs_df[f"%_resolved_initial_{self.config['REQUIREMENTS_DATASET_SETTINGS']['FAILED_EVAL_COL']}"].iloc[0],
        f"%_resolved_final_{self.config['REQUIREMENTS_DATASET_SETTINGS']['FAILED_EVAL_COL']}": self.reqs_df[f"%_resolved_final_{self.config['REQUIREMENTS_DATASET_SETTINGS']['FAILED_EVAL_COL']}"].iloc[0]
        }, index=[0]
        )
        self.results_df = pd.concat([self.results_df, new_result_df], axis=0, ignore_index=False).dropna(subset=['run_id']).reset_index(drop=True)
        self.results_df.drop(columns=[c for c in self.results_df.columns if "Unnamed" in c], inplace=True)
        utils.to_excel(self.results_df, self.config["FILE_LOCATIONS"]["MAIN_DATA_FOLDER"], False, 'results_df')

    def run_eval_loop(self):
        # run evaluation algorithm
        df = self.reqs_df.copy()
        for iter in range(self.iternum):
            self.proj_logger.info(f'Entering: iter num {iter} of run_eval_loop')
            if iter > 0:
                df = pd.read_excel(f"{self.output_data_folder}/revised_df_iter_{iter-1}.xlsx")
                df = df.dropna(subset=[self.config['REQUIREMENTS_DATASET_SETTINGS']['REQ_COLNAME']])
            df = df[[self.config['REQUIREMENTS_DATASET_SETTINGS']['REQ_COLNAME'], f"{self.config['REQUIREMENTS_DATASET_SETTINGS']['REQ_COLNAME']}_#"]]
            self.proj_logger.info(f'Calling evaluations for iter num {iter} of run_eval_loop')
            # run evals on df
            df = self.incose_reviewer.run_eval_sequence(df, self.config['REQUIREMENTS_DATASET_SETTINGS']['REQ_COLNAME'], self.config['REQUIREMENTS_DATASET_SETTINGS']['FAILED_EVAL_COL'], None, self.eval_func_to_rule_id_map)
            self.proj_logger.info(f'Evaluations completed for iter num {iter} of run_eval_loop')
            if (df is not None):
                df = df.fillna('')
                pass_cond = df[self.config['REQUIREMENTS_DATASET_SETTINGS']['FAILED_EVAL_COL']].str.len() == 0
                pass_rows = df[pass_cond]
                prior_revision_df = df.copy()[[f"{self.config['REQUIREMENTS_DATASET_SETTINGS']['REQ_COLNAME']}_#", self.config['REQUIREMENTS_DATASET_SETTINGS']['REQ_COLNAME'], self.config['REQUIREMENTS_DATASET_SETTINGS']['FAILED_EVAL_COL']]]#, f"{self.config['REQUIREMENTS_DATASET_SETTINGS']['FAILED_EVAL_COL']}_rule_ids"]]
                renamed_columns = [f"{c}_prior_revision" for c in prior_revision_df.columns if c != f"{self.config['REQUIREMENTS_DATASET_SETTINGS']['REQ_COLNAME']}_#"]
                prior_revision_df.columns = [f"{self.config['REQUIREMENTS_DATASET_SETTINGS']['REQ_COLNAME']}_#"] + renamed_columns
                if len(pass_rows) > 0:
                    self.proj_logger.info(f'{len(pass_rows)}/{len(df)} Requirements passed all criteria during iter num {iter} of run_eval_loop')
                    df = df[~pass_cond]
                if len(df) > 0:
                    self.proj_logger.info(f'{len(df)} Requirements still require evaluation')
                    # run prompts for requirements containing failed evals
                    evals_lists = list(df[self.config['REQUIREMENTS_DATASET_SETTINGS']['FAILED_EVAL_COL']].values)
                    args_lists = list(df[self.config['REQUIREMENTS_DATASET_SETTINGS']['REQ_COLNAME']].values)
                    # run revision prompts
                    # the argument from the config file for BASE_PROMPT_TEMPLATES[self.template]["func"] is reading entry None as a string (currently commented out)
                    revised_df = self.incose_reviewer.revise(evals_lists, args_lists, None)#BASE_PROMPT_TEMPLATES[self.template]["func"])
                    revised_df['revision'] = iter + 1
                    revised_df.index = df.index
                    revised_df = revised_df.reset_index().rename(columns={'index':f"{self.config['REQUIREMENTS_DATASET_SETTINGS']['REQ_COLNAME']}_#"})
                    revised_df = pd.merge(
                        left=revised_df, right=prior_revision_df, on=f"{self.config['REQUIREMENTS_DATASET_SETTINGS']['REQ_COLNAME']}_#", how='inner'
                    )
                    # if any output from ai is blank, then use the previous revision
                    revised_df[self.config['REQUIREMENTS_DATASET_SETTINGS']['REQ_COLNAME']] = revised_df[self.config['REQUIREMENTS_DATASET_SETTINGS']['REQ_COLNAME']].fillna(prior_revision_df[f"{self.config['REQUIREMENTS_DATASET_SETTINGS']['REQ_COLNAME']}_prior_revision"]) 
                    utils.to_excel(revised_df, self.output_data_folder, str(iter), 'revised_df_iter')
                self.proj_logger.info(f'Exiting: iter num: {iter} of run_eval_loop')

    def generate_revisions_df(self, 
                              pat: str = 'revised_df*', 
                              revision_number_col: str = 'revision'
                              ):
        matching_files = list(Path(self.output_data_folder).rglob(pat))
        dfs=[]
        for file in matching_files:
            temp_df = pd.read_excel(file)
            temp_df = temp_df.rename(columns={self.config['REQUIREMENTS_DATASET_SETTINGS']['REQ_COLNAME']:f"Revised_{self.config['REQUIREMENTS_DATASET_SETTINGS']['REQ_COLNAME']}"}).drop(columns=['Unnamed: 0'])
            dfs.append(temp_df)
        # concat dfs
        self.revisions_df = pd.concat(dfs, ignore_index=True, axis=0)
        self.revisions_df = self.revisions_df[self.revisions_df[f"Revised_{self.config['REQUIREMENTS_DATASET_SETTINGS']['REQ_COLNAME']}"].str.strip() != '']
        utils.to_excel(self.revisions_df, self.output_data_folder, False, 'revisions_df')
        return self.revisions_df

    def merge_revisions_df(self,
                           revision_number_col='revision'
                           ):
        #merge latest revisions to original requirements dataframe
        self.revisions_df = self.revisions_df.sort_values(by=[f"{self.config['REQUIREMENTS_DATASET_SETTINGS']['REQ_COLNAME']}_#", revision_number_col], ascending=True).drop_duplicates(subset=[f"{self.config['REQUIREMENTS_DATASET_SETTINGS']['REQ_COLNAME']}_#"], keep='last').reset_index()
        self.reqs_df = pd.merge(
            left=self.reqs_df, right=self.revisions_df[[f"Revised_{self.config['REQUIREMENTS_DATASET_SETTINGS']['REQ_COLNAME']}",f"{self.config['REQUIREMENTS_DATASET_SETTINGS']['REQ_COLNAME']}_#"]], on=f"{self.config['REQUIREMENTS_DATASET_SETTINGS']['REQ_COLNAME']}_#", how='left'
        )
        return self.reqs_df