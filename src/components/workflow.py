import logging
from pathlib import Path
import pandas as pd
from langchain_ollama import ChatOllama, OllamaEmbeddings
import src
from src import utils
from src.prj_logger import get_logs
from src.components.incose import (
    preprocess_incose_guide,
    build_incose_templates,
    build_incose_eval_config,
    IncoseRequirementReviewer,
    extract_incose_rule_info
)


class BasicWorkflow:
    """
    A high-level workflow for processing and revising requirements using the INCOSE guide.
    
    This class orchestrates the entire workflow from preprocessing the INCOSE guide,
    building templates, evaluating requirements, and generating revisions.
    """
    LOGGER_NAME = f"{src.BASE_LOGGERNAME}.workflow"
    
    def __init__(self, config: dict):
        """
        Initialize the BasicWorkflow with configuration settings.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        self.data = self.config['FILE_LOCATIONS']['DATASET_FILE_PATH']
        self.model = self.config['MODEL']
        self.template = self.config['SELECTED_BASE_TEMPLATE']
        self.iternum = self.config['ITERNUM']
        
        # Data containers
        self.incose_df = None
        self.templates = None
        self.evals_config = None
        self.incose_reviewer = None
        self.base_template_messages = None
        self.run_name = None
        self.output_data_folder = None
        self.eval_func_to_rule_id_map = None
        self.reqs_df = None
        self.revisions_df = None
        self.results_df = None
        
        # Logger
        self.logger = logging.getLogger(self.LOGGER_NAME)
        
    def preprocess_data(self):
        """
        Preprocess the INCOSE guide and build templates for requirement revision.
        
        This method:
        1. Creates a unique run folder
        2. Preprocesses the INCOSE guide
        3. Builds templates based on INCOSE rules
        4. Creates evaluation configuration
        """
        # Create run folder
        self.run_name = f"run-{utils.get_current_date_time()}"
        self.output_data_folder = f"{self.config['FILE_LOCATIONS']['MAIN_DATA_FOLDER']}/{self.run_name}"
        Path(self.output_data_folder).mkdir(parents=True, exist_ok=True)
        
        # Preprocess INCOSE guide
        self.logger.info("Preprocessing INCOSE guide...")
        self.incose_df = preprocess_incose_guide(
            input_path=Path(self.config['FILE_LOCATIONS']['INCOSE_GUIDE']),
            output_path=Path(self.output_data_folder),
            start_page=65,
            end_page=115,
            regex=self.config['INCOSE_GUIDE_SETTINGS']['SECTIONS_REGEX_PAT'],
            replace_tokens=self.config['INCOSE_GUIDE_SETTINGS']['REPLACE_TOKENS'],
            subpatterns=self.config['INCOSE_GUIDE_SETTINGS']['SUBPATTERNS'],
            replace_with=self.config['INCOSE_GUIDE_SETTINGS']['REPLACE_WITH']
        )
        
        # Get base template messages
        self.base_template_messages = self.config['BASE_PROMPT_TEMPLATES'][self.template]
        
        # Build templates
        self.logger.info("Building templates from INCOSE guide...")
        self.templates = build_incose_templates(
            incose_df=self.incose_df,
            base_messages=self.base_template_messages,
            output_folder_path=Path(self.output_data_folder)
        )
        
        # Create evaluation configuration
        self.logger.info("Creating evaluation configuration...")
        self.evals_config = build_incose_eval_config(
            incose_df=self.incose_df,
            output_folder_path=Path(self.output_data_folder),
            templates=self.templates,
            rule_to_eval_map=self.config['PROMPT_EVALUTION_CONFIG']
        )

    def load_requirements(self):
        """
        Load requirements dataset and results tracking file.
        
        This method:
        1. Loads the requirements dataset from Excel
        2. Loads or creates the results tracking file
        """
        # Load requirements dataset
        try:
            self.logger.info(f"Loading requirements from {self.data}")
            self.reqs_df = pd.read_excel(self.data)    
        except FileNotFoundError:
            self.logger.error(f"Requirements file not found: {self.data}")
            raise
            
        # Load or create master results file
        try:
            results_path = Path(self.config['FILE_LOCATIONS']['MAIN_DATA_FOLDER']) / "results.xlsx"
            self.logger.info(f"Loading results tracking file from {results_path}")
            self.results_df = pd.read_excel(results_path)
        except FileNotFoundError:
            self.logger.info("Results tracking file not found, creating new one")
            results_df_columns = [
                'run_id', 'dataset', 'model', 'template', 'iternum',
                f"%_resolved_initial_{self.config['REQUIREMENTS_DATASET_SETTINGS']['FAILED_EVAL_COL']}",
                f"%_resolved_final_{self.config['REQUIREMENTS_DATASET_SETTINGS']['FAILED_EVAL_COL']}",
            ]
            self.results_df = pd.DataFrame(columns=results_df_columns, index=[0])

    def revise_requirements(self):
        """
        Evaluate and revise requirements.
        
        This method:
        1. Creates the requirement reviewer
        2. Runs initial evaluation
        3. Performs iterative revision
        4. Runs final evaluation
        """
        # Create requirement reviewer
        self.logger.info(f"Creating requirement reviewer with model {self.model}")
        self.incose_reviewer = IncoseRequirementReviewer(
            llm=ChatOllama(model=self.model),
            use_structured_llm=False,
            pydantic_model=None,
            templates=self.templates,
            evals_config=self.evals_config,
            id_col=self.config['REQUIREMENTS_DATASET_SETTINGS']['REQ_COLNAME']
        )
        
        # Run initial evaluation
        self.logger.info("Running initial requirement evaluation")
        self.reqs_df = self.incose_reviewer.run_eval_sequence(
            df=self.reqs_df, 
            col=self.config['REQUIREMENTS_DATASET_SETTINGS']['REQ_COLNAME'], 
            failed_eval_col=self.config['REQUIREMENTS_DATASET_SETTINGS']['FAILED_EVAL_COL'], 
            col_suffix='initial', 
            eval_to_rule_map=self.eval_func_to_rule_id_map
        )
        
        # Run iterative revision process
        self.logger.info(f"Starting revision process with {self.iternum} iterations")
        self.run_eval_loop()
        
        # Collect and merge revisions
        self.generate_revisions_df()
        self.merge_revisions_df()
        
        # Run final evaluation on revised requirements
        self.logger.info("Running final evaluation on revised requirements")
        self.reqs_df = self.incose_reviewer.run_eval_sequence(
            df=self.reqs_df, 
            col=f"Revised_{self.config['REQUIREMENTS_DATASET_SETTINGS']['REQ_COLNAME']}", 
            failed_eval_col=self.config['REQUIREMENTS_DATASET_SETTINGS']['FAILED_EVAL_COL'], 
            col_suffix='final', 
            eval_to_rule_map=self.eval_func_to_rule_id_map
        )

    def save_output(self):
        """
        Save all results and update the results tracking file.
        
        This method:
        1. Saves the requirements with revisions
        2. Updates the results tracking file
        """
        # Save requirements with revisions
        self.logger.info(f"Saving revised requirements to {self.output_data_folder}")
        utils.to_excel(self.reqs_df, self.output_data_folder, False, 'reqs_df_with_revisions')
        
        # Create new result entry
        self.logger.info("Updating results tracking file")
        new_result_df = pd.DataFrame(data={
            'run_id': self.run_name,
            'dataset': self.data,
            'model': self.model,
            'template': self.template,
            'iternum': self.iternum,
            f"%_resolved_initial_{self.config['REQUIREMENTS_DATASET_SETTINGS']['FAILED_EVAL_COL']}": 
                self.reqs_df[f"%_resolved_initial_{self.config['REQUIREMENTS_DATASET_SETTINGS']['FAILED_EVAL_COL']}"].iloc[0],
            f"%_resolved_final_{self.config['REQUIREMENTS_DATASET_SETTINGS']['FAILED_EVAL_COL']}": 
                self.reqs_df[f"%_resolved_final_{self.config['REQUIREMENTS_DATASET_SETTINGS']['FAILED_EVAL_COL']}"].iloc[0]
        }, index=[0])
        
        # Update results tracking file
        self.results_df = pd.concat([self.results_df, new_result_df], axis=0, ignore_index=False)
        self.results_df = self.results_df.dropna(subset=['run_id']).reset_index(drop=True)
        self.results_df = self.results_df.drop(columns=[c for c in self.results_df.columns if "Unnamed" in c])
        
        # Save results tracking file
        utils.to_excel(
            self.results_df, 
            self.config["FILE_LOCATIONS"]["MAIN_DATA_FOLDER"], 
            False, 
            'results'
        )
        self.logger.info("Output saved successfully")

    def run_eval_loop(self):
        """
        Run the iterative evaluation and revision loop.
        
        This method:
        1. Evaluates requirements against INCOSE rules
        2. Identifies requirements that need revision
        3. Generates revisions for failing requirements
        4. Repeats until max iterations or all requirements pass
        """
        df = self.reqs_df.copy()
        
        for iter_num in range(self.iternum):
            self.logger.info(f'Starting iteration {iter_num+1} of {self.iternum}')
            
            # Load previous iteration results if not first iteration
            if iter_num > 0:
                prev_file = f"{self.output_data_folder}/revised_df_iter_{iter_num-1}.xlsx"
                self.logger.info(f'Loading previous iteration results from {prev_file}')
                df = pd.read_excel(prev_file)
                df = df.dropna(subset=[self.config['REQUIREMENTS_DATASET_SETTINGS']['REQ_COLNAME']])
            
            # Keep only necessary columns
            df = df[[
                self.config['REQUIREMENTS_DATASET_SETTINGS']['REQ_COLNAME'], 
                f"{self.config['REQUIREMENTS_DATASET_SETTINGS']['REQ_COLNAME']}_#"
            ]]
            
            # Run evaluations
            self.logger.info(f'Evaluating requirements for iteration {iter_num+1}')
            df = self.incose_reviewer.run_eval_sequence(
                df=df, 
                col=self.config['REQUIREMENTS_DATASET_SETTINGS']['REQ_COLNAME'], 
                failed_eval_col=self.config['REQUIREMENTS_DATASET_SETTINGS']['FAILED_EVAL_COL'], 
                col_suffix=None, 
                eval_to_rule_map=self.eval_func_to_rule_id_map
            )
            
            if df is None:
                self.logger.error("Evaluation returned None - stopping iteration")
                break
                
            # Handle null values
            df = df.fillna('')
            
            # Identify requirements that pass all evaluations
            pass_cond = df[self.config['REQUIREMENTS_DATASET_SETTINGS']['FAILED_EVAL_COL']].str.len() == 0
            pass_rows = df[pass_cond]
            
            # Save current state for comparison
            prior_revision_df = df.copy()[[
                f"{self.config['REQUIREMENTS_DATASET_SETTINGS']['REQ_COLNAME']}_#", 
                self.config['REQUIREMENTS_DATASET_SETTINGS']['REQ_COLNAME'], 
                self.config['REQUIREMENTS_DATASET_SETTINGS']['FAILED_EVAL_COL']
            ]]
            
            # Rename columns for tracking
            renamed_columns = [
                f"{c}_prior_revision" for c in prior_revision_df.columns 
                if c != f"{self.config['REQUIREMENTS_DATASET_SETTINGS']['REQ_COLNAME']}_#"
            ]
            prior_revision_df.columns = [f"{self.config['REQUIREMENTS_DATASET_SETTINGS']['REQ_COLNAME']}_#"] + renamed_columns
            
            # Log passing requirements
            if len(pass_rows) > 0:
                self.logger.info(f'{len(pass_rows)}/{len(df)} requirements passed all criteria')
                df = df[~pass_cond]  # Keep only failing requirements
            
            # Process remaining requirements that need revision
            if len(df) > 0:
                self.logger.info(f'{len(df)} requirements still require revision')
                
                # Extract failed evaluations and requirements
                evals_lists = list(df[self.config['REQUIREMENTS_DATASET_SETTINGS']['FAILED_EVAL_COL']].values)
                args_lists = list(df[self.config['REQUIREMENTS_DATASET_SETTINGS']['REQ_COLNAME']].values)
                
                # Generate revisions
                self.logger.info(f'Generating revisions for {len(df)} requirements')
                revised_df = self.incose_reviewer.revise(evals_lists, args_lists, None)
                revised_df['revision'] = iter_num + 1
                
                # Preserve indices and merge with prior revision data
                revised_df.index = df.index
                revised_df = revised_df.reset_index().rename(
                    columns={'index': f"{self.config['REQUIREMENTS_DATASET_SETTINGS']['REQ_COLNAME']}_#"}
                )
                revised_df = pd.merge(
                    left=revised_df, 
                    right=prior_revision_df, 
                    on=f"{self.config['REQUIREMENTS_DATASET_SETTINGS']['REQ_COLNAME']}_#", 
                    how='inner'
                )
                
                # Handle empty revisions by using previous version
                revised_df[self.config['REQUIREMENTS_DATASET_SETTINGS']['REQ_COLNAME']] = revised_df[
                    self.config['REQUIREMENTS_DATASET_SETTINGS']['REQ_COLNAME']
                ].fillna(
                    prior_revision_df[f"{self.config['REQUIREMENTS_DATASET_SETTINGS']['REQ_COLNAME']}_prior_revision"]
                ) 
                
                # Save iteration results
                utils.to_excel(revised_df, self.output_data_folder, str(iter_num), 'revised_df_iter')
            else:
                self.logger.info('All requirements passed evaluation - stopping iterations')
                break
                
            self.logger.info(f'Completed iteration {iter_num+1}')

    def generate_revisions_df(self, pat: str = 'revised_df*', revision_number_col: str = 'revision'):
        """
        Collect all revisions into a single DataFrame.
        
        Args:
            pat: File pattern to match revision files
            revision_number_col: Column name for revision number
            
        Returns:
            DataFrame containing all revisions
        """
        self.logger.info(f'Collecting revisions from {self.output_data_folder}')
        matching_files = list(Path(self.output_data_folder).rglob(pat))
        
        if not matching_files:
            self.logger.warning(f'No revision files found matching pattern {pat}')
            return None
            
        dfs = []
        for file in matching_files:
            self.logger.debug(f'Loading revision file: {file}')
            temp_df = pd.read_excel(file)
            
            # Rename requirement column to indicate it's a revision
            temp_df = temp_df.rename(
                columns={
                    self.config['REQUIREMENTS_DATASET_SETTINGS']['REQ_COLNAME']: 
                    f"Revised_{self.config['REQUIREMENTS_DATASET_SETTINGS']['REQ_COLNAME']}"
                }
            )
            
            # Remove unnamed columns
            temp_df = temp_df.drop(columns=[c for c in temp_df.columns if 'Unnamed:' in c])
            dfs.append(temp_df)
        
        # Combine all revisions
        self.revisions_df = pd.concat(dfs, ignore_index=True, axis=0)
        
        # Remove empty revisions
        self.revisions_df = self.revisions_df[
            self.revisions_df[f"Revised_{self.config['REQUIREMENTS_DATASET_SETTINGS']['REQ_COLNAME']}"].str.strip() != ''
        ]
        
        # Save combined revisions
        utils.to_excel(self.revisions_df, self.output_data_folder, False, 'revisions_df')
        self.logger.info(f'Generated revisions DataFrame with {len(self.revisions_df)} entries')
        
        return self.revisions_df

    def merge_revisions_df(self, revision_number_col: str = 'revision'):
        """
        Merge the latest revisions into the original requirements DataFrame.
        
        Args:
            revision_number_col: Column name for revision number
            
        Returns:
            DataFrame with original requirements and their latest revisions
        """
        self.logger.info('Merging latest revisions with original requirements')
        
        if self.revisions_df is None or self.revisions_df.empty:
            self.logger.warning('No revisions to merge')
            return self.reqs_df
            
        # Get the latest revision for each requirement
        self.revisions_df = (
            self.revisions_df
            .sort_values(
                by=[f"{self.config['REQUIREMENTS_DATASET_SETTINGS']['REQ_COLNAME']}_#", revision_number_col], 
                ascending=True
            )
            .drop_duplicates(
                subset=[f"{self.config['REQUIREMENTS_DATASET_SETTINGS']['REQ_COLNAME']}_#"], 
                keep='last'
            )
            .reset_index(drop=True)
        )
        
        # Merge with original requirements
        self.reqs_df = pd.merge(
            left=self.reqs_df, 
            right=self.revisions_df[[
                f"Revised_{self.config['REQUIREMENTS_DATASET_SETTINGS']['REQ_COLNAME']}",
                f"{self.config['REQUIREMENTS_DATASET_SETTINGS']['REQ_COLNAME']}_#"
            ]], 
            on=f"{self.config['REQUIREMENTS_DATASET_SETTINGS']['REQ_COLNAME']}_#", 
            how='left'
        )
        
        self.logger.info(f'Merged revisions with {len(self.reqs_df)} requirements')
        return self.reqs_df