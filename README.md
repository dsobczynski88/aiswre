## Applying NLP and AI to improve quality of software requirement statements using INCOSE Guide to Writing Requirements


### Introduction
High-quality software requirements serve as the foundation for the successful execution of software projects, and properly articulating these requirements enables software to be developed in line with stakeholder expectations, budget, and timeline. Furthermore, downstream changes to requirements may result in increased development costs, delayed delivery, hotfixes, and dissatisfied customers. As such, software teams need to put careful thought into how these requirements are written and who has been included in the review process.

Typically, software requirements are housed in a requirements management tool in the form of textual statements. Naturally, many aspects of high-quality requirements can only be realized by experts and cross-functional team members working on the project; however, there are best practice guidelines as described in the INCOSE Guide to Writing Requirements which, when followed, improve the quality (e.g., accuracy, concision, non-ambiguity, singularity, etc.) of the written requirement. Additionally, the INCOSE Guide is well-recognized in the field of systems engineering and thus trusted amongst industry experts.

This project (titled `aiswre`) seeks to integrate the best practices described in the INCOSE Guide to Writing Requirements to enhance software requirement quality using NLP and AI. In this article specifically, a project using `langchain` is described which refines a given software requirement based on the rules described in Section 4 of the INCOSE Guide.

### Overview of the INCOSE Guide to Writing Requirements

The [INCOSE Guide](https://www.incose.org/docs/default-source/working-groups/requirements-wg/gtwr/incose_rwg_gtwr_v4_040423_final_drafts.pdf?sfvrsn=5c877fc7_2) is structured around a framework of principles aimed at ensuring high-quality requirements documentation. Section 4 of the Guide focuses on rules for individual need and requirement statements as well as for sets of needs and requirements. The focus of this project is on applying the rules to individual requirement statements. Each rule is described in detail with definitions, elaborations, and examples. Furthermore, syntactical patterns that exemplify these rules are detailed in Appendix C of the INCOSE Guide. A brief summary of the Section 4 rules is described in the below table.

### Overview of the `aiswre` project

This project, `aiswre` intends to apply AI, NLP, and data science methodologies to improve the quality of software quality processes. The initial features of the project focus on using prompt engineering techniques to refine software requirements based on the rules described in the section **Overview of the INCOSE Guide to Writing Requirements**. This project was inspired by the desire to enhance the field of software quality with AI and system engineering best practices. Application of LLMs bear the opportunity to advance the field of requirements engineering as initial studies have shown promising results<sup>1,2</sup>.

### Design description

This project leverages `langchain-openai` to build a software requirement refiner application. The project will take a requirement as input, assess it against a variety of criteria, and based on the identified gaps, will run a sequential chain of prompts to refine the requirement to align with rules as described in INCOSE Guide to Writing Requirements Section 4. At present, the application only leverages the input requirement and INCOSE Guide (no other information about the project) to perform the refinement chain. After each refinement of the requirement, each requirement is re-evaluated against the input criteria to check whether the refinement chain resulted in passing of the acceptance criteria as provided in Section 4 of the INCOSE Guide to Writing Requirements. Once all criteria have been met or the number of maximum iterations has been met, the program will stop running and provide an output file showing the input requirement, final output (refined) requirement, and the final acceptance criteria evaluation.

### Overview of Software Modules

|name         |description|
|------------ |-----------|
|aiswre         |This is the main script called when running the program. In addition to loading all configuration settings and input data, it contains the function `run_eval_loop`, which executes the requirement refinement process. Specifically, the function takes in a list of requirements, evaluates them using specific evaluation functions, and based on which criteria failed, runs tailored prompts designed to correct those specific failures. The algorithm is designed to run until all requirements have passed all criteria or the set number of maximum iterations has been met.|
|preprocess   |Contains the base `TextPreprocessor` class used to clean the extracted text. Additionally, the `Sectionalize` class is used to split PDFs into specific sections and extract specific subsections of the INCOSE Guide. This module also contains the base `BuildTemplates` class to create prompts and templates from a dataframe structure that is convenient for the purposes of this project. The classes `BuildIncoseTemplates` and `PreprocessIncoseGuide` are tailored to perform preprocessing and template creation specific for this project; for example, one of the methods of `BuildIncoseTemplates` is  `load_evaluation_config`, which returns a dictionary linking prompt templates (based on INCOSE Section 4 rules) with specific evaluation functions. This allows the application to know which prompts correspond to specific failed evaluations.|
|promptrunner |Contains the base `PromptRunner` class and the child class `IncosePromptRunner`, which contains the methods to build and run chains asynchronously based on requirements' failed evaluations. Specifically, the method `run_multiple_chains` accepts a list of chains (e.g., langchain Runnable) and a list of arguments where each element of the list is a requirement. However, prompts could be constructed with multiple arguments. In the prompts defined here, the only input to the prompt at runtime is the requirement. The first element in the chain arguments corresponds to the arguments of the first chain in the chains list and thus both these lists are of equal length. These chains and their respective arguments are run asynchronously when the `run_multiple_chains` method is called. The `IncosePromptRunner` class also contains methods to build the `RunnableSequence` objects, which comprise the chains as well as methods to assemble a `RunnableLambda`, which is used to feed a requirement to an LLM and parse the result (in this case, the revised version of the input requirement).|
|prompteval   |Contains functions used to evaluate requirement quality based on INCOSE Guide to Writing Requirements|
|prj_exception|Contains the `CustomException` class used to catch exceptions from project functions|
|prj_logger   |Contains the `ProjectLogger` class used to configure the logger used for the project. Additionally, the decorator `get_logs` is used to log exceptions using `CustomException` and output runtime.|
|prj_utils    |Contains a variety of functions used to manipulate dataframes and input/output data|

**NOTE:** The folder `./src/data` houses all input and output data from the project.

### Program workflow

This section provides a synopsis of the workflow that occurs when running `aiswre`. This assumes the following base template is used to construct the rule-specific templates designed to revise requirements based on specific failed criteria.

- **Load environment variables**
	- This is where to store your OPENAI_API_KEY, for example.
- **Load the dataset (e.g., dataframe containing software requirements)**
	- For this project, a software requirements dataset from Kaggle will be explored [(See Requirements Dataset)](https://www.kaggle.com/datasets/iamsouvik/software-requirements-dataset).
	- Each row of the loaded dataset (pandas dataframe) contains a unique software requirement in the column named *Requirement*.
- **Configure logger**
	- The logger is configured so that it can be used to log outputs from all program functions.
- **Parse the INCOSE Guide to Writing Requirements**
	- This step involves use of `regex` to parse specific subsections within each rule description in Section 4.
	- A text preprocessing pipeline is also applied during this step.
- **Build Requirement Evaluation Prompt Templates**
	- Load the selected base template.
	- The base template is a generic prompt template that is designed to generate a requirement revision based on a given set of criteria and examples.
	- Using the selected base template, rule-specific information is used to populate the base template resulting in a tailored prompt for a specific INCOSE rule. For example, since there are 50 rules in Section 4, 50 unique prompts could be constructed from this base template.
	- Once the prompt templates have been created per the selected base template, specific evaluation functions need to be associated with these prompts so that when the requirement fails a criterion associated with a specific rule, the appropriate prompt can be added to the evaluation sequence.
	- The `evals_config` associates prompt templates (INCOSE rules) with specific evaluation criteria (functions) and is used as a key input for executing prompts for revising the requirements.
- **Instantiate the `IncosePromptRunner` class**
	- The `PromptRunner` instance accepts a large language model (LLM) and can accept pydantic output models for handling prompt responses.
	- The method `assemble_eval_chain_list` takes in a list of evaluation function lists (see module `prompteval`) and the `evals_config` discussed earlier to generate a list of chains where each chain is of type `RunnableSequence`.
	- Each element of a given `RunnableSequence` is a chain which takes in a requirement, passes it to an LLM, retrieves the content, and parses the result to obtain the revised requirement text.
	- The number of failed evaluations for a specific requirement defines the number of elements comprising the `RunnableSequence` but all elements follow the mentioned workflow of revising a requirement.
- **Run the requirement evaluation loop function `run_eval_loop`**
	- The `run_eval_loop` method takes will evaluate a set of requirements for a predefined number of iterations or until all requirements have passed all imposed criteria.
	- During each iteration of the algorithm, the current revision of requirements is loaded and evaluated per `call_evals`.
	- Requirements that pass all criteria are popped from the dataframe as no further action is needed.
	- Requirements that fail any criteria remain in the dataframe and are evaluated using the specific prompts design to address the specific criteria that was not met.
- **Create the consolidated revisions and results output file**
	- An output Excel file within the generated run folder is created containing the original requirement, its final revision, the result of the evaluation program and the total number of revisions created titled reqs_df_with_revisions.
	- The settings associated with each run and the metric value for percent of requirements resolved (all evaluation criteria passed) is appended to the Excel file results.

### Getting started

- Set up your OpenAI API key [OPEN AI Developer quickstart](https://platform.openai.com/)
- Add requirements dataset to the directory
- Open a powershell terminal and enter the following to clone the repository:
	- `git clone https://github.com/dsobczynski88/aiswre.git <your_desired_folder_name>`
- Navigate to the folder containing the cloned repository:
	- `cd <your_desired_folder_name>`
- Create a blank `.env` file in this location and enter:
	- `OPENAI_API_KEY = <your_api_key>`
- Create a virtual env:
	- `python -m venv venv` 
- Activate the environment (Windows Powershell):
	- `.\\venv\Scripts\activate.bat`
- Enter the following commands to install the code and dependencies:
	- `python -m pip install -r requirements.txt`
	- `python -m pip install -e .`
- Enter the following command to run the program:
	- `python -m aiswre -d <reqs_filename> -m <openai_model> -t <base_prompt_template> -i <max_iter>`

**NOTE:** There are other ways (besides using a .env file) to define the API key. This is the approach used in this project.

### Example Demo

In this example, we will evaluate five (5) requirements from the Kaggle dataset. For simplification, the extent of our evaluation will be limited to six (6) Section 4 Rules (R3,R7,R8,R9,R10,R19). The requirements to be reviewed are defined below and saved to an Excel file in the ./src/data folder. 

- <u>Define the dataset</u>.
	1. The Disputes System shall record the name of the user and the date for any activity that creates or modifies the disputes case in the system.  A detailed history of the actions taken on the case, including the date and the user that performed the action, must be maintained for auditing purposes.,
	2. The WCS system shall use appropriate nomenclature and terminology as defined by the Corporate Community Grants organization. All interfaces and reports will undergo usability tests by CCR users.,
	3.  The system will notify affected parties when changes occur affecting clinicals  including but not limited to clinical section capacity changes  and clinical section cancellations.,
	4. Application testability DESC: Test environments should be built for the application to allow testing of the application's different functions.,
	5. The product shall be platform independent. The product shall enable access to any type of development environment and platform.

- <u>Run the program.</u>
	- Now that the requirement dataset has been defined, the program will be run to evaluate these requirements. Note that the program main script (`aiswre.py`) takes in four (4) required command-line arguments. In the command prompt enter:
	```python 
	python -m aiswre -d <path/to/dataset/> -m <model name> -t <template name> -i <num of maxiter>
	``` 

	|argument            |description                                                                                     |
	|--------------------|------------------------------------------------------------------------------------------------|
	|`--data` or `-d`    |The requirements dataset file name                                                              | 
	|`--model` or `-m`   |The string name of the LLM model to be used for revising the requirements                       |
	|`--template` or `-t`|The string name of the base template to be used, as defined in `preprocess.py`                  | 
	|`--iternum` or `-i` |The maximum number of iterations to be run during the evaluation loop function (`run_eval_loop`)|

- <u>Review the results.</u>
	- To view the results of the program, open results.xlsx from the ./src/data folder. The results.xlsx will display the metric `%_resolved_final_failed_evals`, which captures the percentage of requirements that passed all evaluation criteria post LLM-assisted revisioning. For example, consider the case where this program has been run with the following input argument values:
	
	|model           |template                 |iternum   |
	|----------------|-------------------------|----------|
	|gpt-4o-mini     |req-reviewer-instruct-2  |3         |
	
	- For the above five (5) mentioned requirements, three (3) of the five (5) were revised such that all evaluation functions passed. The original and revised requirements for these cases are presented below:
	**NOTE:** To view the revised requirement outputs, go to the generated run folder within ./src/data and open the file reqs_df_with_revisions.xlsx
	
	|Original Requirement        |Revised Requirement                                                                             |
	|----------------------------|------------------------------------------------------------------------------------------------|
	|The WCS system shall use appropriate nomenclature and terminology as defined by the Corporate Community Grants organization. All interfaces and reports will undergo usability tests by CCR users.|The WCS system shall **implement** the nomenclature defined by the Corporate Community Grants organization. All interfaces **shall complete usability tests with a minimum of 15 CCR users within a maximum time frame of 30 days following their development.**|
	|The system will notify affected parties when changes occur affecting clinicals  including but not limited to clinical section capacity changes  and clinical section cancellations.|The system shall notify affected parties when changes occur **affecting clinical section capacity. The system shall notify affected parties when changes occur affecting clinical section cancellations.**|
	|Application testability DESC: Test environments should be built for the application to allow testing of the applications different functions.|The test environments **shall** be built for the application **to test its different functions.**|

- <u>Discuss the results.</u>
	- The original requirements failed tests for presence of vague terms, combinators, open-end clauses, and superfluous infinitives. For example, the requirement beginning with "The system will notify..." contains the phrase "including but not limited to", which is an open-end clause not recommended by Rule 9. The requirement beginning "Application testability..." failed evaluation for superfluous infinitives as it contains the infinitive "to allow".
	- The revised version of these requirements do not contain these ambiguous phrases and as a result, all criteria is passed.
	- The question remains, is this now a *"high-quality"* requirement? Ultimately, the answer will come from the team working on this project; however, it is concluded that the program successfully revised the requirements to be compliant with the six (6) INCOSE rules in-scope for this demo.  
       
### Future Work

At present, this work is structured in a deterministic way that limits its ability to improvise, and the use of more complex LCEL expressions and AI agents is a future area of exploration. In addition, a more in-depth exploration of prompt engineering offers potential for the application to yield more useful results. This same thought process applies to the evaluation functions to refine the extent to which outputs can be measured. An additional area is to perform a feedback study from industry experts on the results from the tool and compare this with an AI-enabled feedback study.

The program itself will greatly benefit from usage of more advanced approaches in AI such as langgraph flows and agent frameworks. Furthermore, the work at best is designed for a handful of INCOSE rules and therefore are still in progress. To improve the robustness and utility of this work, there are opportunities to leverage more efficient design patterns, and this too is a subject of ongoing project activities. 

### Thoughts and Acknowledgements

I'd like to thank my colleagues who have encouraged me to pursue learning new skills and following through on ideas. In addition, I was able to develop this project with the help of countless articles from Medium and stackoverflow. Additionally, the Udemy course *The Complete Prompt Engineering for AI Bootcamp (2025)* has been extremely helpful. I am very thankful for the individuals who create and make available such resources.
 
A few comments about me: I am a software quality engineer who is passionate about designing creative ways to solve problems and continuously improve on standard ways of working. I find NLP and generative AI to be a remarkably exciting and interesting field, and I truly enjoy working "hands on" with this technology. Please clap for the article if you enjoyed it. I really do appreciate the support :sunglasses:. Future work also involves refining the methods to be more efficient and enhance readability through choice of data structure. Please send me comments on aspects that you found interesting and feedback on how it could be improved.

### Code: [aiswre](https://github.com/dsobczynski88/aiswre)

### References

1. A. Fantechi, S. Gnesi, L. Passaro and L. Semini, "Inconsistency Detection in Natural Language Requirements using ChatGPT: a Preliminary Evaluation," 2023 IEEE 31st International Requirements Engineering Conference (RE), Hannover, Germany, 2023, pp. 335-340, doi: 10.1109/RE57278.2023.00045.

2. Frattini, Julian, et al. "NLP4RE Tools: Classification, Overview and Management." Handbook on Natural Language Processing for Requirements Engineering. Cham: Springer Nature Switzerland, 2025. 357-380.
