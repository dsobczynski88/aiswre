'''
The purpose of this script is to auto-generate evaluation functions for each INCOSE Guide Rule using NLP
'''
import sys
import argparse
import numpy as np
from pprint import pformat
from pathlib import Path
import logging
import pandas as pd
from dotenv import dotenv_values
from dotenv import load_dotenv
from tqdm import tqdm
import asyncio
import nest_asyncio
from langchain_ollama import ChatOllama
import src
from src.prj_logger import ProjectLogger, get_logs
from src import utils
from src.components.preprocess import BuildTemplates
from src.components.promptrunner import PromptRunner
from src.components.workflow import BasicWorkflow

if __name__ == "__main__":

    config = utils.load_config()
    output_directory = utils.make_output_directory(config["FILE_LOCATIONS"])
    ProjectLogger(src.BASE_LOGGERNAME,f"{output_directory}/{src.BASE_LOGGERNAME}.log").config()
    wf = BasicWorkflow(config=config)
    wf.preprocess_data()

    incose_df = wf.incose_preprocessor.df.head(1)
    incose_df.to_excel(f"{config['FILE_LOCATIONS']['MAIN_DATA_FOLDER']}/incose_df.xlsx")
    base_template_messages = {
        'generate-eval-funcs':
        {
            'name': 'generate-eval-funcs',
            'description': 'A prompt which uses examples of evaluation functions for INCOSE Section 4 Rules and context from the Guide to generate Evaluation Functions',
            'system': 'You are a seasoned in NLP and software development using python. The user will hand over a a description. Your task is to generate a function given the description and format instructions as specified in the examples.',
            'user': '## Examples: \n\nDescription: Generate a function that checks whether a requirement statement meets the following: R11 - SEPARATE CLAUSES \n\nContext: Use a separate clause for each condition or qualification.Each need or requirement should have a main verb describing a basic function or need. If appropriate, the main sentence may then be supplemented by clauses that provide conditions or qualifications (performance values, trigger, or constraints). A single, clearly identifiable clause should be used for each condition or qualification expressed.As mentioned in R1 and Appendix C, a need and requirement should match one, and only one, pattern from the catalog of agreed patterns.If an applicable qualifying clause or condition is not stated explicitly within the need or requirement statement, the need or requirement statement is not Complete (C4), Verifiable/Validatable (C7), nor Correct (C8).If a qualifying clause is not stated explicitly within the need or requirement statement, the need or requirement statement is not Complete (C4)), Verifiable/Validatable (C7), nor Correct (C8)—for example, performance associated with the action verb or for an interface requirement where a pointer to where the specific interaction is defined (such as in the ICD).When using clauses, make sure the clause does not separate the object of the sentence from the verb. See also R1, R18, R27 and Appendix C. \n\nFunction: def eval_has_separate_clauses_for_conditions(text: str) -> bool: """\n    R11: Criteria from INCOSE Guide to Writing Requirements:\n        check if text uses a separate clause for each condition or qualification\n		\n	This function analyzes text to determine if it follows the requirement of using separate clauses for each condition or qualification. It works by:\n	Identifying potential indicators of multiple conditions (conjunctions, conditional markers)\n	Checking if these conditions appear to be properly separated into distinct clauses\n	Using part-of-speech tagging to count verbs as an estimate of clause boundaries\n	Returning False if it detects multiple conditions that aren"t properly separated into distinct clauses\n	Note that natural language analysis is complex, and this function provides a reasonable heuristic rather than perfect detection. More sophisticated analysis would require deeper syntactic parsing.\n    """\n    # Ensure NLTK data is available\n    try:\n        nltk.data.find("tokenizers/punkt")\n        nltk.data.find("taggers/averaged_perceptron_tagger")\n    except LookupError:\n        nltk.download("punkt")\n        nltk.download("averaged_perceptron_tagger")\n    \n    # Tokenize into sentences\n    sentences = sent_tokenize(text)\n    \n    for sentence in sentences:\n        # Check for multiple conditions in a single clause\n        \n        # Look for conjunction patterns that might indicate multiple conditions\n        # in a single clause (e.g., "and", "or", "as well as", etc.)\n        conjunction_pattern = r"\b(and|or|but|as well as|along with|together with|plus)\b"\n        conjunctions = re.findall(conjunction_pattern, sentence.lower())\n        \n        # Look for multiple conditional markers in the same clause\n        conditional_markers = ["if", "when", "unless", "until", "provided that", "in case", "as long as"]\n        condition_count = sum(1 for marker in conditional_markers if marker in sentence.lower())\n        \n        # Check for multiple comma-separated clauses that might indicate conditions\n        comma_separated_clauses = len(re.findall(r",\s*(?:which|that|who|where|when)", sentence))\n        \n        # If we find multiple conjunctions or conditional markers, it might indicate\n        # multiple conditions not properly separated into distinct clauses\n        if len(conjunctions) > 1 or condition_count > 1 or comma_separated_clauses > 1:\n            # Further analysis to confirm if these are actually condition clauses\n            tokens = word_tokenize(sentence)\n            tagged = pos_tag(tokens)\n            \n            # Count verbs to estimate number of clauses\n            verb_tags = ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]\n            verb_count = sum(1 for _, tag in tagged if tag in verb_tags)\n            \n            # If we have multiple conjunctions/conditions but not enough verbs\n            # to support separate clauses, it"s likely conditions aren"t properly separated\n            if verb_count < len(conjunctions) + condition_count:\n                return False\n    \n    # If we"ve checked all sentences and found no issues, return True\n    return True \n\nDescription: Generate a function that checks whether a requirement statement meets the following: R18 - SINGLE THOUGHT SENTENCE \n\nContext: Write a single sentence that contains a single thought conditioned and qualified by relevant sub-clauses.Need statements and requirement statements (based on the concepts of allocation, traceability, validation, and verification) must contain a single thought allowing:• needs to be traced to their source;• the single thought within a requirement statement to be allocated;• the resulting single-thought child requirements to trace to their allocated parent,• requirements to trace to a single-thought source;• design and system validation and design and system verification against the single-thoughtneed or requirement.Sometimes a need statement or requirement statement is only applicable under a specific trigger, condition, or multiple conditions as discussed in Section 1.11. If multiple actions are needed for a single condition, each action should be repeated in the text of a separate need statement or requirement statement along with the triggering condition, rather than stating the condition and then listing the multiple actions to be taken. Using this convention, the system can be verified to perform each action, and each action can be separately allocated to the entities at the next level of the architecture.Also avoid stating the condition or trigger for an action in a separate sentence. Instead write a simple affirmative declarative sentence with a single subject, a single main action verb and a single object, framed and qualified by one or more sub-clauses.Avoid compound sentences containing more than one subject/verb/object sequence. This constraint is enforced in the catalog of agreed patterns (see R1 and Appendix C).Often when there are multiple sentences for one requirement, the writer is using the second sentence to communicate the conditions for use or rationale for the requirement for the first sentence. This practice is not acceptable—rather include rationale in the attribute A1 - Rationaleas part of the requirement expression and include the condition of use within the need statement or requirement statement or an attribute within the need or requirement expression.See also R1, R11, R27, and R28. \n\nFunction: def eval_is_singular_statement(text: str) -> bool:\n    """\n    R18: Criteria from INCOSE Guide to Writing Requirements:\n        check if text contains a singular statement (one thought)\n		\n	This function evaluates whether text contains a singular statement (one thought) by analyzing the use of conjunctions like "and" and "or". It distinguishes between:\n\n	Acceptable uses of conjunctions in logical conditions (e.g., "The system shall activate when X is true AND Y is false")\n	Unacceptable uses that join multiple thoughts (e.g., "The system shall process data AND generate reports")\n	The implementation uses part-of-speech tagging to identify verbs and subjects, helping determine if conjunctions are joining separate clauses (indicating multiple thoughts) or just elements within a single thought. It also looks for logical condition indicators and operators that would make conjunction use acceptable.\n\n	Note that natural language analysis is complex, and this function provides a reasonable heuristic rather than perfect detection. More sophisticated analysis would require deeper syntactic parsing.\n    """\n	\n	def is_logical_condition(tagged, conj_idx):\n		"""\n		Check if the conjunction is part of a logical condition expression\n		"""\n		# Look for logical condition indicators\n		logical_indicators = ["if", "when", "while", "unless", "until"]\n		\n		# Check for logical indicators before the conjunction\n		for i in range(max(0, conj_idx - 10), conj_idx):\n			if tagged[i][0].lower() in logical_indicators:\n				return True\n				\n		# Check for parentheses or brackets which might indicate logical grouping\n		before_conj = " ".join([word for word, _ in tagged[:conj_idx]])\n		if "(" in before_conj or "[" in before_conj:\n			return True\n			\n		# Check for logical operators nearby (AND, OR, XOR, NOT in all caps)\n		for i in range(max(0, conj_idx - 3), min(len(tagged), conj_idx + 3)):\n			if tagged[i][0] in ["AND", "OR", "XOR", "NOT"]:\n				return True\n            \n		return False\n\n	def is_joining_thoughts(tagged, conj_idx):\n		"""\n		Check if the conjunction is joining two separate thoughts/statements\n		"""\n		# Count verbs before and after conjunction to see if we have multiple clauses\n		verbs_before = 0\n		verbs_after = 0\n		\n		# Verb POS tags\n		verb_tags = ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]\n		\n		# Count verbs before conjunction\n		for i in range(conj_idx):\n			if tagged[i][1] in verb_tags:\n				verbs_before += 1\n				\n		# Count verbs after conjunction\n		for i in range(conj_idx + 1, len(tagged)):\n			if tagged[i][1] in verb_tags:\n				verbs_after += 1\n		\n		# If we have verbs both before and after, it might be joining thoughts\n		if verbs_before > 0 and verbs_after > 0:\n			# Check for subject-verb pairs on both sides (indicating separate clauses)\n			has_subject_before = False\n			has_subject_after = False\n			\n			# Subject POS tags (simplified)\n			subject_tags = ["NN", "NNS", "NNP", "NNPS", "PRP"]\n			\n			# Check for subjects before conjunction\n			for i in range(conj_idx):\n				if tagged[i][1] in subject_tags:\n					has_subject_before = True\n					break\n					\n			# Check for subjects after conjunction\n			for i in range(conj_idx + 1, len(tagged)):\n				if tagged[i][1] in subject_tags:\n					has_subject_after = True\n					break\n			\n			# If we have subject-verb pairs on both sides, it"s likely joining thoughts\n			if has_subject_before and has_subject_after:\n				return True\n		\n		return False\n	\n    # Ensure NLTK data is available\n    try:\n        nltk.data.find("tokenizers/punkt")\n        nltk.data.find("taggers/averaged_perceptron_tagger")\n    except LookupError:\n        nltk.download("punkt")\n        nltk.download("averaged_perceptron_tagger")\n    \n    # Split into sentences\n    sentences = sent_tokenize(text)\n    \n    for sentence in sentences:\n        # Tokenize and tag parts of speech\n        tokens = word_tokenize(sentence)\n        tagged = pos_tag(tokens)\n        \n        # Find all instances of "and", "or", etc.\n        conjunctions = []\n        for i, (word, tag) in enumerate(tagged):\n            if word.lower() == "and" or word.lower() == "or":\n                conjunctions.append((i, word.lower()))\n        \n        # If no conjunctions, the sentence is likely singular\n        if not conjunctions:\n            continue\n            \n        # Check each conjunction to determine if it"s joining thoughts or conditions\n        for conj_idx, conj_word in conjunctions:\n            # Check if this is a logical condition (acceptable use)\n            if is_logical_condition(tagged, conj_idx):\n                continue\n                \n            # Check if this conjunction is joining two separate thoughts (not acceptable)\n            if is_joining_thoughts(tagged, conj_idx):\n                return False\n    \n    # If we"ve checked all sentences and found no issues, it"s a singular statement\n    return True \n\nDescription: Generate a function that checks whether a requirement statement meets the following: {rule_number} {rule_title} \nContext: {definition}{elaboration} \nFunction:'
        }
    }
    trace_template_builder = BuildTemplates(
        df=incose_df,
        base_messages=base_template_messages['generate-eval-funcs'],
    )
    trace_template_builder.add_message_col_to_frame("system")
    trace_template_builder.add_message_col_to_frame("user")
    
    trace_template_builder.assemble_templates_from_df(
        system_message_colname='system_message',
        user_message_colname='user_message',
        template_name_prefix='Temp'
    )
    # instantiate the PromptRunner
    runner = PromptRunner(
        llm=ChatOllama(model='llama3.1'),
        use_structured_llm=False,
        pydantic_model=None,
    )
    # build the chains using templates
    chains = []
    #for idx, tmp in enumerate(trace_template_builder.templates):       
    chain = trace_template_builder.templates['Temp1'] | runner.llm | (lambda x: x.content)
    # define the run-time arguments
    argdict_list=[]
    for index, row in incose_df.iterrows():
        arg_dict = {'rule_number': row['rule_number'], 'rule_title': row['rule_title'], 'definition': row['definition'], 'elaboration': row['elaboration']}
        argdict_list.append(arg_dict)
        chains.append(chain) 

    # run the prompts
    results = asyncio.run(runner.run_multiple_chains(chains, argdict_list))
    results_df = pd.DataFrame(results)
    #results_df = results_df.rename(columns={0:'assessment'})
    #results_df['assessment_tuple']=results_df['assessment'].apply(lambda s: utils.recast_str(s))
    #results_df['content']=results_df['assessment_tuple'].apply(lambda t: t[1])
    results_df.to_excel(f"{config['FILE_LOCATIONS']['MAIN_DATA_FOLDER']}/generated_eval_funcs_test.xlsx")
