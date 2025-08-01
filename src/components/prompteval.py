"""
Module that evaluates the quality of requirement specifications based on INCOSE 
(International Council on Systems Engineering) Guide to Writing Requirements.

This module contains functions that assess various aspects of requirements to 
identify potential issues such as passive voice, vague verbs, indefinite articles, 
and other problematic constructs as referenced in the INCOSE guide.

Functions:
- convert_bool_to_ohe: Converts a boolean input to a one-hot encoded integer (0 or 1).
- eval_is_in_passive_voice: (Placeholder) Checks if a text is written in passive voice.
- eval_if_vague_verb: Identifies if a requirement uses any vague verbs.
- eval_has_a_def_article: Checks if a text contains an indefinite article "a".
- eval_has_vague_terms: Detects the presence of vague terms within the text.
- eval_has_escape_clause: Checks for escape clauses which indicate vague conditions.
- eval_has_open_end_clause: Detects open-end clauses that suggest incompleteness.
- eval_has_superfl_inf: Identifies the presence of superfluous infinitive phrases.
- eval_has_combinators: Checks for words that join or combine clauses.

Logging:
- Configures a logger 'proj_logger' for accurate logging of results and possible issues. 
"""


from typing import Union, List 
import re
import logging
import pandas as pd
from tqdm import tqdm
import nltk
import src

nltk.download('averaged_perceptron_tagger')

LOGGERNAME = f"{src.BASE_LOGGERNAME}.prompteval"
proj_logger = logging.getLogger(LOGGERNAME)


def convert_bool_to_ohe(bool_result: bool) -> int:
    """Convert input boolean to an integer"""
    if bool_result:
        return 1
    else:
        return 0

def eval_is_in_passive_voice(text: str) -> bool:
    """
    R2: Criteria from 4.1.2 INCOSE Guide to Writing Requirements:
        check if text is written in passive voice
    """
    be_forms = {"is", "are", "was", "were", "been", "be", "am", "'s", "'re", "'m"}
    def check_for_passive_voice(tagged):
        for i in range(len(tagged) - 1):
            # Check if current word is a form of "to be"
            if tagged[i][0] in be_forms:
                # Check if next word is a past participle
                if tagged[i+1][1] == 'VBN':
                    # Check for optional "by" phrase
                    if i+2 < len(tagged) and tagged[i+2][0] == 'by':
                        return True
                    # Even without "by", it could still be passive
                    return True
        return False
        
    sents = nltk.sent_tokenize(text)
    for s in sents:
        tokens = nltk.word_tokenize(s)
        tagged = nltk.pos_tag(tokens)
        if check_for_passive_voice(tagged):
            return True
    return False

def eval_if_vague_verb(text: str) -> bool:
    """
    R3: Criteria from 4.1.3 INCOSE Guide to Writing Requirements:
        check if the requirements uses a vague verb 
    """
    
    vague_verbs = [
        "support", "process", "handle", "track", "manage", "flag"
    ]
    text = text.split()
    for verb in vague_verbs:
        if verb in text:
            return True
    else:
        return False


def eval_has_a_def_article(text: str) -> bool:
    """
    R5: Criteria from 4.1.5 INCOSE Guide to Writing Requirements:
        check if text contains indefinite article \"a\" 
    """
    text = text.split()
    if ["a"] in text:
        return True
    else:
        return False


def eval_has_vague_terms(text: str) -> bool:
    """
    R7: Criteria from 4.1.7 INCOSE Guide to Writing Requirements:
        check if text contains vague terms        
    """
    vague_terms = [
        "some", "any", "allowable", "several", "many", "a lot of", "a few", "almost always", 
        "very nearly", "nearly", "about", "close to", "almost","approximate","ancillary", "relevant", 
        "routine", "common", "generic", "significant","flexible", "expandable", "typical", "sufficient", 
        "adequate", "appropriate", "efficient", "effective", "proficient", "reasonable","customary",
        "usually", "approximately", "sufficiently","typically"
    ]
    text = text.split()
    for term in vague_terms:
        if term in text:
            return True
    else:
        return False


def eval_has_escape_clause(text:str) -> bool:
    """
    R8: Criteria from 4.1.8 INCOSE Guide to Writing Requirements:
        check if text contains escape clauses which state vague
        conditions or possibilities        
    """
    clauses = [
         "so far as is possible", "as little as possible", "where possible", 
         "as much as possible", "if it should prove necessary", "if necessary", 
         "to the extent necessary", "as appropriate", "as required", "to the extent practical", 
         "if practicable"
    ]
    for clause in clauses:
        if clause in text:
            return True
    else:
        return False
    

def eval_has_open_end_clause(text:str) -> bool:
    """
    R9: Criteria from 4.1.9 INCOSE Guide to Writing Requirements:
        check if text contains open-end clauses
    """
    clauses = [
          "including but not limited to", "etc.", "and so on",
    ]
    for clause in clauses:
        if clause in text:
            return True
    else:
        return False
    

def eval_has_superfl_inf(text:str) -> bool: # revise function to check for form "to" + "a verb"
    """
    R10: Criteria from 4.1.10 INCOSE Guide to Writing Requirements:
        check if text contains superfluous infinitives
    """
    superfl_inf = [
           "to be designed to", "to be able to", "to be capable of", "to enable", "to allow"
    ]

    for _inf in superfl_inf:
        if _inf in text:
            return True
    else:
        return False
    

def eval_has_combinators(text:str) -> bool: # revise function to check for form "to" + "a verb"
    """
    R19: Criteria from 4.4.2 INCOSE Guide to Writing Requirements:
        check if text contains  words that join or combine clauses
    """
    combinators = [
            "and", "or", "then", "unless", "but", "as well as", "but also", 
            "however", "whether", "meanwhile", "whereas", "on the other hand",
            "otherwise"
    ]
    text = text.split()
    for comb in combinators:
        if comb in text:
            return True
    else:
        return False