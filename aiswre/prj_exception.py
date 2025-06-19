'''
Define custom exception handling
'''

import sys
import logging
from pathlib import Path

def exception_logger(loggername):       
    def decorator(func):
        def wrapper(*args, **kwargs):          
            try:
                return func(*args, **kwargs)
            except Exception as e:
                ce = CustomException(e, sys)
                logger = logging.getLogger(loggername)               
                logger.debug(ce.error_message)
                #raise ce
                return None
        return wrapper
    return decorator

def parse_error_traceback(error_detail:sys):
    _,_,exc_tb=error_detail.exc_info()
    return exc_tb

def get_error_message(error, type, tb):   
    error_message=f"{type}:{error} occurred at line {tb.tb_lineno} in {Path(tb.tb_frame.f_code.co_filename).name}"
    return error_message

class CustomException(Exception):
    def __init__(self, error, error_detail:sys):
        super().__init__(error)
        self.tb = parse_error_traceback(error_detail)
        self.error_type = type(error).__name__
        self.error_message = get_error_message(error, self.error_type, self.tb)

    def __str__(self):
        return self.error_message
     
