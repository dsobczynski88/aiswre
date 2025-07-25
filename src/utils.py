import pandas as pd
from pathlib import Path
from datetime import datetime
import flatdict
import ast
import yaml
import src
from src.prj_logger import get_logs


def load_yaml(file_path):
    """
    Load a YAML file and return the contents.

    Parameters:
    file_path (str): The path to the YAML file.

    Returns:
    dict: The contents of the YAML file as a dictionary.
    """
    with open(file_path, 'r') as file:
        try:
            data = yaml.safe_load(file)
            return data
        except yaml.YAMLError as e:
            print(f"Error loading YAML file: {e}")
            return None

def write_to_yaml(file_path, data):
    """
    Write data to an existing YAML file.

    Args:
        file_path (str): The path to the YAML file.
        data (dict): The data to write to the YAML file.
    """
    try:
        with open(file_path, 'r') as file:
            # Load existing data
            existing_data = yaml.safe_load(file) or {}
        
        # Update existing data with new data
        existing_data.update(data)

        # Write the updated data back to the YAML file
        with open(file_path, 'w') as file:
            yaml.dump(existing_data, file, default_flow_style=False)

        print(f"Successfully written to {file_path}")

    except FileNotFoundError:
        print(f"Error: The file {file_path} does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")

def get_current_date_time():
    # Get the current date and time
    now = datetime.now()
    # Extract date, month, and time
    current_date = now.date()  # YYYY-MM-DD format
    current_month = now.month  # Numeric month (1-12)
    current_time = now.time()  # HH:MM:SS.microseconds format
    formatted_time = now.strftime("%Y-%m-%d-%H-%M-%S")
    return formatted_time  

@get_logs(src.BASE_LOGGERNAME)
def map_A_to_B(list_of_A:list, mapdict_AB:dict) -> list:
    """This function takes in a list (call as A) and a 
    dictionary who keys include elements of the list A. Using
    this dictionary and the input list A, an output list is generaed
    where the original elements of the list A are mapped to the 
    values of the dictionary provided for each key.
    
    Args:
        list_of_A (list): an input list
        mapdict_AB (dict): a dictionary with keys corresponding to the input list elements
    """
    return [*map(mapdict_AB.get, list_of_A)]

@get_logs(src.BASE_LOGGERNAME)
def get_types_dict(df: pd.DataFrame) -> dict:
    """This function takes in a dataframe and gets the data type name of each column.
    For example, if the type of a specific column is a list, then the values of that 
    column would be equal to 'list'. Columns of type 'list' or those with mixed data
    types are printed to the terminal. 

    Args:
        _df (pd.DataFrame): Pandas dataframe
    """    
    df_types = df.apply(lambda series: (series.apply(lambda val: type((val)).__name__)))    
    types_dict = {}
    for column in df_types.columns:
        column_types = list(df_types[column].unique())
        if len(column_types) == 1:
            column_types = column_types[0]
            types_dict[column] = column_types
            # log in future
            if column_types == 'list':
                print(f'Column {column} is of data type:{column_types}')
        else:
            # log in future
            print(f'Column {column} has multiple data types: {column_types}')
            continue
    return types_dict

@get_logs(src.BASE_LOGGERNAME)
def flatten_df_series_dict(df: pd.DataFrame, types_dict:dict) -> pd.DataFrame:
    """This function takes in a dataframe and a dictionary of type names. For 
    those columns of the dataframe of type 'dict', the Flatdict class is called
    on this series. The result is a flatten dictionary which can then be 
    converted into a dataframe using the dict_list_to_df function.

    Args:
        df (pd.DataFrame): Pandas dataframe
        types_dict (dict): Dictionary of data types
    """
    for column, column_type in types_dict.items():
        if column_type == 'dict':
            df[f'{column}_flat'] = df[column].apply(lambda s: flatdict.FlatDict(s, delimiter='.'))   
    return df

@get_logs(src.BASE_LOGGERNAME)
def flatten(df):
    df = flatten_df_series_dict(df, get_types_dict(df))
    dfs_to_add = []
    for col in df.columns:
        if '_flat' in col:
            dfs_to_add.append(pd.DataFrame(list(df[col].values)))
    df = pd.concat([df] + dfs_to_add, axis=1)
    return df

@get_logs(src.BASE_LOGGERNAME)
def replace_null(df:pd.DataFrame, colname:str, replace_with:str) -> pd.DataFrame:
    """
    This function will take in a dataframe and a specific column name and 
    apply the loc accessor to replace null values in that series with the specified
    string via the replace_with argument
    
    Args:
        df (pd.DataFrame): a dataframe
        colname: (str): the name of the the column to replace nulls
        replace_with (str): the str to replace the nulls of column named colname
    """
    df.fillna({colname: replace_with}, inplace=True)
    return df

@get_logs(src.BASE_LOGGERNAME) 
def recast_str(_str:str, na_value=[]):
    """This function takes in a str and default value for errors or NaNs. The built-in
    python function eval() is applied on the input string in effort to cast the string
    to some expected data type (e.g., list, dict). This is particularly useful as exporting
    pandas dataframes to excel may result in loss of data typing and this is a way to 
    recover this infomation. In the event there is a type or syntax error with the eval()
    function, the na_value is returned.

    """ 
    if type(_str) == float:
        return na_value
    if str(_str) == 'nan':
        return na_value
    else:
        try:
            casted = ast.literal_eval(_str)
        except SyntaxError:
            print(f'The following string was unable to be casted using eval()')
            print(f'The value will be converted to data type: {type(na_value)}')
            return na_value
        except TypeError:
            print(f'The input data type: {type(_str).__name__} cannot be evaluated')
            return na_value
        except NameError:
            return na_value
        else:
            return casted

@get_logs(src.BASE_LOGGERNAME)
def to_excel(df, output_folder, _id, df_name):
    if _id:
        df.to_excel(f'{output_folder}/{df_name}_{_id}.xlsx')
    else:
        df.to_excel(f'{output_folder}/{df_name}.xlsx')

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
    to_excel(revisions_df, op, False, 'revisions_df')
    return revisions_df
    
@get_logs(src.BASE_LOGGERNAME)
def merge_revisions_df(op, reqs_df, revisions_df, requirement_col='Requirement', revision_number_col='revision'):
    #merge latest revisions to original requirements dataframe
    revisions_df = revisions_df.sort_values(by=[f'{requirement_col}_#', revision_number_col], ascending=True).drop_duplicates(subset=[f'{requirement_col}_#'], keep='last').reset_index()
    reqs_df = pd.merge(
        left=reqs_df, right=revisions_df[[f'Revised_{requirement_col}',f'{requirement_col}_#']], on=f'{requirement_col}_#', how='left'
    )
    return reqs_df
    
def mk_dict_from_df(df:pd.DataFrame, cols_to_keep:list) -> dict:
    """Takes in a pandas dataframe and a list of columns and returns
    a dictionary where the first column in the list is the keys and 
    the second column the values

    Args:
        df (pd.DataFrame): Pandas dataframe
        cols_to_keep (list): Two columns from the dataframe desired 
            to use as keys and values of the output dict
    """
    return dict(df[cols_to_keep].drop_duplicates(subset=cols_to_keep).values)