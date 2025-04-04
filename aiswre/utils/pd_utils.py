import pandas as pd
import flatdict
        
def flatten(df):
    df = flatten_df_series_dict(df, get_types_dict(df))
    dfs_to_add = []
    for col in df.columns:
        if '_flat' in col:
            dfs_to_add.append(pd.DataFrame(list(df[col].values)))
    df = pd.concat([df] + dfs_to_add, axis=1)
    return df

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
    
    
def mk_df_from_dict(_dict: dict) -> pd.DataFrame:
    """
    This function takes in a dictionary and converts it to a dataframe where each 
    unique key in the dictionary is a column of the dataframe
    
    Args:
        _dict (dict): a dictionary to be converted to a dataframe.
    """
    item_list = []
    item_fields = set()
    for _, item in enumerate(_dict):
        item_fields.update(list(item.keys()))
        item_list.append(item)
    return pd.DataFrame(data=item_list, columns=list(item_fields))

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

def to_excel(df, output_folder, _id, df_name):
    if _id:
        df.to_excel(f'{output_folder}/{df_name}_{_id}.xlsx')
    else:
        df.to_excel(f'{output_folder}/{df_name}.xlsx')