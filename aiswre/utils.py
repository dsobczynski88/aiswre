from __future__ import annotations
import os
import re
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Callable, Dict, Iterable, Mapping, Optional, Union, Sequence, Pattern
from pathlib import Path
from datetime import datetime
import flatdict
import ast
import yaml
import aiswre
from aiswre.prj_logger import get_logs

import pandas as pd


def load_prompt(prompt_base_bath: str, prompt_name: str, prompt_type: str = "system") -> str:
    """
    Load a prompt text file from ../src/prompts directory.

    Args:
        prompt_name: logical name of the prompt (e.g., 'test_A', 'test_prewarm')
        prompt_type: 'system' or 'user'

    Returns:
        The text content of the prompt file.
    """
    filename = f"{prompt_type}_{prompt_name}.txt"
    filepath = os.path.join(prompt_base_bath, filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Prompt file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

def concat_matching_dataframes(
    _path: Union[str, Path],
    _regex: str,
    *,
    recursive: bool = True,
    case_sensitive: bool = True,
    match_on: str = "name",  # "name" matches filename only, "path" matches full path
    ignore_index: bool = True,
    sort: bool = False,
    file_readers: Optional[Mapping[str, Callable[..., pd.DataFrame]]] = None,
    read_kwargs: Optional[Mapping[str, dict]] = None,
    check_list_like_columns: bool = False,
    axis: int = 0
    ) -> pd.DataFrame:
    """
    Scan a directory for files whose names match a regex, read them into DataFrames,
    and concatenate the results.
    Args:
        _path: Directory path to scan.
        _regex: Regular expression to match against file names or full paths.
        recursive: Whether to search subdirectories recursively.
        case_sensitive: If False, performs a case-insensitive match.
        match_on: "name" to apply regex to the filename only; "path" to apply to the full path.
        ignore_index: Passed to pandas.concat.
        sort: Passed to pandas.concat.
        file_readers: Optional mapping of file extensions to reader callables. Keys should be
                    lowercase extensions including the leading dot (e.g., ".csv").
                    If None, sensible defaults are used.
        read_kwargs: Optional mapping of file extensions to dicts of keyword args passed to
                    the corresponding reader (e.g., {".csv": {"dtype_backend": "pyarrow"}}).
        check_list_like_columns: If True, inspects string/object columns and converts values
                                that parse via ast.literal_eval into Python lists. Only columns
                                that plausibly contain list-like strings (e.g., "[1, 2]") are
                                attempted.
        axis: The way the concatenation should take place (0 for row-wise, 1 for column-wise)

    Returns:
        Concatenated DataFrame containing all rows from the matched files.

    Raises:
        FileNotFoundError: If the provided path does not exist.
        ValueError: If no files match the regex.
        RuntimeError: If all matching files were of unsupported types.
    """
    import re
    import ast
    from pathlib import Path
    from typing import Union, Optional, Mapping, Callable, Iterable
    import pandas as pd
    from pandas.api.types import is_string_dtype

    base = Path(_path).expanduser()
    if not base.exists():
        raise FileNotFoundError(f"Path does not exist: {base}")

    flags = 0 if case_sensitive else re.IGNORECASE
    pattern = re.compile(_regex, flags)

    # Default readers for common dataframe-friendly formats
    if file_readers is None:
        file_readers = {
            ".csv": pd.read_csv,
            ".tsv": lambda p, **kw: pd.read_csv(p, sep="\t", **kw),
            ".parquet": pd.read_parquet,
            ".feather": pd.read_feather,
            ".json": pd.read_json,
            ".xlsx": pd.read_excel,
            ".xls": pd.read_excel,
            ".pkl": pd.read_pickle,
            ".pickle": pd.read_pickle,
        }

    read_kwargs = read_kwargs or {}

    # Generator of candidate file paths
    file_iter: Iterable[Path] = base.rglob("*") if (base.is_dir() and recursive) else (
        base.glob("*") if base.is_dir() else [base]
    )

    # Filter by regex applied to name or full path
    def matches(p: Path) -> bool:
        target = p.name if match_on == "name" else str(p)
        return bool(pattern.search(target))

    candidates = [p for p in file_iter if p.is_file() and matches(p)]

    print(candidates)

    if not candidates:
        raise ValueError(
            f"No files matching regex {_regex!r} found in {base} "
            f"(recursive={recursive}, match_on={match_on})."
        )

    # Ensure deterministic order for reproducibility and caching
    candidates.sort(key=lambda p: str(p).lower())

    # Partition into supported/unsupported by extension
    supported_exts = set(file_readers.keys())
    supported_files = [p for p in candidates if p.suffix.lower() in supported_exts]
    unsupported_files = [p for p in candidates if p.suffix.lower() not in supported_exts]

    if not supported_files:
        raise RuntimeError(
            "All matched files have unsupported extensions. "
            f"Supported: {sorted(supported_exts)}. "
            f"Matched unsupported: {[p.suffix for p in unsupported_files]}"
        )

    # Efficiently stream-read and concat without creating large intermediate lists
    def _dfs() -> Iterable[pd.DataFrame]:
        for fp in supported_files:
            ext = fp.suffix.lower()
            reader = file_readers[ext]
            kwargs = read_kwargs.get(ext, {})
            # Recommend efficient defaults where sensible
            if ext in {".csv", ".tsv"} and "low_memory" not in kwargs:
                kwargs = {**kwargs, "low_memory": False}
            yield reader(fp, **kwargs)

    try:
        result = pd.concat(_dfs(), ignore_index=ignore_index, sort=sort, axis=axis, copy=False)
    except TypeError:
        # Older pandas may not support copy=...
        result = pd.concat(_dfs(), ignore_index=ignore_index, sort=sort, axis=axis)

    if check_list_like_columns and not result.empty:
        def _looks_like_list_str(x) -> bool:
            return isinstance(x, str) and x.strip().startswith("[") and x.strip().endswith("]")

        def _col_is_list_like(s: pd.Series) -> bool:
            # Only consider object/string-like columns
            if not (is_string_dtype(s) or s.dtype == object):
                return False
            mask = s.notna() & s.map(_looks_like_list_str)
            if not mask.any():
                return False
            sample = s[mask].head(20)
            hits = 0
            total = 0
            for v in sample:
                total += 1
                try:
                    parsed = ast.literal_eval(v)
                    if isinstance(parsed, list):
                        hits += 1
                except Exception:
                    pass
            return total > 0 and hits / total >= 0.6  # heuristic threshold

        def _safe_eval_to_list(val):
            if _looks_like_list_str(val):
                try:
                    parsed = ast.literal_eval(val)
                    if isinstance(parsed, list):
                        return parsed
                except Exception:
                    return val
            return val

        for col in result.columns:
            s = result[col]
            if _col_is_list_like(s):
                result[col] = s.map(_safe_eval_to_list)

    return result

def plot_failed_eval_diffs_by_prompt(
    df,
    requirement_id="requirement_id",
    prompt_id="prompt_id",
    rule_groups=None,
    bins=20,
    col_wrap=4,
    figsize=(14, 8),
    sharex=True,
    save_dir=None,
    show=True,
    dpi=120,
    dropna=True,
    ):
    """
    Deduplicate df on requirement_id (keeping the last occurrence), then plot a
    histogram of failed eval diffs per rule group for each prompt_id across all
    requirements.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing:
        - requirement_id column
        - prompt_id column
        - columns named like f"failed_evals_{rule_group}_diff" for each rule group
    requirement_id : str, default "requirement_id"
        Name of the requirement identifier column.
    prompt_id : str, default "prompt_id"
        Name of the prompt identifier column.
    rule_groups : list[str] | None
        List of rule groups to look for. If None, defaults to the provided set:
        [
            'Realism','Conditions','Singularity','Uniformity_Of_Language',
            'Concision','Modularity','Non_Ambiguity','Tolerance','Quantifiers',
            'Quantification','Completeness','Accuracy','Abstraction'
        ]
    bins : int, default 20
        Number of bins per histogram.
    col_wrap : int, default 4
        Maximum number of subplot columns per figure.
    figsize : tuple, default (14, 8)
        Figure size for each prompt's grid of histograms.
    sharex : bool, default True
        Whether subplots share the x-axis.
    save_dir : str | pathlib.Path | None
        If provided, saves one PNG per prompt_id into this directory.
    show : bool, default True
        If True, displays the figures via plt.show().
    dpi : int, default 120
        DPI for saved figures.
    dropna : bool, default True
        If True, drops NaN values from histograms.

    Returns
    -------
    dict
        Mapping: prompt_id -> matplotlib.figure.Figure
    """
    
    if rule_groups is None:
        rule_groups = [
            'Realism',
            'Conditions',
            'Singularity',
            'Uniformity_Of_Language',
            'Concision',
            'Modularity',
            'Non_Ambiguity',
            'Tolerance',
            'Quantifiers',
            'Quantification',
            'Completeness',
            'Accuracy',
            'Abstraction',
        ]

    # Basic validation
    missing_cols = [c for c in [requirement_id, prompt_id] if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing required columns in df: {missing_cols}")

    # Deduplicate on requirement_id, keeping the last occurrence by current order
    df_dedup = df.drop_duplicates(subset=[requirement_id], keep="last")

    # Collect diff columns present in the dataframe for the given rule_groups
    diff_cols = []
    for rg in rule_groups:
        prefix = f"failed_evals_{rg}_diff"
        present = [c for c in df_dedup.columns if c.startswith(prefix)]
        diff_cols.extend(present)

    # Deduplicate column list while preserving order
    seen = set()
    diff_cols = [c for c in diff_cols if not (c in seen or seen.add(c))]

    if not diff_cols:
        raise ValueError(
            "No columns found matching pattern f'failed_evals_{rule_group}_diff' "
            "for the provided rule_groups."
        )

    # Melt to long format for easier plotting
    long_df = df_dedup.melt(
        id_vars=[requirement_id, prompt_id],
        value_vars=diff_cols,
        var_name="metric",
        value_name="value",
    )

    # Extract the rule_group from the metric name
    # Expected form: "failed_evals_{rule_group}_diff..."
    def extract_rule_group(metric: str) -> str | None:
        for rg in rule_groups:
            if metric.startswith(f"failed_evals_{rg}_diff"):
                return rg
        return None

    long_df["rule_group"] = long_df["metric"].map(extract_rule_group)

    if dropna:
        long_df = long_df.dropna(subset=["value"])

    # Ensure save dir exists if provided
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    # Plot one figure per prompt_id, one histogram per rule_group
    figures = {}
    for pid, sub in long_df.groupby(prompt_id, sort=False):
        # Keep consistent rule_group order, showing only those present
        present_rgs = [rg for rg in rule_groups if rg in set(sub["rule_group"])]

        if len(present_rgs) == 0:
            # Nothing to plot for this prompt_id
            continue

        n_plots = len(present_rgs)
        ncols = min(col_wrap, n_plots)
        nrows = math.ceil(n_plots / ncols)

        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=figsize,
            sharex=sharex,
            constrained_layout=False,
        )
        if nrows * ncols == 1:
            axes = np.array([axes])
        axes = axes.ravel()

        # Determine global x limits if sharex requested
        if sharex:
            all_vals = sub["value"].to_numpy()
            if all_vals.size > 0:
                xmin, xmax = np.nanmin(all_vals), np.nanmax(all_vals)
                # Handle degenerate case
                if xmin == xmax:
                    xmin -= 0.5
                    xmax += 0.5
            else:
                xmin, xmax = None, None
        else:
            xmin = xmax = None

        for i, rg in enumerate(present_rgs):
            ax = axes[i]
            vals = sub.loc[sub["rule_group"] == rg, "value"].to_numpy()
            if vals.size == 0:
                ax.set_visible(False)
                continue

            ax.hist(vals, bins=bins, edgecolor="white", alpha=0.85)
            ax.set_title(rg)
            ax.set_xlabel("Failed evals diff")
            ax.set_ylabel("Count")

            # Optional reference line at zero if diffs can be signed
            try:
                ax.axvline(0, color="k", linestyle="--", alpha=0.5)
            except Exception:
                pass

            if sharex and xmin is not None and xmax is not None:
                ax.set_xlim(xmin, xmax)

        # Hide any unused axes
        for j in range(len(present_rgs), len(axes)):
            axes[j].set_visible(False)

        req_count = sub[requirement_id].nunique()
        fig.suptitle(f"Failed eval diffs per rule group for prompt_id={pid} (n={req_count})")
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

        if save_dir is not None:
            out_path = save_dir / f"failed_eval_diffs_prompt_{pid}.png"
            fig.savefig(str(out_path), dpi=dpi, bbox_inches="tight")

        figures[pid] = fig

    if show:
        import matplotlib.pyplot as _plt  # to avoid shadowing
        _plt.show()

    return figures

def yaml_loader(config_file='config.yml'):
    """
    Load a YAML file and return the contents.

    Parameters:
        config_file (str): String path to config file

    Returns:
        dict: The contents of the YAML file as a dictionary.
    """
    try:
        with open(config_file, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        return None
        
def yaml_writer(yaml_data, output_file='config.yml', sort_keys=False, indent=2):
    try:
        with open(output_file, 'w') as f:
            yaml.dump(yaml_data, f, sort_keys=sort_keys, indent=2)
    except FileNotFoundError:
        return None

def load_yaml(file_path):

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

def load_config(config_file='config.yaml', update_globals=False):
        # load config
        config = yaml_loader(config_file)
        if config is not None:
            if update_globals:
                globals().update(config)
            return config
        else:
            raise

def get_current_date_time():
    # Get the current date and time
    now = datetime.now()
    # Extract date, month, and time
    current_date = now.date()  # YYYY-MM-DD format
    current_month = now.month  # Numeric month (1-12)
    current_time = now.time()  # HH:MM:SS.microseconds format
    formatted_time = now.strftime("%Y-%m-%d-%H-%M-%S")
    return formatted_time  

def make_output_directory(file_locations, output_folder_name):
    run_name = f"run-{get_current_date_time()}"
    output_directory = f"{file_locations[output_folder_name]}/{run_name}"
    Path(output_directory).mkdir(parents=True, exist_ok=True)
    return output_directory

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

def flatten(df):
    df = flatten_df_series_dict(df, get_types_dict(df))
    dfs_to_add = []
    for col in df.columns:
        if '_flat' in col:
            dfs_to_add.append(pd.DataFrame(list(df[col].values)))
    df = pd.concat([df] + dfs_to_add, axis=1)
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


def to_excel(df, output_folder, _id, df_name):
    if _id:
        df.to_excel(f'{output_folder}/{df_name}_{_id}.xlsx')
    else:
        df.to_excel(f'{output_folder}/{df_name}.xlsx')

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


from pandas.api.types import is_numeric_dtype, is_bool_dtype

def check_failed_evals(
    df: pd.DataFrame,
    rg: str,
    output_col: Optional[str] = None,
    min_sum: float = 1,
    match_mode: str = "contains"
) -> Union[pd.Series, pd.DataFrame]:
    """
    For each row, check if the sum across columns whose names match the pattern
    'failed_evals_{rg}' is >= min_sum.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    rg : str
        The variable part of the pattern 'failed_evals_{rg}'. Can be a literal or a regex
        depending on match_mode.
    output_col : str, optional
        If provided, add the boolean result as this column to df and return the modified df.
        If None, return a boolean Series.
    min_sum : float, default 1
        Threshold that the row-wise sum must meet or exceed to be marked True.
    match_mode : {'contains', 'startswith', 'regex'}, default 'contains'
        How to match columns:
        - 'contains': column name contains the substring f"failed_evals_{rg}"
        - 'startswith': column name starts with f"failed_evals_{rg}"
        - 'regex': treat `rg` as a regex and match any column whose name matches
          r'failed_evals_{rg}'

    Returns
    -------
    pd.Series or pd.DataFrame
        Boolean Series if output_col is None; otherwise, the original DataFrame with
        an added boolean column.

    Raises
    ------
    ValueError
        If no columns match the pattern.
    TypeError
        If matched columns are not numeric or boolean.
    """
    base = "failed_evals_"
    if match_mode == "contains":
        pattern = f"{base}{rg}"
        cols = [c for c in df.columns if pattern in c]
    elif match_mode == "startswith":
        pattern = f"{base}{rg}"
        cols = [c for c in df.columns if c.startswith(pattern)]
    elif match_mode == "regex":
        regex = re.compile(fr"{base}{rg}")
        cols = [c for c in df.columns if regex.search(c)]
    else:
        raise ValueError("match_mode must be one of {'contains','startswith','regex'}")

    if not cols:
        raise ValueError(f"No columns matched the pattern using match_mode='{match_mode}' with rg='{rg}'")

    # Validate dtypes: require numeric or boolean
    bad = [c for c in cols if not (is_numeric_dtype(df[c]) or is_bool_dtype(df[c]))]
    if bad:
        raise TypeError(
            f"Matched columns must be numeric or boolean. Non-numeric columns found: {bad}. "
            "Convert them to numeric (e.g., df[bad] = df[bad].apply(pd.to_numeric, errors='coerce')) before calling."
        )

    sub = df[cols].copy()

    # Convert booleans to integers to ensure they contribute to the sum
    bool_cols = [c for c in cols if is_bool_dtype(sub[c])]
    if bool_cols:
        sub[bool_cols] = sub[bool_cols].astype("int8")

    # Sum across matched columns; skip NaNs
    row_sum = sub.sum(axis=1, numeric_only=True, skipna=True)

    mask = row_sum.ge(min_sum)
    print(mask)

    if output_col is not None:
        out = df.copy()
        out[output_col] = mask
        return out
    return mask

def drop_columns_by_regex(
    df: pd.DataFrame,
    patterns: Union[str, Pattern[str], Sequence[Union[str, Pattern[str]]]],
    *,
    flags: int = 0,
    how: str = "any",
    inplace: bool = False,
    errors: str = "ignore",
) -> Optional[pd.DataFrame]:
    """
    Drop columns whose names match regex pattern(s).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    patterns : str | Pattern[str] | sequence of these
        Regex pattern(s) to match against column names. Strings will be compiled
        with the provided `flags`. Compiled patterns are used as-is.
    flags : int, default 0
        Flags for regex compilation (e.g., re.IGNORECASE). Only used when compiling
        string patterns.
    how : {'any', 'all'}, default 'any'
        - 'any': drop columns that match at least one pattern.
        - 'all': drop columns that match all patterns.
    inplace : bool, default False
        If True, modify `df` in place and return None. Otherwise, return a copy.
    errors : {'ignore', 'raise'}, default 'ignore'
        If 'raise' and no columns match, raise ValueError. If 'ignore', do nothing.

    Returns
    -------
    Optional[pd.DataFrame]
        Returns a new DataFrame if inplace=False, otherwise None.

    Raises
    ------
    ValueError
        If `patterns` is empty, or if `errors='raise'` and no columns match.
    """
    # Normalize patterns input to a list
    if isinstance(patterns, (str, re.Pattern)):
        pat_list: list[Union[str, Pattern[str]]] = [patterns]
    else:
        pat_list = list(patterns)

    if not pat_list:
        raise ValueError("patterns must contain at least one regex pattern")

    if how not in {"any", "all"}:
        raise ValueError("how must be one of {'any', 'all'}")

    if errors not in {"ignore", "raise"}:
        raise ValueError("errors must be one of {'ignore', 'raise'}")

    # Compile string patterns; keep compiled ones as-is
    compiled: list[Pattern[str]] = []
    for p in pat_list:
        if isinstance(p, str):
            compiled.append(re.compile(p, flags))
        else:
            compiled.append(p)

    cols_str = df.columns.astype(str)

    # Initialize mask depending on 'how'
    mask = np.zeros(len(cols_str), dtype=bool) if how == "any" else np.ones(len(cols_str), dtype=bool)

    # Evaluate each pattern and combine
    for rp in compiled:
        m = cols_str.str.contains(rp)  # returns array-like (Series/ndarray/BooleanArray)
        m_arr = np.asarray(m, dtype=bool)  # robust conversion to numpy array
        if how == "any":
            mask |= m_arr
        else:
            mask &= m_arr

    to_drop = df.columns[mask]

    if len(to_drop) == 0:
        if errors == "raise":
            raise ValueError("No columns matched the provided regex pattern(s)")
        if inplace:
            return None
        return df.copy()

    if inplace:
        df.drop(columns=to_drop, inplace=True)
        return None
    return df.drop(columns=to_drop)


def save_graph_png(graph, output_path: Union[str, Path]) -> None:
    """
    Render a compiled LangGraph runnable as a Mermaid PNG and save it to disk.

    Uses LangGraph's built-in draw_mermaid_png() which calls the Mermaid.ink
    public API — requires an internet connection.

    Args:
        graph: A compiled LangGraph runnable (result of StateGraph.compile()).
        output_path: Destination path for the PNG file. Parent directories are
                     created automatically.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    png_bytes = graph.get_graph().draw_mermaid_png()
    output_path.write_bytes(png_bytes)
    print(f"Graph diagram saved to: {output_path}")