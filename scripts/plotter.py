import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional, Tuple, Dict, Iterable, Union, Sequence, Any, Mapping
from matplotlib.axes import Axes
from matplotlib.figure import Figure

def plot_grouped_histograms(
    df: pd.DataFrame,
    groupby_cols: Sequence[str] = ("prompt_type",),
    value_col: str = "weighted_value",
    bins: Union[int, str, Iterable[float]] = "auto",
    density: bool = False,
    alpha: float = 0.5,
    histtype: str = "stepfilled",
    edgecolor: Optional[str] = "white",
    linewidth: float = 0.5,
    colormap: str = "Pastel1",
    ax: Optional[Axes] = None,
    figsize: Tuple[int, int] = (10, 6),
    title: Optional[str] = None,
    # New options
    show_median_text: bool = True,
    show_median_vlines: bool = True,
    legend_include_median: bool = True,
    median_fmt: str = "{:.3g}",
    median_text_loc: Tuple[float, float] = (0.02, 0.98),
    median_text_kwargs: Optional[Dict] = None,
    median_line_kwargs: Optional[Dict] = None,
    ) -> Tuple[Figure, Axes]:
    """
    Plot shaded, overlaid histograms for each group defined by groupby_cols.

    - Groups df by groupby_cols
    - Plots one (filled) histogram per group using values from value_col
    - Legend shows one entry per group (combination of groupby_cols), optionally including median
    - Adds a vertical line at the median of value_col for each group (matching the histogram color)
    - Adds a text box summarizing medians per group

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    groupby_cols : sequence of str
        Column(s) to group by. Default ("prompt_type",).
    value_col : str
        Column name for the numeric values to histogram (default "weighted_value").
    bins : int | str | array-like
        Histogram bins. If int or str, common bin edges are computed across all data.
    density : bool
        If True, plot density instead of counts.
    alpha : float
        Transparency for filled histograms (0..1). Lower values help visualize overlap.
    histtype : str
        Matplotlib histtype. Use "stepfilled" (default) for shaded/overlaid look.
    edgecolor : str | None
        Edge color for bars. Use None to disable edges.
    linewidth : float
        Edge line width.
    colormap : str
        Name of a matplotlib colormap to draw group colors from.
    ax : matplotlib.axes.Axes | None
        Existing axes to draw on. If None, a new figure/axes is created.
    figsize : tuple
        Figure size if ax is None.
    title : str | None
        Optional title. If None, a default is created.
    show_median_text : bool
        If True, shows a text box with median per group.
    show_median_vlines : bool
        If True, draws a vertical line at the median for each group.
    legend_include_median : bool
        If True, appends the median value to each legend label.
    median_fmt : str
        Format string for median values, e.g., "{:.3g}" or "{:,.3f}".
    median_text_loc : (float, float)
        Axes-relative location (x,y) for the median text box.
    median_text_kwargs : dict | None
        Additional kwargs for ax.text when drawing the median text box.
    median_line_kwargs : dict | None
        Additional kwargs for ax.axvline when drawing median lines.

    Returns
    -------
    (fig, ax)
    """
    # Validate inputs
    if not groupby_cols:
        raise ValueError("groupby_cols must contain at least one column name.")
    missing = [c for c in list(groupby_cols) + [value_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in df: {missing}")

    # Prepare data: coerce value to numeric, drop NaNs in any grouping/value col
    cols_needed = list(groupby_cols) + [value_col]
    work = df[cols_needed].copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.dropna(subset=cols_needed)
    if work.empty:
        raise ValueError("No data to plot after filtering/NaN handling.")

    # Helper to create legend label from group key
    def _label_from_key(key) -> str:
        if isinstance(key, tuple):
            if len(groupby_cols) == 1:
                return str(key[0])
            return ", ".join(f"{c}={v}" for c, v in zip(groupby_cols, key))
        else:
            # Single-column groupby returns scalar key
            return str(key)

    # Prepare axes
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True
    else:
        fig = ax.figure

    # Compute common bin edges for consistent comparison across groups
    if isinstance(bins, (int, str)):
        all_vals = work[value_col].to_numpy()
        bin_edges = np.histogram_bin_edges(all_vals, bins=bins)
    else:
        bin_edges = np.asarray(list(bins), dtype=float)

    # Build list of groups and labels (preserve order of appearance)
    grouped = work.groupby(list(groupby_cols), sort=False)
    groups = []
    labels = []
    for key, g in grouped:
        vals = g[value_col].to_numpy()
        if vals.size == 0:
            continue
        groups.append(vals)
        labels.append(_label_from_key(key))
    if not groups:
        raise ValueError("No non-empty groups to plot.")

    # Compute medians per group
    medians: Dict[str, float] = {lab: float(np.median(vals)) for lab, vals in zip(labels, groups)}

    # Assign colors: one per group label
    cmap = plt.get_cmap("tab20" if colormap is None else colormap)
    color_map = {lab: cmap(i % cmap.N) for i, lab in enumerate(labels)}

    # Plot each group as shaded, overlaid histogram (legend label can include median)
    for vals, lab in zip(groups, labels):
        legend_label = f"{lab} (med={median_fmt.format(medians[lab])})" if legend_include_median else lab
        ax.hist(
            vals,
            bins=bin_edges,
            density=density,
            histtype=histtype,  # "stepfilled" by default for shading
            alpha=alpha,
            color=color_map[lab],
            edgecolor=edgecolor,
            linewidth=linewidth,
            label=legend_label,
        )

    # Add vertical median lines for each group, matching histogram colors
    if show_median_vlines:
        vline_kwargs = dict(linestyle="--", linewidth=2.0, alpha=min(1.0, alpha + 0.4))
        if median_line_kwargs:
            vline_kwargs.update(median_line_kwargs)
        for lab in labels:
            ax.axvline(
                medians[lab],
                color=color_map[lab],
                label="_nolegend_",
                **vline_kwargs,
            )

    # Add a text box with medians
    if show_median_text:
        txt_kwargs = dict(
            ha="left",
            va="top",
            fontsize=9,
            transform=ax.transAxes,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="gray"),
        )
        if median_text_kwargs:
            # Merge/override defaults but keep transform unless explicitly provided
            base_bbox = txt_kwargs.get("bbox", {}).copy()
            txt_kwargs.update({k: v for k, v in median_text_kwargs.items() if k != "bbox"})
            if "bbox" in median_text_kwargs:
                bb = base_bbox
                bb.update(median_text_kwargs["bbox"])
                txt_kwargs["bbox"] = bb

        header = f"Medians of {value_col}"
        lines = [f"- {lab}: {median_fmt.format(medians[lab])}" for lab in labels]
        text = "\n".join([header] + lines)
        ax.text(median_text_loc[0], median_text_loc[1], text, **txt_kwargs)

    # Labels, title, legend, grid
    ax.set_xlabel(value_col)
    ax.set_ylabel("Density" if density else "Count")
    if title is None:
        if len(groupby_cols) == 1:
            title = f"Histograms of {value_col} grouped by {groupby_cols[0]}"
        else:
            title = f"Histograms of {value_col} grouped by {', '.join(groupby_cols)}"
    ax.set_title(title)
    ax.legend(title=", ".join(groupby_cols))
    ax.grid(True, alpha=0.3)

    if created_fig:
        fig.tight_layout()

    return fig, ax

def plot_grouped_histograms(
    df: pd.DataFrame,
    groupby_cols: Sequence[str] = ("prompt_type",),
    value_col: str = "weighted_value",
    alpha: float = 0.9,
    edgecolor: Optional[str] = "white",
    linewidth: float = 0.5,
    colormap: str = "Pastel1",
    ax: Optional[Axes] = None,
    figsize: Tuple[int, int] = (10, 6),
    title: Optional[str] = None,
    normalize: bool = False,
    ) -> Tuple[Figure, Axes]:
    """
    Plot stacked bar charts for each group defined by groupby_cols.

    - Groups df by groupby_cols (preserving order of appearance)
    - For each group on the x-axis, stacks counts of each unique label from value_col
    - Legend shows one entry per unique value in value_col (the stacked segments)

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    groupby_cols : sequence of str
        Column(s) to group by along the x-axis. Default ("prompt_type",).
        If multiple columns are provided, their combinations are used as groups.
    value_col : str
        Column name for the categorical values used to form the stacks.
    alpha : float
        Transparency for bars (0..1).
    edgecolor : str | None
        Edge color for bar segments. Use None to disable edges.
    linewidth : float
        Edge line width for bar segments.
    colormap : str
        Name of a matplotlib colormap to draw value colors from.
    ax : matplotlib.axes.Axes | None
        Existing axes to draw on. If None, a new figure/axes is created.
    figsize : tuple
        Figure size if ax is None.
    title : str | None
        Optional title. If None, a default is created.
    normalize : bool
        If True, each stacked bar shows proportions (0..1) instead of raw counts.

    Returns
    -------
    (fig, ax)
    """
    # Validate inputs
    if not groupby_cols:
        raise ValueError("groupby_cols must contain at least one column name.")
    missing = [c for c in list(groupby_cols) + [value_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in df: {missing}")

    # Prepare data: drop NaNs in any grouping/value col
    cols_needed = list(groupby_cols) + [value_col]
    work = df[cols_needed].copy()
    work = work.dropna(subset=cols_needed)
    if work.empty:
        raise ValueError("No data to plot after filtering/NaN handling.")

    # Determine order of categories (values) by first appearance
    categories = pd.unique(work[value_col])
    # Make value column categorical to preserve order during unstack
    work[value_col] = pd.Categorical(work[value_col], categories=categories, ordered=True)

    # Compute counts per group and value
    grouped_counts = (
        work.groupby(list(groupby_cols) + [value_col], sort=False)
        .size()
        .unstack(value_col, fill_value=0)
    )

    if grouped_counts.empty or len(grouped_counts.columns) == 0:
        raise ValueError(f"No categories found in '{value_col}' to plot.")

    # Optional normalization to proportions
    if normalize:
        denom = grouped_counts.sum(axis=1).replace(0, np.nan)
        grouped_counts = grouped_counts.div(denom, axis=0).fillna(0.0)

    # Helper to create label from group key
    def _label_from_key(key) -> str:
        if isinstance(key, tuple):
            if len(groupby_cols) == 1:
                return str(key[0])
            return ", ".join(f"{c}={v}" for c, v in zip(groupby_cols, key))
        else:
            return str(key)

    # Prepare x-axis labels for groups
    index_keys = list(grouped_counts.index)
    x_labels = [_label_from_key(k) for k in index_keys]
    n_groups = len(x_labels)
    x = np.arange(n_groups)

    # Prepare axes
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True
    else:
        fig = ax.figure

    # Colors for each category in value_col
    n_cats = len(categories)
    cmap = plt.get_cmap("tab20" if colormap is None else colormap)
    colors = [cmap(i % cmap.N) for i in range(n_cats)]

    # Plot stacked bars
    bottoms = np.zeros(n_groups, dtype=float)
    for j, cat in enumerate(categories):
        heights = grouped_counts[cat].to_numpy()
        ax.bar(
            x,
            heights,
            bottom=bottoms,
            color=colors[j],
            alpha=alpha,
            edgecolor=edgecolor,
            linewidth=linewidth,
            label=str(cat),
        )
        bottoms += heights

    # Labels, title, legend, grid
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=0)
    x_label = groupby_cols[0] if len(groupby_cols) == 1 else ", ".join(groupby_cols)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Proportion" if normalize else "Count")
    if title is None:
        if len(groupby_cols) == 1:
            title = f"Stacked counts of {value_col} by {groupby_cols[0]}"
        else:
            title = f"Stacked counts of {value_col} by {', '.join(groupby_cols)}"
    ax.set_title(title)
    ax.legend(title=value_col)
    ax.grid(axis="y", alpha=0.3)

    if created_fig:
        fig.tight_layout()

    return fig, ax

def plot_grouped_stackedbars(
    df: pd.DataFrame,
    groupby_cols: Sequence[str] = ("prompt_type",),
    value_col: str = "weighted_value",
    alpha: float = 0.9,
    edgecolor: Optional[str] = "white",
    linewidth: float = 0.5,
    colormap: str = "Pastel1",
    ax: Optional[Axes] = None,
    figsize: Tuple[int, int] = (10, 6),
    title: Optional[str] = None,
    normalize: bool = False,
    value_colors: Optional[Union[Sequence[Any], Mapping[Any, Any]]] = None,
    group_order: Optional[Union[Sequence[Any], Mapping[Any, int]]] = None,
    legend_categories_order: Union[bool, Sequence[Any]] = False,
    ) -> Tuple[Figure, Axes]:
    """
    Plot stacked bar charts for each group defined by groupby_cols.

    - Groups df by groupby_cols (preserving order of appearance unless group_order is provided)
    - For each group on the x-axis, stacks counts of each unique label from value_col
    - Legend shows one entry per unique value in value_col (the stacked segments)

    Parameters added:
    - value_colors: list or dict to set colors per category in value_col.
        • If dict: {category_value: color_spec}
        • If list/sequence: colors applied in the order of categories as they appear.
        Extra colors are ignored; missing colors fall back to colormap.
    - group_order: list or dict to control the order of groups on the x-axis.
        • If list/sequence: order is taken as-is. Values must match the group key:
            - If len(groupby_cols)==1: the scalar group values (e.g., "A", "B", ...)
            - If len(groupby_cols)>1: tuples of group values (e.g., ("A", "x"), ("B", "y"))
        Any groups not specified are appended afterward in their original order.
        • If dict: {group_key: rank_int} lower rank appears first. Unspecified groups
        are appended afterward in their original order.
    - legend_categories_order: False or sequence to control both legend and stacking order.
        • If False (default): legend and stacking follow the original plotting order
        (first encountered category is at the bottom of the stack and first in legend).
        • If sequence: interpreted as top-to-bottom legend order. The stacked bars will
        follow the same top-to-bottom order, meaning the first element in the sequence
        will appear as the top segment in each stacked bar. Any categories not listed
        are appended after the provided sequence (i.e., they will appear below the
        specified categories in the stack) and keep their original relative order.
    """
    # Validate inputs
    if not groupby_cols:
        raise ValueError("groupby_cols must contain at least one column name.")
    missing = [c for c in list(groupby_cols) + [value_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in df: {missing}")

    # Prepare data: drop NaNs in any grouping/value col
    cols_needed = list(groupby_cols) + [value_col]
    work = df[cols_needed].copy()
    work = work.dropna(subset=cols_needed)
    if work.empty:
        raise ValueError("No data to plot after filtering/NaN handling.")

    # Determine order of categories (values) by first appearance
    categories_seen = pd.unique(work[value_col])
    # Make value column categorical to preserve order during unstack
    work[value_col] = pd.Categorical(work[value_col], categories=categories_seen, ordered=True)

    # Compute counts per group and value
    grouped_counts = (
        work.groupby(list(groupby_cols) + [value_col], sort=False)
        .size()
        .unstack(value_col, fill_value=0)
    )

    if grouped_counts.empty or len(grouped_counts.columns) == 0:
        raise ValueError(f"No categories found in '{value_col}' to plot.")

    # Optional normalization to proportions
    if normalize:
        denom = grouped_counts.sum(axis=1).replace(0, np.nan)
        grouped_counts = grouped_counts.div(denom, axis=0).fillna(0.0)

    # Apply group ordering if requested
    if group_order is not None:
        idx = list(grouped_counts.index)

        def _as_rank_map(go):
            if isinstance(go, Mapping):
                return dict(go)
            seq = list(go)
            return {k: i for i, k in enumerate(seq)}

        rank_map = _as_rank_map(group_order)
        specified = [k for k in idx if k in rank_map]
        specified.sort(key=lambda k: rank_map[k])
        unspecified_idx = [k for k in idx if k not in rank_map]
        new_order = specified + unspecified_idx
        grouped_counts = grouped_counts.loc[new_order]

    # Helper to create label from group key
    def _label_from_key(key) -> str:
        if isinstance(key, tuple):
            if len(groupby_cols) == 1:
                return str(key[0])
            return ", ".join(f"{c}={v}" for c, v in zip(groupby_cols, key))
        else:
            return str(key)

    # Prepare x-axis labels for groups
    index_keys = list(grouped_counts.index)
    x_labels = [_label_from_key(k) for k in index_keys]
    n_groups = len(x_labels)
    x = np.arange(n_groups)

    # Prepare axes
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True
    else:
        fig = ax.get_figure()

    # Categories present in the data (order as in columns)
    categories = list(grouped_counts.columns)
    n_cats = len(categories)

    # Colors for each category in value_col
    cmap = plt.get_cmap("tab20" if colormap is None else colormap)
    default_colors = [cmap(i % cmap.N) for i in range(n_cats)]

    # Build final color mapping per category
    if value_colors is not None:
        if isinstance(value_colors, Mapping):
            color_by_cat = {
                cat: value_colors.get(cat, default_colors[j]) for j, cat in enumerate(categories)
            }
        else:
            seq = list(value_colors)
            color_by_cat = {
                cat: (seq[j] if j < len(seq) else default_colors[j]) for j, cat in enumerate(categories)
            }
    else:
        color_by_cat = {cat: default_colors[j] for j, cat in enumerate(categories)}

    # Determine legend and stacking orders
    if legend_categories_order is False or legend_categories_order is None:
        # Original behavior: plot and legend follow the original category order.
        legend_order = list(categories)  # top-to-bottom legend equals plotting order (bottom-to-top)
        plot_order_bottom_to_top = list(categories)
    else:
        # Normalize and validate requested order (top-to-bottom)
        try:
            req_seq = list(legend_categories_order)
        except TypeError:
            raise TypeError("legend_categories_order must be False or a sequence of category values.")

        # Deduplicate while preserving order
        seen = set()
        requested_top = [c for c in req_seq if (c in categories) and (not (c in seen or seen.add(c)))]
        unspecified = [c for c in categories if c not in requested_top]

        # Legend order: top-to-bottom as requested, then unspecified
        legend_order = requested_top + unspecified

        # Stacking order: bottom-to-top so that requested appear on top in the same order.
        # Place unspecified at the bottom (preserving their original relative order),
        # then requested in reverse so the first requested becomes the very top.
        plot_order_bottom_to_top = unspecified + list(reversed(requested_top))

    # Plot stacked bars following the desired bottom-to-top order
    bottoms = np.zeros(n_groups, dtype=float)
    for cat in plot_order_bottom_to_top:
        heights = grouped_counts[cat].to_numpy()
        ax.bar(
            x,
            heights,
            bottom=bottoms,
            color=color_by_cat[cat],
            alpha=alpha,
            edgecolor=edgecolor,
            linewidth=linewidth,
            label=str(cat),
        )
        bottoms += heights

    # Labels, title, legend, grid
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=0)
    x_label = groupby_cols[0] if len(groupby_cols) == 1 else ", ".join(groupby_cols)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Proportion" if normalize else "Count")
    if title is None:
        if len(groupby_cols) == 1:
            title = f"Stacked counts of {value_col} by {groupby_cols[0]}"
        else:
            title = f"Stacked counts of {value_col} by {', '.join(groupby_cols)}"
    ax.set_title(title)

    # Build legend with the chosen order (top-to-bottom)
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(
            facecolor=color_by_cat[cat],
            edgecolor=edgecolor,
            linewidth=linewidth,
            alpha=alpha,
            label=str(cat),
        )
        for cat in legend_order
    ]
    ax.legend(handles=legend_handles, title=value_col)

    ax.grid(axis="y", alpha=0.3)

    if created_fig and hasattr(fig, "tight_layout"):
        fig.tight_layout()

    return fig, ax

from dotenv import dotenv_values
from src import utils
import json
import flatdict
from src import utils

# Load config settings
DOT_ENV = dotenv_values("../.env")
config = utils.load_config("../config.yaml")
run_directory = 'run-2025-10-15-21-30-23'
output_directory = f'../src/data/'

from src.components import prompteval as pe

revisions_df = pd.read_excel(f'{output_directory}/revisions_df_1_prompt_H_o4_mini.xlsx')
id_col = 'requirement_id'
requirement_col_post_revision = 'proposed_rewrite'
rule_groups_to_evaluate = ['Accuracy']
include_funcs = [
    'eval_avoids_vague_terms',
    'eval_definite_articles_usage',
    'eval_has_appropriate_subject_verb',
    'eval_has_common_units_of_measure',
    'eval_has_escape_clauses',
    'eval_has_no_open_ended_clauses',
    'eval_is_active_voice',
]

score_weights = [
    0.35,
    0.05,
    0.15,
    0.05,
    0.10,
    0.10,
    0.20
]
initial_df = pd.read_excel(f'{output_directory}/run-2025-10-17-15-47-20/eval_df_iter_prompt_H_0.xlsx')
initial_weighted_values = pe.add_weighted_column(initial_df, include_funcs, score_weights, "weighted_value_initial")
initial_requirement_ids = list(initial_df[id_col].values)
initial_weighted_values_dict = dict(zip(initial_requirement_ids, initial_weighted_values))
# Make evaluation function config
eval_config = pe.make_eval_config(pe, include_funcs=include_funcs)

# Compile all results
all_results_df = utils.concat_matching_dataframes(
    _path=output_directory,                     # base directory to scan
    _regex=rf"revisions_df.*.xlsx$",             # regex applied to filenames
    recursive=False,
    case_sensitive=True,
    match_on="name",
    read_kwargs=None,
    check_list_like_columns=True)

def get_result_label(x):
    if x < 0:
        return "Worse"
    elif x == 0:
        return "Same"
    elif x > 0:
        return "Better"
    else:
        return "Error: Unknown"


eval_df = pe.call_evals(all_results_df, col=requirement_col_post_revision, eval_config=eval_config)
# Get list of failed eval functions
eval_df = pe.get_failed_evals(eval_df)
post_weighted_values = pe.add_weighted_column(eval_df, include_funcs, score_weights, "weighted_value")
#eval_df.to_excel(f'{output_directory}/eval_df_H_o4mini.xlsx')
# Map the failed eval functions to rule groups (as defined in the config.yaml file)
eval_df = pe.map_failed_eval_col_to_rule_group(eval_df, eval_to_rule_map=config["SECTION_4_RULE_GROUPS"], failed_eval_col='failed_evals')
# Drop requirements which pass acceptance criteria
# At present, the criteria is len(failed_evals_rule_ids) == 0
eval_df['initial_weighted_values'] = eval_df[id_col].map(initial_weighted_values_dict)
#eval_df.to_excel(f'{output_directory}/compiled_df_H_o4mini.xlsx')

eval_df['Accuracy_change'] = eval_df['weighted_value'] - eval_df['initial_weighted_values']
eval_df['Accuracy_Result'] = eval_df['Accuracy_change'].map(get_result_label)

def common_requirement_ids_by_prompt_type(
    df: pd.DataFrame,
    requirement_col: str = "requirement_id",
    prompt_col: str = "prompt_type",
    rewrite_col: str = "proposed_rewrite",
    ):
    """
    Filters out rows with missing proposed_rewrite and finds requirement IDs
    that are present in every prompt_type group.

    Returns:
        filtered_df: DataFrame after filtering
        common_ids: list of requirement IDs present across all prompt_type groups
    """
    # Remove rows where proposed_rewrite is None/NaN
    filtered_df = df[df[rewrite_col].notna()].copy()

    # Optionally drop rows with missing requirement IDs (cannot be common if missing)
    filtered_df = filtered_df[filtered_df[requirement_col].notna()].copy()

    # Number of distinct prompt_type groups remaining
    n_groups = filtered_df[prompt_col].nunique(dropna=True)

    if n_groups == 0:
        return filtered_df, []

    # Count in how many distinct prompt_type groups each requirement_id appears
    counts = (
        filtered_df.groupby(requirement_col)[prompt_col]
        .nunique(dropna=True)
    )

    # Requirement IDs that appear in all groups
    common_ids = counts[counts == n_groups].index.tolist()

    return filtered_df, common_ids

eval_df, common_ids = common_requirement_ids_by_prompt_type(eval_df)
eval_df = eval_df[eval_df[id_col].isin(common_ids)]

labels_to_keep = {
    "prompt_prewarm":"PW",
    "prompt_A":"BI",
    "prompt_C":"RIF",
    "prompt_G":"IFE",
    "prompt_H":"RIFE",
    "prompt_H_o4mini":"RIFE+o4-mini",
}

eval_df['prompt_type'] = eval_df['prompt_type'].map(labels_to_keep)

eval_df = eval_df[eval_df['prompt_type'].isin(list(labels_to_keep.values()))]

eval_df.to_excel(f"{output_directory}/combined_eval_df-recalc-test.xlsx")

#initial_values_df = pd.DataFrame(
#    {"weighted_value": list(all_results_df.drop_duplicates(subset=[requirement_id_col])['initial_weighted_values'].values)}
#)
#initial_values_df["prompt_type"] = "baseline"
#all_results_df = pd.concat([all_results_df, initial_values_df], axis=0)

# Plot bar charts
import plotter
import matplotlib.pyplot as plt

partial_value_colors = {"Worse":"#e33f3f", "Better": "#0dc20a", "Same": "#1f77b4"}  # Only override Positive
#partial_group_order = [
#    ("prompt_prewarm"),
#    ("prompt_A"),
#    ("prompt_C"),
#    ("prompt_D"),
#    ("prompt_E"),
#    ("prompt_F"),
#    ("prompt_G"),
#    ("prompt_H"),
#    ("prompt_H_o4mini"),
#]
partial_group_order = [
    ("PW"),
    ("BI"),
    ("RIF"),
    ("IFE"),
    ("RIFE"),
    ("RIFE+o4-mini")
]

fig, axes = plotter.plot_grouped_stackedbars(
    df=eval_df,
    groupby_cols=("prompt_type",),
    value_col="Accuracy_Result",   # place a single legend above the subplots
    colormap="Pastel1",
    normalize=True,
    value_colors=partial_value_colors,
    group_order=partial_group_order,
    legend_categories_order=["Worse", "Same", "Better"],
    title="Requirement Accuracy Results Across Prompt Type",
)
plt.show()