import pandas as pd

def pop_target(df: pd.DataFrame, target_col: str):
    """Extract target variable from dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    target_col : str
        Name of the target column.

    Returns
    -------
    pd.DataFrame
        Features dataframe (target column removed).
    pd.Series
        Target series.
    """
    df_copy = df.copy()
    target = df_copy.pop(target_col)  # raises KeyError if not found (expected)
    return df_copy, target
