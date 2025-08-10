import pandas as pd

def pop_target(df, target_col):
    """Extract target variable from dataframe and return (features, target)."""
    df_copy = df.copy()
    target = df_copy.pop(target_col)
    return df_copy, target
