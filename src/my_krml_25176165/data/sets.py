# src/my_krml_25176165/data/sets.py
from __future__ import annotations
from typing import Tuple, Optional
import os.path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def pop_target(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    """Extract target variable from dataframe."""
    df_copy = df.copy()
    target = df_copy.pop(target_col)  # KeyError if not found → expected
    return df_copy, target


def save_sets(X_train=None, y_train=None, X_val=None, y_val=None, X_test=None, y_test=None,
              path: str = "../data/processed/") -> None:
    """Save the different sets locally as .npy."""
    if X_train is not None:
        np.save(f"{path}X_train", X_train)
    if X_val is not None:
        np.save(f"{path}X_val", X_val)
    if X_test is not None:
        np.save(f"{path}X_test", X_test)
    if y_train is not None:
        np.save(f"{path}y_train", y_train)
    if y_val is not None:
        np.save(f"{path}y_val", y_val)
    if y_test is not None:
        np.save(f"{path}y_test", y_test)


def load_sets(path: str = "../data/processed/"):
    """Load the different locally saved sets if present, else None."""
    X_train = np.load(f"{path}X_train.npy", allow_pickle=True) if os.path.isfile(f"{path}X_train.npy") else None
    X_val   = np.load(f"{path}X_val.npy",   allow_pickle=True) if os.path.isfile(f"{path}X_val.npy")   else None
    X_test  = np.load(f"{path}X_test.npy",  allow_pickle=True) if os.path.isfile(f"{path}X_test.npy")  else None
    y_train = np.load(f"{path}y_train.npy", allow_pickle=True) if os.path.isfile(f"{path}y_train.npy") else None
    y_val   = np.load(f"{path}y_val.npy",   allow_pickle=True) if os.path.isfile(f"{path}y_val.npy")   else None
    y_test  = np.load(f"{path}y_test.npy",  allow_pickle=True) if os.path.isfile(f"{path}y_test.npy")  else None
    return X_train, y_train, X_val, y_val, X_test, y_test


def subset_x_y(target: pd.Series, features: pd.DataFrame, start_index: int, end_index: int):
    """Keep only the rows for X and y from the specified indexes (slice semantics)."""
    return features[start_index:end_index], target[start_index:end_index]


def split_sets_by_time(df: pd.DataFrame, target_col: str, test_ratio: float = 0.2):
    """Split sets by index order for an ordered dataframe: last 20% test, previous 20% val."""
    df_copy = df.copy()
    target = df_copy.pop(target_col)
    cutoff = int(len(df_copy) / 5)

    X_train, y_train = subset_x_y(target=target, features=df_copy, start_index=0,            end_index=-cutoff*2)
    X_val,   y_val   = subset_x_y(target=target, features=df_copy, start_index=-cutoff*2,    end_index=-cutoff)
    X_test,  y_test  = subset_x_y(target=target, features=df_copy, start_index=-cutoff,      end_index=len(df_copy))
    return X_train, y_train, X_val, y_val, X_test, y_test


def split_sets_random(features: pd.DataFrame,
                      target: pd.Series,
                      test_ratio: float = 0.2,
                      random_state: int = 42,
                      stratify: bool = False
                      ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Randomly split into train / val / test with equal val and test ratios.
    Example: test_ratio=0.2 → 60% train, 20% val, 20% test.
    """
    if not 0 < test_ratio < 0.5:
        raise ValueError("test_ratio must be in (0, 0.5) so val and test can be equal.")

    strat = target if stratify else None

    # First split off test
    X_rem, X_test, y_rem, y_test = train_test_split(
        features, target,
        test_size=test_ratio,
        random_state=random_state,
        stratify=strat
    )

    # Then split val from the remaining data so final val ratio == test_ratio
    val_size_within_remaining = test_ratio / (1 - test_ratio)
    strat_rem = y_rem if stratify else None

    X_train, X_val, y_train, y_val = train_test_split(
        X_rem, y_rem,
        test_size=val_size_within_remaining,
        random_state=random_state,
        stratify=strat_rem
    )
    return X_train, y_train, X_val, y_val, X_test, y_test
