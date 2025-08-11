import pandas as pd

def pop_target(df, target_col):
    """Extract target variable from dataframe. """

    df_copy = df.copy()
    target = df_copy.pop(target_col)

    return df_copy, target
import pytest
import pandas as pd

from my_krml_25176165.data.sets import pop_target


@pytest.fixture
def features_fixture():
    features_data = [
        [1, 25, "Junior"],
        [2, 33, "Confirmed"],
        [3, 42, "Manager"],
    ]
    return pd.DataFrame(features_data, columns=["employee_id", "age", "level"])

@pytest.fixture
def target_fixture():
    target_data = [5, 10, 20]
    return pd.Series(target_data, name="salary", copy=False)

def test_pop_target_with_data_fixture(features_fixture, target_fixture):
    input_df = features_fixture.copy()
    input_df["salary"] = target_fixture

    features, target = pop_target(df=input_df, target_col='salary')

    pd.testing.assert_frame_equal(features, features_fixture)
    pd.testing.assert_series_equal(target, target_fixture)

def test_pop_target_no_col_found(features_fixture, target_fixture):
    input_df = features_fixture.copy()

    with pytest.raises(KeyError):
        features, target = pop_target(df=input_df, target_col='salary')

def test_pop_target_col_none(features_fixture, target_fixture):
    input_df = features_fixture.copy()

    with pytest.raises(KeyError):
        features, target = pop_target(df=input_df, target_col=None)

def test_pop_target_df_none(features_fixture, target_fixture):
    input_df = features_fixture.copy()

    with pytest.raises(AttributeError):
        features, target = pop_target(df=None, target_col="salary")

def save_sets(X_train=None, y_train=None, X_val=None, y_val=None, X_test=None, y_test=None, path='../data/processed/'):
    """Save the different sets locally. """
    import numpy as np

    if X_train is not None:
      np.save(f'{path}X_train', X_train)
    if X_val is not None:
      np.save(f'{path}X_val',   X_val)
    if X_test is not None:
      np.save(f'{path}X_test',  X_test)
    if y_train is not None:
      np.save(f'{path}y_train', y_train)
    if y_val is not None:
      np.save(f'{path}y_val',   y_val)
    if y_test is not None:
      np.save(f'{path}y_test',  y_test)

def load_sets(path='../data/processed/'):
    """Load the different locally save sets. """
    import numpy as np
    import os.path

    X_train = np.load(f'{path}X_train.npy', allow_pickle=True) if os.path.isfile(f'{path}X_train.npy') else None
    X_val   = np.load(f'{path}X_val.npy'  , allow_pickle=True) if os.path.isfile(f'{path}X_val.npy')   else None
    X_test  = np.load(f'{path}X_test.npy' , allow_pickle=True) if os.path.isfile(f'{path}X_test.npy')  else None
    y_train = np.load(f'{path}y_train.npy', allow_pickle=True) if os.path.isfile(f'{path}y_train.npy') else None
    y_val   = np.load(f'{path}y_val.npy'  , allow_pickle=True) if os.path.isfile(f'{path}y_val.npy')   else None
    y_test  = np.load(f'{path}y_test.npy' , allow_pickle=True) if os.path.isfile(f'{path}y_test.npy')  else None

    return X_train, y_train, X_val, y_val, X_test, y_test

def subset_x_y(target, features, start_index:int, end_index:int):
    """Keep only the rows for X and y (optional) sets from the specified indexes. """

    return features[start_index:end_index], target[start_index:end_index]

def split_sets_by_time(df, target_col, test_ratio=0.2):
    """Split sets by indexes for an ordered dataframe. """

    df_copy = df.copy()
    target = df_copy.pop(target_col)
    cutoff = int(len(df_copy) / 5)

    X_train, y_train = subset_x_y(target=target, features=df_copy, start_index=0, end_index=-cutoff*2)
    X_val, y_val     = subset_x_y(target=target, features=df_copy, start_index=-cutoff*2, end_index=-cutoff)
    X_test, y_test   = subset_x_y(target=target, features=df_copy, start_index=-cutoff, end_index=len(df_copy))

    return X_train, y_train, X_val, y_val, X_test, y_test

def split_sets_random(features, target, test_ratio=0.2):
    """Split sets randomly. """
    from sklearn.model_selection import train_test_split

    val_ratio = test_ratio / (1 - test_ratio)
    X_data, X_test, y_data, y_test = train_test_split(features, target, test_size=test_ratio, random_state=8)
    X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=val_ratio, random_state=8)

    return X_train, y_train, X_val, y_val, X_test, y_test