import pytest
import pandas as pd
from my_krml_25176165.data.sets import pop_target

@pytest.fixture
def features_fixture():
    data = [
        [1, 25, "Junior"],
        [2, 33, "Confirmed"],
        [3, 42, "Manager"],
    ]
    return pd.DataFrame(data, columns=["employee_id", "age", "level"])

@pytest.fixture
def target_fixture():
    return pd.Series([5, 10, 20], name="salary", copy=False)

def test_pop_target_with_data_fixture(features_fixture, target_fixture):
    input_df = features_fixture.copy()
    input_df["salary"] = target_fixture
    features, target = pop_target(df=input_df, target_col="salary")
    pd.testing.assert_frame_equal(features, features_fixture)
    pd.testing.assert_series_equal(target, target_fixture)

def test_pop_target_no_col_found(features_fixture):
    input_df = features_fixture.copy()
    with pytest.raises(KeyError):
        pop_target(df=input_df, target_col="salary")

def test_pop_target_col_none(features_fixture):
    input_df = features_fixture.copy()
    with pytest.raises(KeyError):
        pop_target(df=input_df, target_col=None)

def test_pop_target_df_none():
    with pytest.raises(AttributeError):
        pop_target(df=None, target_col="salary")

# --------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------
import numpy as np
from my_krml_25176165.data.sets import split_sets_random

def test_split_sets_random_shapes_and_ratio():
    n = 100
    X = pd.DataFrame({"f1": np.random.randn(n), "f2": np.random.randn(n)})
    y = pd.Series(np.random.randint(0, 3, size=n))

    X_train, y_train, X_val, y_val, X_test, y_test = split_sets_random(
        X, y, test_ratio=0.2, random_state=42, stratify=False
    )

    # sizes add up and val ~= test (allow ±1 due to rounding)
    assert len(X_train) + len(X_val) + len(X_test) == n
    assert abs(len(X_val) - len(X_test)) <= 1
    assert len(y_train) == len(X_train) and len(y_val) == len(X_val) and len(y_test) == len(X_test)

def test_split_sets_random_stratify_keeps_label_ratio():
    # Make a class-imbalanced target to verify stratify behavior
    n = 200
    X = pd.DataFrame({"f1": np.random.randn(n), "f2": np.random.randn(n)})
    y = pd.Series(np.r_[np.zeros(160, dtype=int), np.ones(40, dtype=int)])  # 80%:20%

    X_train, y_train, X_val, y_val, X_test, y_test = split_sets_random(
        X, y, test_ratio=0.2, random_state=0, stratify=True
    )

    # Class ratio should be close across splits (tolerance 10%)
    def frac_ones(s): 
        return (s == 1).mean()

    overall = frac_ones(y)
    for split in [y_train, y_val, y_test]:
        assert abs(frac_ones(split) - overall) < 0.10
