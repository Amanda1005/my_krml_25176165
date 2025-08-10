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
    df = features_fixture.copy()
    df["salary"] = target_fixture
    X, y = pop_target(df=df, target_col="salary")
    pd.testing.assert_frame_equal(X, features_fixture)
    pd.testing.assert_series_equal(y, target_fixture)

def test_pop_target_no_col_found(features_fixture):
    with pytest.raises(KeyError):
        pop_target(df=features_fixture.copy(), target_col="salary")

def test_pop_target_col_none(features_fixture):
    with pytest.raises(KeyError):
        pop_target(df=features_fixture.copy(), target_col=None)

def test_pop_target_df_none():
    with pytest.raises(AttributeError):
        pop_target(df=None, target_col="salary")

