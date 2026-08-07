from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from ml.preprocessing import engineer_features, handle_missing_values, preprocess_train


@pytest.fixture
def mock_missing_data() -> pd.DataFrame:
    """Create a small DataFrame for missing-value tests."""
    return pd.DataFrame(
        {
            "PoolQC": [np.nan, "Ex"],
            "GarageArea": [np.nan, 500],
            "LotFrontage": [50.0, np.nan],
            "Fence": ["MnPrv", np.nan],
            **{
                col: [np.nan, np.nan]
                for col in [
                    "Alley",
                    "FireplaceQu",
                    "GarageType",
                    "GarageFinish",
                    "GarageQual",
                    "GarageCond",
                    "BsmtQual",
                    "BsmtCond",
                    "BsmtExposure",
                    "BsmtFinType1",
                    "BsmtFinType2",
                    "MiscFeature",
                ]
            },
        }
    )


@pytest.fixture
def mock_feature_data() -> pd.DataFrame:
    """Create a small DataFrame for feature engineering tests."""
    return pd.DataFrame(
        {
            "TotalBsmtSF": [500, 1000],
            "1stFlrSF": [500, 1000],
            "2ndFlrSF": [0, 500],
            "YrSold": [2010, 2020],
            "YearBuilt": [2000, 2015],
            "YearRemodAdd": [2005, 2018],
        }
    )


def test_missing_required_columns() -> None:
    bad_df = pd.DataFrame({"SomeRandomColumn": [1, 2, 3]})
    with pytest.raises(ValueError, match="Missing required columns"):
        handle_missing_values(bad_df)


def test_empty_dataframe() -> None:
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError, match="empty"):
        handle_missing_values(empty_df)


def test_handle_missing_values_no_mutation(mock_missing_data: pd.DataFrame) -> None:
    original_df = mock_missing_data.copy()
    _ = handle_missing_values(mock_missing_data)
    pd.testing.assert_frame_equal(mock_missing_data, original_df)


def test_handle_missing_values_logic(mock_missing_data: pd.DataFrame) -> None:
    df_clean = handle_missing_values(mock_missing_data)

    assert df_clean["PoolQC"].iloc[0] == "None"
    assert df_clean["GarageArea"].iloc[0] == 0
    assert df_clean["LotFrontage"].iloc[1] == 50.0
    assert df_clean.isnull().sum().sum() == 0


def test_engineer_features_logic(mock_feature_data: pd.DataFrame) -> None:
    df_feat = engineer_features(mock_feature_data)

    assert "TotalSF" in df_feat.columns
    assert "HouseAge" in df_feat.columns
    assert "RemodAge" in df_feat.columns

    assert df_feat["TotalSF"].iloc[0] == 1000
    assert df_feat["HouseAge"].iloc[0] == 10
    assert df_feat["RemodAge"].iloc[0] == 5


def test_preprocess_train_pipeline(
    mock_missing_data: pd.DataFrame, tmp_path: Path
) -> None:
    df = handle_missing_values(mock_missing_data)
    df["GarageType"] = ["Attchd", "Detchd"]
    df["ExterQual"] = ["TA", "Gd"]

    num_cols = ["GarageArea", "LotFrontage"]
    cat_cols = ["GarageType"]
    ord_cols = ["ExterQual"]

    save_filepath = tmp_path / "test_preprocessor.pkl"

    x_processed = preprocess_train(
        df=df,
        num_cols=num_cols,
        cat_cols=cat_cols,
        ord_cols=ord_cols,
        save_path=str(save_filepath),
    )

    assert save_filepath.exists()
    assert isinstance(x_processed, np.ndarray)

    mean_val = np.mean(x_processed[:, 0])
    std_val = np.std(x_processed[:, 0])

    assert np.isclose(mean_val, 0, atol=1e-7)
    assert np.isclose(std_val, 1, atol=1e-7)
