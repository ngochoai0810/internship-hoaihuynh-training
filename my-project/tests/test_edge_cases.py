from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest
from ml.preprocessing import (
    TARGET_COLUMN,
    engineer_features,
    fit_transform_house_prices,
    handle_missing_values,
    preprocess_test,
    preprocess_train,
    transform_house_prices,
)


@pytest.fixture
def base_dummy_data() -> pd.DataFrame:
    """Create valid data for edge-case preprocessing tests."""
    required_cols = [
        "PoolQC",
        "Alley",
        "Fence",
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

    df = pd.DataFrame({col: ["None", "None"] for col in required_cols})

    df["GarageArea"] = [500, 300]
    df["LotFrontage"] = [50.0, 70.0]
    df["GarageType"] = ["Attchd", "Detchd"]
    df["ExterQual"] = ["TA", "Gd"]

    df["TotalBsmtSF"] = [500, 1000]
    df["1stFlrSF"] = [500, 1000]
    df["2ndFlrSF"] = [0, 500]
    df["YrSold"] = [2010, 2020]
    df["YearBuilt"] = [2000, 2015]
    df["YearRemodAdd"] = [2005, 2018]

    return df


def test_unseen_categories_in_test_set(
    base_dummy_data: pd.DataFrame, tmp_path: Path
) -> None:
    save_path = str(tmp_path / "preprocessor.pkl")

    num_cols = ["GarageArea", "LotFrontage"]
    cat_cols = ["GarageType"]
    ord_cols = ["ExterQual"]

    x_train = preprocess_train(
        df=base_dummy_data,
        num_cols=num_cols,
        cat_cols=cat_cols,
        ord_cols=ord_cols,
        save_path=save_path,
    )

    df_test = base_dummy_data.copy()
    df_test["GarageType"] = ["Basment", "BuiltIn"]

    x_test = preprocess_test(df=df_test, load_path=save_path)

    assert x_train.shape[1] == x_test.shape[1]


def test_joblib_serialization_consistency(
    base_dummy_data: pd.DataFrame, tmp_path: Path
) -> None:
    save_path = str(tmp_path / "preprocessor_consistency.pkl")

    num_cols = ["GarageArea", "LotFrontage"]
    cat_cols = ["GarageType"]
    ord_cols = ["ExterQual"]

    x_train_memory = preprocess_train(
        df=base_dummy_data,
        num_cols=num_cols,
        cat_cols=cat_cols,
        ord_cols=ord_cols,
        save_path=save_path,
    )

    loaded_preprocessor: Any = joblib.load(save_path)

    df_clean = handle_missing_values(base_dummy_data)
    df_feat = engineer_features(df_clean)
    x_train_loaded = loaded_preprocessor.transform(df_feat)

    np.testing.assert_array_equal(
        x_train_memory,
        x_train_loaded,
        err_msg="Loaded preprocessor output does not match in-memory output.",
    )


def test_dataframe_transform_keeps_stable_columns_with_unseen_categories(
    tmp_path: Path,
) -> None:
    train_df = pd.DataFrame(
        {
            "OverallQual": [7, 5, 8],
            "GrLivArea": [1710, 1262, 1786],
            "GarageCars": [2.0, 1.0, 3.0],
            "GarageArea": [548.0, 460.0, 836.0],
            "TotalBsmtSF": [856.0, 1262.0, 920.0],
            "1stFlrSF": [856, 1262, 920],
            "FullBath": [2, 2, 2],
            "TotRmsAbvGrd": [8, 6, 6],
            "YearBuilt": [2003, 1976, 2001],
            "YearRemodAdd": [2003, 1976, 2002],
            "Neighborhood": ["CollgCr", "Veenker", "Crawfor"],
            "GarageType": ["Attchd", "Detchd", "BuiltIn"],
            "ExterQual": ["Gd", "TA", "Ex"],
            "KitchenQual": ["Gd", "TA", "Ex"],
            "BsmtQual": ["Gd", "TA", "Ex"],
            "SalePrice": [208500, 181500, 223500],
        }
    )
    test_df = train_df.drop(columns=[TARGET_COLUMN]).copy()
    test_df["Neighborhood"] = ["NoRidge", "StoneBr", "NridgHt"]
    test_df["GarageType"] = ["CarPort", "Basment", "2Types"]

    preprocessor_path = tmp_path / "house_prices_preprocessor.pkl"
    processed_train = fit_transform_house_prices(
        df=train_df,
        preprocessor_path=preprocessor_path,
    )
    processed_test = transform_house_prices(
        df=test_df,
        preprocessor_path=preprocessor_path,
    )

    assert TARGET_COLUMN in processed_train.columns
    assert TARGET_COLUMN not in processed_test.columns
    assert "SalePriceLog" not in processed_test.columns
    assert processed_train.drop(columns=[TARGET_COLUMN, "SalePriceLog"]).shape[1] == (
        processed_test.shape[1]
    )
    assert processed_test.isna().sum().sum() == 0
