"""Preprocessing utilities for the Kaggle House Prices dataset."""

import joblib
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

NA_MEANS_NONE: list[str] = [
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

NUMERIC_ZEROS: list[str] = [
    "GarageYrBlt",
    "GarageCars",
    "GarageArea",
    "BsmtFinSF1",
    "BsmtFinSF2",
    "BsmtUnfSF",
    "TotalBsmtSF",
    "BsmtFullBath",
    "BsmtHalfBath",
]

MEDIAN_FILL: list[str] = ["LotFrontage", "MasVnrArea"]

ORDINAL_MAP: dict[str, int] = {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
ORDINAL_COLS: list[str] = ["ExterQual", "KitchenQual"]
ORDINAL_QUALITY_ORDER: list[str] = ["Po", "Fa", "TA", "Gd", "Ex"]


class HousePricesMissingValueImputer(BaseEstimator, TransformerMixin):
    """Fit House Prices missing-value rules on train data only."""

    def __init__(self, none_columns: list[str] | None = None) -> None:
        self.none_columns = none_columns

    def fit(
        self, X: pd.DataFrame, y: object | None = None
    ) -> "HousePricesMissingValueImputer":
        """Learn train-only fill values for columns present in the input."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("HousePricesMissingValueImputer expects a DataFrame.")

        none_columns = (
            self.none_columns if self.none_columns is not None else NA_MEANS_NONE
        )
        self.columns_: list[str] = list(X.columns)
        self.none_columns_: list[str] = [
            col for col in none_columns if col in X.columns
        ]
        self.numeric_medians_: dict[str, float] = {}
        self.categorical_modes_: dict[str, str] = {}

        for col in X.columns:
            if col in self.none_columns_:
                continue

            if pd.api.types.is_numeric_dtype(X[col]):
                median_value = X[col].median()
                self.numeric_medians_[col] = (
                    0.0 if pd.isna(median_value) else float(median_value)
                )
                continue

            mode_values = X[col].mode(dropna=True)
            fallback_value = "TA" if col in ORDINAL_COLS else "None"
            self.categorical_modes_[col] = (
                fallback_value if mode_values.empty else str(mode_values.iloc[0])
            )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values using values learned during fit."""
        check_is_fitted(self, "columns_")

        if not isinstance(X, pd.DataFrame):
            raise TypeError("HousePricesMissingValueImputer expects a DataFrame.")

        missing_cols = [col for col in self.columns_ if col not in X.columns]
        if missing_cols:
            raise ValueError(f"Missing columns during transform: {missing_cols}")

        df_clean = X.copy()

        for col in self.none_columns_:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna("None")

        for col, numeric_fill_value in self.numeric_medians_.items():
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna(numeric_fill_value)

        for col, categorical_fill_value in self.categorical_modes_.items():
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna(categorical_fill_value)

        return df_clean


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values according to the House Prices EDA strategy."""
    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    missing_required = [col for col in NA_MEANS_NONE if col not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    df_clean = df.copy()

    for col in NA_MEANS_NONE:
        df_clean[col] = df_clean[col].fillna("None")

    for col in NUMERIC_ZEROS:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(0)

    for col in MEDIAN_FILL:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    for col in df_clean.columns:
        if df_clean[col].isna().any():
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            else:
                mode_values = df_clean[col].mode(dropna=True)
                if not mode_values.empty:
                    df_clean[col] = df_clean[col].fillna(mode_values.iloc[0])

    return df_clean


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create reusable features identified during EDA."""
    df_feat = df.copy()

    if all(col in df_feat.columns for col in ["TotalBsmtSF", "1stFlrSF", "2ndFlrSF"]):
        df_feat["TotalSF"] = (
            df_feat["TotalBsmtSF"] + df_feat["1stFlrSF"] + df_feat["2ndFlrSF"]
        )

    if all(col in df_feat.columns for col in ["YrSold", "YearBuilt"]):
        df_feat["HouseAge"] = df_feat["YrSold"] - df_feat["YearBuilt"]

    if all(col in df_feat.columns for col in ["YrSold", "YearRemodAdd"]):
        df_feat["RemodAge"] = df_feat["YrSold"] - df_feat["YearRemodAdd"]

    return df_feat


def build_preprocessing_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    ordinal_features: list[str],
) -> ColumnTransformer:
    """Build a sklearn ColumnTransformer for numeric and categorical features."""
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    )
    ordinal_categories = [ORDINAL_QUALITY_ORDER.copy() for _ in ordinal_features]
    ordinal_transformer = Pipeline(
        steps=[
            (
                "ordinal",
                OrdinalEncoder(
                    categories=ordinal_categories,
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            )
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
            ("ord", ordinal_transformer, ordinal_features),
        ],
        remainder="drop",
    )


def preprocess_train(
    df: pd.DataFrame,
    num_cols: list[str],
    cat_cols: list[str],
    ord_cols: list[str],
    save_path: str = "preprocessor.pkl",
) -> NDArray[np.float64]:
    """Fit preprocessing on train data, persist it, and return transformed data."""
    df_clean = handle_missing_values(df)
    df_feat = engineer_features(df_clean)

    all_required_cols = num_cols + cat_cols + ord_cols
    missing_cols = [col for col in all_required_cols if col not in df_feat.columns]
    if missing_cols:
        raise ValueError(f"Configured pipeline columns are missing: {missing_cols}")

    preprocessor = build_preprocessing_pipeline(num_cols, cat_cols, ord_cols)
    x_processed = preprocessor.fit_transform(df_feat)

    joblib.dump(preprocessor, save_path)

    return np.asarray(x_processed, dtype=np.float64)


def preprocess_test(
    df: pd.DataFrame, load_path: str = "preprocessor.pkl"
) -> NDArray[np.float64]:
    """Load a fitted preprocessor and transform test or inference data."""
    df_clean = handle_missing_values(df)
    df_feat = engineer_features(df_clean)

    try:
        preprocessor = joblib.load(load_path)
    except FileNotFoundError as exc:
        raise ValueError(f"Preprocessor file not found: {load_path}") from exc

    x_processed = preprocessor.transform(df_feat)

    return np.asarray(x_processed, dtype=np.float64)
