"""Preprocessing utilities for the Kaggle House Prices dataset."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

TARGET_COLUMN = "SalePrice"
SALE_PRICE_LOG_COLUMN = "SalePriceLog"

NUMERIC_FEATURES: list[str] = [
    "OverallQual",
    "GrLivArea",
    "GarageCars",
    "GarageArea",
    "TotalBsmtSF",
    "1stFlrSF",
    "FullBath",
    "TotRmsAbvGrd",
    "YearBuilt",
    "YearRemodAdd",
]
NOMINAL_FEATURES: list[str] = ["Neighborhood", "GarageType"]
ORDINAL_FEATURES: list[str] = ["ExterQual", "KitchenQual", "BsmtQual"]
FEATURE_COLUMNS: list[str] = NUMERIC_FEATURES + NOMINAL_FEATURES + ORDINAL_FEATURES

FEATURE_NOTES: list[dict[str, str]] = [
    {
        "field": "OverallQual",
        "type": "numeric",
        "signal": "Strongest numeric signal; corr with SalePrice is about 0.79.",
        "preprocessing": "Median impute fallback, then StandardScaler.",
    },
    {
        "field": "GrLivArea",
        "type": "numeric",
        "signal": "Above-ground living area has strong positive price correlation.",
        "preprocessing": "Median impute fallback, then StandardScaler.",
    },
    {
        "field": "GarageCars",
        "type": "numeric",
        "signal": "Garage capacity tracks higher-value homes.",
        "preprocessing": "Fill missing with 0, then StandardScaler.",
    },
    {
        "field": "GarageArea",
        "type": "numeric",
        "signal": "Garage size is strongly related to price and garage capacity.",
        "preprocessing": "Fill missing with 0, then StandardScaler.",
    },
    {
        "field": "TotalBsmtSF",
        "type": "numeric",
        "signal": "Basement area adds usable space and correlates with SalePrice.",
        "preprocessing": "Fill missing with 0, then StandardScaler.",
    },
    {
        "field": "1stFlrSF",
        "type": "numeric",
        "signal": "First-floor size is a strong area-based price driver.",
        "preprocessing": "Median impute fallback, then StandardScaler.",
    },
    {
        "field": "FullBath",
        "type": "numeric",
        "signal": "Bathroom count reflects utility and home size.",
        "preprocessing": "Median impute fallback, then StandardScaler.",
    },
    {
        "field": "TotRmsAbvGrd",
        "type": "numeric",
        "signal": "Room count captures scale of above-ground living space.",
        "preprocessing": "Median impute fallback, then StandardScaler.",
    },
    {
        "field": "YearBuilt",
        "type": "numeric",
        "signal": "Newer homes generally command higher sale prices.",
        "preprocessing": "Median impute fallback, then StandardScaler.",
    },
    {
        "field": "YearRemodAdd",
        "type": "numeric",
        "signal": "Recent remodeling improves perceived condition and price.",
        "preprocessing": "Median impute fallback, then StandardScaler.",
    },
    {
        "field": "Neighborhood",
        "type": "nominal",
        "signal": "Median SalePrice varies strongly by neighborhood.",
        "preprocessing": "Mode impute from train, then one-hot encode.",
    },
    {
        "field": "GarageType",
        "type": "nominal",
        "signal": "Built-in/attached garages sell differently from no garage.",
        "preprocessing": "Fill missing with None, then one-hot encode.",
    },
    {
        "field": "ExterQual",
        "type": "ordinal",
        "signal": "Exterior material quality separates high and low price groups.",
        "preprocessing": "Mode impute from train, then ordinal encode.",
    },
    {
        "field": "KitchenQual",
        "type": "ordinal",
        "signal": "Kitchen quality has clear monotonic price relationship.",
        "preprocessing": "Mode impute from train, then ordinal encode.",
    },
    {
        "field": "BsmtQual",
        "type": "ordinal",
        "signal": "Basement quality and missing basement status affect price.",
        "preprocessing": "Fill missing with None, then ordinal encode.",
    },
]

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
ORDINAL_COLS: list[str] = ["ExterQual", "KitchenQual", "BsmtQual"]
ORDINAL_QUALITY_ORDER: list[str] = ["Po", "Fa", "TA", "Gd", "Ex"]
BASEMENT_QUALITY_ORDER: list[str] = ["None", "Po", "Fa", "TA", "Gd", "Ex"]


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
                if col in NUMERIC_ZEROS:
                    self.numeric_medians_[col] = 0.0
                else:
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
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
    ordinal_features: list[str] | None = None,
) -> ColumnTransformer:
    """Build a sklearn ColumnTransformer for numeric and categorical features."""
    numeric_features = (
        NUMERIC_FEATURES if numeric_features is None else numeric_features
    )
    categorical_features = (
        NOMINAL_FEATURES if categorical_features is None else categorical_features
    )
    ordinal_features = (
        ORDINAL_FEATURES if ordinal_features is None else ordinal_features
    )

    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    )
    ordinal_categories = [
        BASEMENT_QUALITY_ORDER.copy()
        if col == "BsmtQual"
        else ORDINAL_QUALITY_ORDER.copy()
        for col in ordinal_features
    ]
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


def _feature_frame(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Return selected model features and validate that all are available."""
    missing_cols = [col for col in feature_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Configured pipeline columns are missing: {missing_cols}")
    return df[feature_columns].copy()


def _build_fit_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    ordinal_features: list[str],
) -> Pipeline:
    """Build the persisted train-only missing-value and preprocessing pipeline."""
    return Pipeline(
        steps=[
            ("missing_values", HousePricesMissingValueImputer()),
            (
                "preprocessor",
                build_preprocessing_pipeline(
                    numeric_features=numeric_features,
                    categorical_features=categorical_features,
                    ordinal_features=ordinal_features,
                ),
            ),
        ]
    )


def _processed_dataframe(
    transformed: NDArray[np.float64],
    preprocessor: Pipeline,
    source_df: pd.DataFrame,
) -> pd.DataFrame:
    """Create an inspectable processed DataFrame from a fitted pipeline."""
    column_transformer = preprocessor.named_steps["preprocessor"]
    if not isinstance(column_transformer, ColumnTransformer):
        raise TypeError("Expected fitted pipeline to contain a ColumnTransformer.")

    columns = [str(col) for col in column_transformer.get_feature_names_out()]
    processed = pd.DataFrame(transformed, columns=columns, index=source_df.index)

    if TARGET_COLUMN in source_df.columns:
        target = pd.to_numeric(source_df[TARGET_COLUMN], errors="raise")
        processed[TARGET_COLUMN] = target.to_numpy()
        processed[SALE_PRICE_LOG_COLUMN] = np.log1p(target.to_numpy())

    return processed


def _write_processed_csv(
    df: pd.DataFrame, processed_csv_path: str | Path | None
) -> None:
    """Persist a processed DataFrame when an output path is provided."""
    if processed_csv_path is None:
        return

    output_path = Path(processed_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def save_feature_notes(path: str | Path) -> Path:
    """Save selected-feature rationale and preprocessing decisions as CSV."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(FEATURE_NOTES).to_csv(output_path, index=False)
    return output_path


def fit_transform_house_prices(
    df: pd.DataFrame,
    preprocessor_path: str | Path,
    processed_csv_path: str | Path | None = None,
) -> pd.DataFrame:
    """Fit default 15-field preprocessing and return an inspectable DataFrame."""
    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    x_raw = _feature_frame(df, FEATURE_COLUMNS)
    preprocessor = _build_fit_pipeline(
        numeric_features=NUMERIC_FEATURES,
        categorical_features=NOMINAL_FEATURES,
        ordinal_features=ORDINAL_FEATURES,
    )
    transformed = np.asarray(preprocessor.fit_transform(x_raw), dtype=np.float64)

    preprocessor_output_path = Path(preprocessor_path)
    preprocessor_output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, preprocessor_output_path)

    processed = _processed_dataframe(
        transformed=transformed,
        preprocessor=preprocessor,
        source_df=df,
    )
    _write_processed_csv(processed, processed_csv_path)
    return processed


def transform_house_prices(
    df: pd.DataFrame,
    preprocessor_path: str | Path,
    processed_csv_path: str | Path | None = None,
) -> pd.DataFrame:
    """Transform data with the persisted default 15-field preprocessor."""
    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    x_raw = _feature_frame(df, FEATURE_COLUMNS)
    try:
        preprocessor = joblib.load(preprocessor_path)
    except FileNotFoundError as exc:
        raise ValueError(f"Preprocessor file not found: {preprocessor_path}") from exc

    transformed = np.asarray(preprocessor.transform(x_raw), dtype=np.float64)
    processed = _processed_dataframe(
        transformed=transformed,
        preprocessor=preprocessor,
        source_df=df,
    )
    _write_processed_csv(processed, processed_csv_path)
    return processed


def preprocess_train(
    df: pd.DataFrame,
    num_cols: list[str] | None = None,
    cat_cols: list[str] | None = None,
    ord_cols: list[str] | None = None,
    save_path: str = "preprocessor.pkl",
) -> NDArray[np.float64]:
    """Fit preprocessing on train data, persist it, and return transformed data."""
    num_cols = NUMERIC_FEATURES if num_cols is None else num_cols
    cat_cols = NOMINAL_FEATURES if cat_cols is None else cat_cols
    ord_cols = ORDINAL_FEATURES if ord_cols is None else ord_cols

    df_feat = engineer_features(df)

    all_required_cols = num_cols + cat_cols + ord_cols
    x_raw = _feature_frame(df_feat, all_required_cols)

    preprocessor = _build_fit_pipeline(num_cols, cat_cols, ord_cols)
    x_processed = preprocessor.fit_transform(x_raw)

    joblib.dump(preprocessor, save_path)

    return np.asarray(x_processed, dtype=np.float64)


def preprocess_test(
    df: pd.DataFrame, load_path: str = "preprocessor.pkl"
) -> NDArray[np.float64]:
    """Load a fitted preprocessor and transform test or inference data."""
    df_feat = engineer_features(df)

    try:
        preprocessor = joblib.load(load_path)
    except FileNotFoundError as exc:
        raise ValueError(f"Preprocessor file not found: {load_path}") from exc

    x_processed = preprocessor.transform(df_feat)

    return np.asarray(x_processed, dtype=np.float64)
