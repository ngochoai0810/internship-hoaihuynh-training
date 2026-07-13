"""
ex2_implement_and_test.py
Đáp án Bài tập 2: Viết logic cho 2 hàm cơ bản và thiết lập khối chạy thử nghiệm.
"""

import pandas as pd
import numpy as np

NA_MEANS_NONE: list[str] = ["PoolQC", "Alley", "Fence", "FireplaceQu", "GarageType", "GarageFinish", "GarageQual", "GarageCond", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "MiscFeature"]
NUMERIC_ZEROS: list[str] = ["GarageYrBlt", "GarageCars", "GarageArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "BsmtFullBath", "BsmtHalfBath"]
MEDIAN_FILL: list[str] = ["LotFrontage", "MasVnrArea"]

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    
    df_clean = df.copy()
    for col in NA_MEANS_NONE:
        if col in df_clean.columns: df_clean[col] = df_clean[col].fillna("None")
    for col in NUMERIC_ZEROS:
        if col in df_clean.columns: df_clean[col] = df_clean[col].fillna(0)
    for col in MEDIAN_FILL:
        if col in df_clean.columns: df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    # Fill remaining missing values

    for col in df_clean.columns:
        if df_clean[col].isna().any():
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            else:
                mode_values = df_clean[col].mode(dropna=True)
                if not mode_values.empty: df_clean[col] = df_clean[col].fillna(mode_values.iloc[0])
    return df_clean

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    # Create new features based on existing ones
    df_feat = df.copy()
    df_feat["TotalSF"] = df_feat["TotalBsmtSF"] + df_feat["1stFlrSF"] + df_feat["2ndFlrSF"]
    df_feat["HouseAge"] = df_feat["YrSold"] - df_feat["YearBuilt"]
    df_feat["RemodAge"] = df_feat["YrSold"] - df_feat["YearRemodAdd"]
    return df_feat

def encode_categorical(df: pd.DataFrame) -> pd.DataFrame: ...
def scale_numeric(df: pd.DataFrame) -> pd.DataFrame: ...
def preprocess(df: pd.DataFrame) -> pd.DataFrame: ...

# Main execution block
if __name__ == '__main__':
    try:
        df_raw = pd.read_csv('data/raw/train.csv')
        df_clean = handle_missing_values(df_raw)
        print(f'Missing after cleaning: {df_clean.isnull().sum().sum()}')  # Expected: 0
        df_feat = engineer_features(df_clean)
        print(df_feat[['TotalSF', 'HouseAge', 'RemodAge']].head())
    except FileNotFoundError:
        print("Please ensure the 'data/raw/train.csv' file exists in the specified path.")