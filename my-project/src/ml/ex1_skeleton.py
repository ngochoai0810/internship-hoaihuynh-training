"""
ex1_skeleton.py
Đáp án Bài tập 1: Chỉ tạo bộ khung (signatures & docstrings), không viết logic.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Constants
NA_MEANS_NONE: list[str] = [
    "PoolQC", "Alley", "Fence", "FireplaceQu",
    "GarageType", "GarageFinish", "GarageQual", "GarageCond",
    "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
    "MiscFeature",
]
NUMERIC_ZEROS: list[str] = [
    "GarageYrBlt", "GarageCars", "GarageArea",
    "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF",
    "TotalBsmtSF", "BsmtFullBath", "BsmtHalfBath",
]
MEDIAN_FILL: list[str] = ["LotFrontage", "MasVnrArea"]
ORDINAL_MAP: dict[str, int] = {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
ORDINAL_COLS: list[str] = ["ExterQual", "KitchenQual"]
SKEW_THRESHOLD: float = 0.75

# Functions
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    
    return pd.DataFrame()
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame()

def encode_categorical(df: pd.DataFrame, ordinal_cols: list[str] = ORDINAL_COLS) -> pd.DataFrame:
    return pd.DataFrame()

def scale_numeric(df: pd.DataFrame, skew_threshold: float = SKEW_THRESHOLD) -> pd.DataFrame:
    return pd.DataFrame()

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame()