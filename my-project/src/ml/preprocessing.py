"""
src/preprocessing.py

- Ngày 11: Thiết kế bộ khung 
- Ngày 12: Hiện thực hóa logic xử lý Missing Values & Feature Engineering.
"""

import pandas as pd
import numpy as np
from typing import List, Dict

# PHẦN 1: HẰNG SỐ 

NA_MEANS_NONE: List[str] = [
    "PoolQC", "Alley", "Fence", "FireplaceQu",
    "GarageType", "GarageFinish", "GarageQual", "GarageCond",
    "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
    "MiscFeature",
]

NUMERIC_ZEROS: List[str] = [
    "GarageYrBlt", "GarageCars", "GarageArea",
    "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF",
    "TotalBsmtSF", "BsmtFullBath", "BsmtHalfBath",
]

MEDIAN_FILL: List[str] = ["LotFrontage", "MasVnrArea"]

ORDINAL_MAP: Dict[str, int] = {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
ORDINAL_COLS: List[str] = ["ExterQual", "KitchenQual"]


# PHẦN 2: THỰC THI LOGIC 


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Xử lý các giá trị bị thiếu (Missing Values) trong tập dữ liệu.
    (Ngày 11: Viết Docstring này + Type Hint)
    
    Args:
        df (pd.DataFrame): DataFrame đầu vào chứa dữ liệu thô.
        
    Returns:
        pd.DataFrame: DataFrame mới đã được xử lý toàn bộ giá trị bị thiếu.
        
    Raises:
        ValueError: Nếu DataFrame đầu vào thiếu các cột bắt buộc.
    """
    # Ngày 12: Đảm bảo code chạy độc lập, dùng .copy()
    missing_required = [col for col in NA_MEANS_NONE if col not in df.columns]
    if missing_required:
        raise ValueError(f"Thiếu các cột bắt buộc: {missing_required}")

    df_clean = df.copy()

    # 1. NA mang ý nghĩa 'None'
    for col in NA_MEANS_NONE:
        df_clean[col] = df_clean[col].fillna("None")

    # 2. NA mang ý nghĩa 0
    for col in NUMERIC_ZEROS:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(0)

    # 3. Điền Median cho một số cột chỉ định
    for col in MEDIAN_FILL:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    # 4. Fallback: Xử lý các cột còn sót lại
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
    """
    Tạo các đặc trưng mới (Feature Engineering) dựa trên kiến thức miền.
    
    Args:
        df (pd.DataFrame): DataFrame đã được xử lý missing values.
        
    Returns:
        pd.DataFrame: DataFrame mới được bổ sung thêm các cột đặc trưng.
    """
    df_feat = df.copy()
    
    # Cần có try-except hoặc kiểm tra cột tồn tại để tránh lỗi khi test
    if all(col in df_feat.columns for col in ["TotalBsmtSF", "1stFlrSF", "2ndFlrSF"]):
        df_feat["TotalSF"] = df_feat["TotalBsmtSF"] + df_feat["1stFlrSF"] + df_feat["2ndFlrSF"]
        
    if all(col in df_feat.columns for col in ["YrSold", "YearBuilt"]):
        df_feat["HouseAge"] = df_feat["YrSold"] - df_feat["YearBuilt"]
        
    if all(col in df_feat.columns for col in ["YrSold", "YearRemodAdd"]):
        df_feat["RemodAge"] = df_feat["YrSold"] - df_feat["YearRemodAdd"]
        
    return df_feat


# PHẦN 3: KHUNG SƯỜN


def encode_categorical(df: pd.DataFrame, ordinal_cols: List[str] = ORDINAL_COLS) -> pd.DataFrame:
    """
    Mã hóa các biến phân loại (Categorical variables) sang dạng số.
    
    Args:
        df (pd.DataFrame): DataFrame đầu vào cần mã hóa.
        ordinal_cols (List[str]): Danh sách các cột có thứ bậc.
        
    Returns:
        pd.DataFrame: DataFrame mới đã được mã hóa (One-hot và Ordinal).
    """
    # TODO: Sẽ implement logic thật ngày tiếp theo.
    return df.copy()


def scale_numeric(df: pd.DataFrame, skew_threshold: float = 0.75) -> pd.DataFrame:
    """
    Xử lý độ lệch (skewness) và chuẩn hóa (scaling) các biến số học.
    
    Args:
        df (pd.DataFrame): DataFrame đầu vào.
        skew_threshold (float): Ngưỡng độ lệch để áp dụng Log Transform.
        
    Returns:
        pd.DataFrame: DataFrame mới đã được chuẩn hóa.
    """
    # TODO: Sẽ implement vào ngày tiếp theo.
    return df.copy()


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hàm tổng hợp, chạy tuần tự toàn bộ pipeline tiền xử lý.
    
    Args:
        df (pd.DataFrame): Dữ liệu thô ban đầu.
        
    Returns:
        pd.DataFrame: Dữ liệu hoàn chỉnh sẵn sàng đưa vào mô hình.
    """
    df_clean = handle_missing_values(df)
    df_feat = engineer_features(df_clean)
    df_encoded = encode_categorical(df_feat)
    return scale_numeric(df_encoded)