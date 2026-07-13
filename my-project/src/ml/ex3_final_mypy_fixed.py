"""
ex3_final_mypy_fixed.py
Đáp án Bài tập 3: Hoàn thiện toàn bộ pipeline, dọn dẹp lỗi Mypy (Incompatible types).
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


NA_MEANS_NONE: list[str] = ["PoolQC", "Alley", "Fence", "FireplaceQu", "GarageType", "GarageFinish", "GarageQual", "GarageCond", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "MiscFeature"]
NUMERIC_ZEROS: list[str] = ["GarageYrBlt", "GarageCars", "GarageArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "BsmtFullBath", "BsmtHalfBath"]
MEDIAN_FILL: list[str] = ["LotFrontage", "MasVnrArea"]
ORDINAL_MAP: dict[str, int] = {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
ORDINAL_COLS: list[str] = ["ExterQual", "KitchenQual"]
SKEW_THRESHOLD: float = 0.75

# 1.Func clean data
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy()
    for col in NA_MEANS_NONE: df_clean[col] = df_clean[col].fillna("None")
    for col in NUMERIC_ZEROS: df_clean[col] = df_clean[col].fillna(0)
    for col in MEDIAN_FILL: df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    for col in df_clean.columns:
        if df_clean[col].isna().any():
            if pd.api.types.is_numeric_dtype(df_clean[col]): df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            else: df_clean[col] = df_clean[col].fillna(df_clean[col].mode(dropna=True)[0])
    return df_clean

# 2. Function engineer features
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df_feat = df.copy()
    df_feat["TotalSF"] = df_feat["TotalBsmtSF"] + df_feat["1stFlrSF"] + df_feat["2ndFlrSF"]
    df_feat["HouseAge"] = df_feat["YrSold"] - df_feat["YearBuilt"]
    df_feat["RemodAge"] = df_feat["YrSold"] - df_feat["YearRemodAdd"]
    return df_feat

# 3. Endcode categorical
def encode_categorical(df: pd.DataFrame, ordinal_cols: list[str] = ORDINAL_COLS) -> pd.DataFrame:
    df_encoded = df.copy()
    for col in ordinal_cols:
        if col in df_encoded.columns: df_encoded[col] = df_encoded[col].map(ORDINAL_MAP)
    object_cols = df_encoded.select_dtypes(include=["object", "category"]).columns.tolist()
    df_encoded = pd.get_dummies(df_encoded, columns=object_cols, dummy_na=False)
    return df_encoded

# 4. Function scale numeric
def scale_numeric(df: pd.DataFrame, skew_threshold: float = SKEW_THRESHOLD) -> pd.DataFrame:
    df_scaled = df.copy()
    numeric_cols = df_scaled.select_dtypes(include=[np.number]).columns.tolist()
    feature_numeric_cols = [col for col in numeric_cols if col != "SalePrice"]

    skew_values = df_scaled[feature_numeric_cols].skew()
    skewed_cols = skew_values[skew_values.abs() > skew_threshold].index.tolist()

    for col in skewed_cols:
        if not (df_scaled[col] <= -1).any(): df_scaled[col] = np.log1p(df_scaled[col])

    scaler = StandardScaler()
    cols_to_scale = [col for col in feature_numeric_cols if col in df_scaled.columns]
    if cols_to_scale: df_scaled[cols_to_scale] = scaler.fit_transform(df_scaled[cols_to_scale])
    return df_scaled

# 5. Function preprocess
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = handle_missing_values(df)
    df_feat = engineer_features(df_clean)
    df_encoded = encode_categorical(df_feat)
    return scale_numeric(df_encoded)

# 6. Main execution block
if __name__ == '__main__':
    try:
        df_raw = pd.read_csv('data/raw/train.csv')
        
        #Run the entire pipeline
        print("Đang xử lý dữ liệu qua Pipeline...")
        df_final = preprocess(df_raw)
       
        print("\nSuccessfully!")
        print(f"Size before processing: {df_raw.shape}")
        print(f"Size after processing: {df_final.shape}")
        
        print("\nA few rows of the processed features:")
        print(df_final[['TotalSF', 'HouseAge', 'RemodAge']].head())
        
    except FileNotFoundError:
        print("Error: Please ensure the 'data/raw/train.csv' file exists in the specified path.")