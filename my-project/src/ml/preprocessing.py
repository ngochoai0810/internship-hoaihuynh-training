"""
src/ml/preprocessing.py
 
Toàn bộ pipeline tiền xử lý (preprocessing) cho tập dữ liệu Kaggle House Prices.
Chuyển đổi DataFrame gốc thành ma trận đặc trưng (feature matrix) sẵn sàng cho mô hình.
"""
 
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
 
# Constants
# Value NA not means missing
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
    # Check if all required columns are present in the DataFrame
    missing_required = [col for col in NA_MEANS_NONE if col not in df.columns]
    if missing_required:
        raise ValueError(
            f"Missing required columns for NA_MEANS_NONE: {missing_required}"
        )

    # Always create copy of DF
    df_clean = df.copy()

    for col in NA_MEANS_NONE:
        df_clean[col] = df_clean[col].fillna("None")

    for col in NUMERIC_ZEROS:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(0)

    for col in MEDIAN_FILL:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    # Solve remaining missing values
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
    #Create new features based on existing ones
    df_feat = df.copy()
    df_feat["TotalSF"] = df_feat["TotalBsmtSF"] + df_feat["1stFlrSF"] + df_feat["2ndFlrSF"]
    df_feat["HouseAge"] = df_feat["YrSold"] - df_feat["YearBuilt"]
    df_feat["RemodAge"] = df_feat["YrSold"] - df_feat["YearRemodAdd"]
    return df_feat
 
 
def encode_categorical(
    df: pd.DataFrame,
    ordinal_cols: list[str] = ORDINAL_COLS,
) -> pd.DataFrame:
    # Encode columns with ordinal mapping and one-hot encode the rest of categorical columns
    df_encoded = df.copy()

    for col in ordinal_cols:
        if col in df_encoded.columns:
            df_encoded[col] = df_encoded[col].map(ORDINAL_MAP)

    object_cols = df_encoded.select_dtypes(include=["object", "category"]).columns.tolist()
    df_encoded = pd.get_dummies(df_encoded, columns=object_cols, dummy_na=False)
    return df_encoded
 
 
def scale_numeric(
    df: pd.DataFrame,
    skew_threshold: float = SKEW_THRESHOLD,
) -> pd.DataFrame:
    # Scale numeric features using StandardScaler and apply log transformation to skewed features
    df_scaled = df.copy()

    numeric_cols = df_scaled.select_dtypes(include=[np.number]).columns.tolist()
    feature_numeric_cols = [col for col in numeric_cols if col != "SalePrice"]
    skew_values = df_scaled[feature_numeric_cols].skew()
    skewed_cols = skew_values[skew_values.abs() > skew_threshold].index.tolist()

    for col in skewed_cols:
        if (df_scaled[col] <= -1).any():
            continue
        df_scaled[col] = np.log1p(df_scaled[col])

    scaler = StandardScaler()
    cols_to_scale = [col for col in feature_numeric_cols if col in df_scaled.columns]
    if cols_to_scale:
        df_scaled[cols_to_scale] = scaler.fit_transform(df_scaled[cols_to_scale])

    return df_scaled
 
 
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
     # Run all preprocessing steps in sequence
    df_clean = handle_missing_values(df)
    df_feat = engineer_features(df_clean)
    df_encoded = encode_categorical(df_feat)
    return scale_numeric(df_encoded)


# Test the preprocessing pipeline
if __name__ == '__main__':
    try:
        df_raw = pd.read_csv('../../data/raw/train.csv')
        
        df_clean = handle_missing_values(df_raw)
        print(f'Number of missing values after cleaning: {df_clean.isnull().sum().sum()}')  # Expected: 0
        
        df_feat = engineer_features(df_clean)
        print("\nSome engineered features:")
        print(df_feat[['TotalSF', 'HouseAge', 'RemodAge']].head())
        
        # Test all the pipeline steps together
        df_final = preprocess(df_raw)
        print(f"\nSize of final DataFrame: {df_final.shape}")
        
    except FileNotFoundError:
        print("Error: Can't find the file '../../data/raw/train.csv'. Please ensure the path is correct.")