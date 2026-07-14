"""
tests/test_preprocessing.py

Kiểm thử tự động cho module preprocessing.
Đảm bảo hàm xử lý đúng logic và không làm biến đổi dữ liệu gốc (immutability).
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/mlc')))

from preprocessing import handle_missing_values, engineer_features



@pytest.fixture
def mock_missing_data():
    """Tạo ra một DataFrame nhỏ để test hàm handle_missing_values."""
    return pd.DataFrame({
        "PoolQC": [np.nan, "Ex"],           # Cột Categorical -> Cần điền "None"
        "GarageArea": [np.nan, 500],        # Cột Numerical -> Cần điền 0
        "LotFrontage": [50.0, np.nan],      # Cột điền Median -> Sẽ điền 50.0
        "Fence": ["MnPrv", np.nan],         # Thuộc NA_MEANS_NONE nhưng df mẫu bị thiếu nhiều cột bắt buộc khác -> sẽ trigger Exception nếu không mock đủ.
        # Thêm các cột bắt buộc để vượt qua check `missing_required`
        **{col: [np.nan, np.nan] for col in [
            "Alley", "FireplaceQu", "GarageType", "GarageFinish", 
            "GarageQual", "GarageCond", "BsmtQual", "BsmtCond", 
            "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "MiscFeature"
        ]}
    })

@pytest.fixture
def mock_feature_data():
    """Tạo DataFrame nhỏ để test Feature Engineering."""
    return pd.DataFrame({
        "TotalBsmtSF": [500, 1000],
        "1stFlrSF": [500, 1000],
        "2ndFlrSF": [0, 500],
        "YrSold": [2010, 2020],
        "YearBuilt": [2000, 2015],
        "YearRemodAdd": [2005, 2018]
    })


# --- Các Test Cases ---

def test_handle_missing_values_no_mutation(mock_missing_data):
    """Đảm bảo hàm tạo ra df mới, không sửa df gốc."""
    original_df = mock_missing_data.copy()
    _ = handle_missing_values(mock_missing_data)
    pd.testing.assert_frame_equal(mock_missing_data, original_df)


def test_handle_missing_values_logic(mock_missing_data):
    """Kiểm tra xem dữ liệu có được điền đúng logic hay không."""
    df_clean = handle_missing_values(mock_missing_data)
    
    assert df_clean["PoolQC"].iloc[0] == "None", "Lỗi: Chưa điền 'None' cho PoolQC"
    assert df_clean["GarageArea"].iloc[0] == 0, "Lỗi: Chưa điền 0 cho GarageArea"
    assert df_clean["LotFrontage"].iloc[1] == 50.0, "Lỗi: Chưa điền Median cho LotFrontage"
    assert df_clean.isnull().sum().sum() == 0, "Lỗi: Vẫn còn giá trị NaN trong DataFrame"


def test_engineer_features_logic(mock_feature_data):
    """Kiểm tra các cột đặc trưng mới có được tính toán đúng toán học không."""
    df_feat = engineer_features(mock_feature_data)
    
    # Kiểm tra cột mới có tồn tại
    assert "TotalSF" in df_feat.columns
    assert "HouseAge" in df_feat.columns
    assert "RemodAge" in df_feat.columns
    
    # Kiểm tra phép tính dòng đầu tiên
    assert df_feat["TotalSF"].iloc[0] == 1000 (500 + 500 + 0)
    assert df_feat["HouseAge"].iloc[0] == 10 (2010 - 2000)
    assert df_feat["RemodAge"].iloc[0] == 5 (2010 - 2005)