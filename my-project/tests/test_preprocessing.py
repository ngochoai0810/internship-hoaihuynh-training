"""
tests/test_preprocessing.py

Kiểm thử tự động cho module preprocessing.
Đảm bảo hàm xử lý đúng logic, xử lý ngoại lệ an toàn và validate kết quả Pipeline.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Cấu hình đường dẫn tuỳ thuộc vào kiến trúc của bạn, điều chỉnh nếu cần.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from preprocessing import handle_missing_values, engineer_features, preprocess_train, preprocess_test


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


# --- TEST ERROR HANDLING ---

def test_missing_required_columns():
    """Đảm bảo hàm văng lỗi ValueError nếu thiếu các cột nằm trong NA_MEANS_NONE."""
    bad_df = pd.DataFrame({"SomeRandomColumn": [1, 2, 3]})
    with pytest.raises(ValueError, match="Thiếu các cột bắt buộc"):
        handle_missing_values(bad_df)

def test_empty_dataframe():
    """Đảm bảo hàm chặn xử lý DataFrame rỗng."""
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError, match="đang trống"):
        handle_missing_values(empty_df)


# --- TEST LOGIC & IMMUTABILITY ---

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
    
    # Sửa lỗi Syntax ở đây: dùng comment (#) thay vì ngoặc đơn
    assert df_feat["TotalSF"].iloc[0] == 1000   # 500 + 500 + 0
    assert df_feat["HouseAge"].iloc[0] == 10    # 2010 - 2000
    assert df_feat["RemodAge"].iloc[0] == 5     # 2010 - 2005


# --- TEST PIPELINE END-TO-END ---

def test_preprocess_train_pipeline(mock_missing_data, tmp_path):
    """
    Kiểm tra toàn bộ pipeline: Không còn NaN, Scaling hoạt động tốt.
    Sử dụng tmp_path của pytest để lưu trữ file preprocessor.pkl tạm thời.
    """
    # Chuẩn bị dữ liệu mô phỏng
    df = handle_missing_values(mock_missing_data)
    df["GarageType"] = ["Attchd", "Detchd", "Attchd"] # Thêm cột cat để test OneHot
    df["ExterQual"] = ["TA", "Gd", "Ex"]              # Thêm cột ord để test Ordinal
    
    num_cols = ["GarageArea", "LotFrontage"]
    cat_cols = ["GarageType"]
    ord_cols = ["ExterQual"]
    
    save_filepath = tmp_path / "test_preprocessor.pkl"
    
    X_processed = preprocess_train(
        df=df, 
        num_cols=num_cols, 
        cat_cols=cat_cols, 
        ord_cols=ord_cols,
        save_path=str(save_filepath)
    )
    
    # 1. Kiểm tra File pkl đã được tạo ra chưa
    assert save_filepath.exists(), "Pipeline chưa được lưu thành file .pkl"
    
    # 2. Kiểm tra định dạng đầu ra
    assert isinstance(X_processed, np.ndarray), "Kết quả trả về phải là NumPy Array"
    
    # 3. Kiểm tra tính năng Standard Scaler (Mean xấp xỉ 0, Std xấp xỉ 1)
    # 2 cột đầu tiên trong X_processed sẽ tương ứng với num_cols (do ColumnTransformer xếp lên đầu)
    mean_val = np.mean(X_processed[:, 0]) # GarageArea
    std_val = np.std(X_processed[:, 0])
    
    assert np.isclose(mean_val, 0, atol=1e-7), "Scaler chưa đưa Mean về 0"
    assert np.isclose(std_val, 1, atol=1e-7), "Scaler chưa đưa Std về 1"