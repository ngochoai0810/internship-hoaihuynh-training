"""
Kiểm thử các trường hợp ngoại lệ (edge cases) và tính nhất quán của mô hình.
"""

import pytest
import pandas as pd
import numpy as np
import joblib
import sys
import os

# Đường dẫn chỉ tới thư mục chứa file preprocessing.py của bạn
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/ml')))

# KHÔNG import PipelineConfig nữa
from preprocessing import preprocess_train, preprocess_test


@pytest.fixture
def base_dummy_data():
    """Tạo dữ liệu cơ sở vượt qua được bước kiểm tra handle_missing_values."""
    required_cols = [
        "PoolQC", "Alley", "Fence", "FireplaceQu",
        "GarageType", "GarageFinish", "GarageQual", "GarageCond",
        "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
        "MiscFeature"
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


def test_unseen_categories_in_test_set(base_dummy_data, tmp_path):
    """
    Đảm bảo Pipeline không crash khi tập Test xuất hiện category lạ.
    """
    save_path = str(tmp_path / "preprocessor.pkl")
    
    # Định nghĩa trực tiếp các cột thay vì dùng config
    num_cols = ["GarageArea", "LotFrontage"]
    cat_cols = ["GarageType"]
    ord_cols = ["ExterQual"]
    
    # 1. Huấn luyện với dữ liệu gốc
    X_train = preprocess_train(
        df=base_dummy_data, 
        num_cols=num_cols,
        cat_cols=cat_cols,
        ord_cols=ord_cols,
        save_path=save_path
    )
    
    # 2. Tạo dữ liệu Test chứa category lạ ('Basment', 'BuiltIn')
    df_test = base_dummy_data.copy()
    df_test["GarageType"] = ["Basment", "BuiltIn"]
    
    # 3. Transform dữ liệu Test
    X_test = preprocess_test(df=df_test, load_path=save_path)
    
    # 4. Kiểm chứng
    assert X_train.shape[1] == X_test.shape[1], "Lỗi: Số lượng cột ở tập Test bị sai lệch!"


def test_joblib_serialization_consistency(base_dummy_data, tmp_path):
    """
    Đảm bảo model trên RAM và model load từ file .pkl cho ra kết quả giống hệt nhau.
    """
    save_path = str(tmp_path / "preprocessor_consistency.pkl")
    
    num_cols = ["GarageArea", "LotFrontage"]
    cat_cols = ["GarageType"]
    ord_cols = ["ExterQual"]
    
    # 1. Chạy train và lấy mảng Numpy
    X_train_memory = preprocess_train(
        df=base_dummy_data, 
        num_cols=num_cols,
        cat_cols=cat_cols,
        ord_cols=ord_cols,
        save_path=save_path
    )
    
    # 2. Load lại preprocessor
    loaded_preprocessor = joblib.load(save_path)
    
    # 3. Chạy pandas thủ công
    from preprocessing import handle_missing_values, engineer_features
    df_clean = handle_missing_values(base_dummy_data)
    df_feat = engineer_features(df_clean)
    
    # 4. Transform lại
    X_train_loaded = loaded_preprocessor.transform(df_feat)
    
    # 5. Kiểm chứng độ chính xác
    np.testing.assert_array_equal(
        X_train_memory, 
        X_train_loaded, 
        err_msg="Lỗi: Dữ liệu biến đổi từ model load lên không khớp!"
    )