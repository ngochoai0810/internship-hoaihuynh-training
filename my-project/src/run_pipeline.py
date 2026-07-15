import sys
from pathlib import Path

import pandas as pd

from ml.preprocessing import preprocess_test, preprocess_train


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data" / "raw"
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.pkl"


def main():
    # 1. Tai du lieu
    print("Dang tai du lieu...")
    df_train = pd.read_csv(DATA_DIR / "train.csv")
    df_test = pd.read_csv(DATA_DIR / "test.csv")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Tach target vi train co SalePrice nhung test khong co cot nay.
    if "SalePrice" in df_train.columns:
        y_train = df_train["SalePrice"]
        df_train = df_train.drop(columns=["SalePrice"])
        print(f"Da tach target SalePrice: {y_train.shape}")

    # Khai bao cac tap cot (ban tu dieu chinh theo data thuc te)
    num_cols = ["LotFrontage", "MasVnrArea", "TotalBsmtSF"]
    cat_cols = ["GarageType", "Alley"]
    ord_cols = ["ExterQual"]

    # 2. Chay Pipeline cho tap Train (Fit & Transform)
    print("Chay tien xu ly tren tap Train...")
    X_train_processed = preprocess_train(
        df=df_train,
        num_cols=num_cols,
        cat_cols=cat_cols,
        ord_cols=ord_cols,
        save_path=str(PREPROCESSOR_PATH),
    )
    print(f"Kich thuoc X_train sau xu ly: {X_train_processed.shape}")

    # 3. Chay Pipeline cho tap Test (Chi Transform)
    print("Chay tien xu ly tren tap Test...")
    X_test_processed = preprocess_test(
        df=df_test,
        load_path=str(PREPROCESSOR_PATH),
    )
    print(f"Kich thuoc X_test sau xu ly: {X_test_processed.shape}")

if __name__ == "__main__":
    main()
