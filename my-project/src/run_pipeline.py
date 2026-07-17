from pathlib import Path

import pandas as pd
from ml.preprocessing import preprocess_test, preprocess_train

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data" / "raw"
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.pkl"


def main() -> None:
    print("Loading raw data...")
    df_train = pd.read_csv(DATA_DIR / "train.csv")
    df_test = pd.read_csv(DATA_DIR / "test.csv")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    if "SalePrice" in df_train.columns:
        y_train = df_train["SalePrice"]
        df_train = df_train.drop(columns=["SalePrice"])
        print(f"Separated SalePrice target: {y_train.shape}")

    num_cols = ["LotFrontage", "MasVnrArea", "TotalBsmtSF"]
    cat_cols = ["GarageType", "Alley"]
    ord_cols = ["ExterQual"]

    print("Running preprocessing on train data...")
    x_train_processed = preprocess_train(
        df=df_train,
        num_cols=num_cols,
        cat_cols=cat_cols,
        ord_cols=ord_cols,
        save_path=str(PREPROCESSOR_PATH),
    )
    print(f"Processed train shape: {x_train_processed.shape}")

    print("Running preprocessing on test data...")
    x_test_processed = preprocess_test(
        df=df_test,
        load_path=str(PREPROCESSOR_PATH),
    )
    print(f"Processed test shape: {x_test_processed.shape}")


if __name__ == "__main__":
    main()
