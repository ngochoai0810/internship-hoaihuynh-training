#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${DEMO_REPO_ROOT:-"$(cd -- "$SCRIPT_DIR/../.." && pwd)"}
cd "$REPO_ROOT"

PYTHON=${PYTHON_EXECUTABLE:-.venv/Scripts/python.exe}
MODEL_PATH="my-project/model.pkl"
ARTIFACTS_DIR="my-project/artifacts"
SUBSET_PATH="my-project/data/retrain/demo_subset.csv"
SUBSET_SCRIPT="my-project/src/scripts/create_retrain_subset.py"
TRAIN_SCRIPT="my-project/src/ml/train.py"

echo "=== BƯỚC 1: Kiểm tra môi trường ==="

for required_file in "$PYTHON" "$MODEL_PATH" "$SUBSET_SCRIPT" "$TRAIN_SCRIPT"; do
    if [[ ! -f "$required_file" ]]; then
        echo "Không tìm thấy file: $required_file" >&2
        exit 1
    fi
done

echo "=== BƯỚC 2: Backup model hiện tại ==="

mkdir -p "$ARTIFACTS_DIR" "$(dirname -- "$SUBSET_PATH")"
stamp=$(date +"%Y%m%d_%H%M%S")
backup_path="$ARTIFACTS_DIR/model_backup_${stamp}.pkl"
cp -- "$MODEL_PATH" "$backup_path"

echo "Đã backup model: $backup_path"

echo "=== BƯỚC 3: Tạo subset mới ==="

"$PYTHON" "$SUBSET_SCRIPT" \
    --destination "$SUBSET_PATH" \
    --fraction 0.75 \
    --random-state 2027

echo "=== BƯỚC 4: Retrain Ridge model ==="

"$PYTHON" "$TRAIN_SCRIPT" \
    --data-path "$SUBSET_PATH" \
    --model ridge \
    --alpha 10

echo "=== RETRAIN HOÀN TẤT ==="
echo "Model mới: $MODEL_PATH"
echo "Model backup: $backup_path"
echo "Hãy restart FastAPI để nạp model mới."
