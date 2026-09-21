#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .

python my-project/src/scripts/download_house_prices.py --destination my-project/data/raw
python -m ml.baseline_round2
python -c "from ml.ridge_tuning import run_ridge_tuning; result = run_ridge_tuning(); print(result.candidates.to_string(index=False))"
python -m ml.ensemble_round3
python -m ml.finalize --allow-overwrite
