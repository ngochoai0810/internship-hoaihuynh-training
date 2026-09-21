#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"

: "${RENDER_DISK_MOUNT:=/var/data}"
: "${DATABASE_URL:=sqlite:///${RENDER_DISK_MOUNT}/app.db}"
: "${MODEL_PATH:=${RENDER_DISK_MOUNT}/model.pkl}"

export DATABASE_URL
export MODEL_PATH

mkdir -p "$RENDER_DISK_MOUNT"
if [[ ! -f "$MODEL_PATH" ]]; then
    cp my-project/model.pkl "$MODEL_PATH"
fi

alembic -c my-project/alembic.ini upgrade head
exec uvicorn api.main:app --app-dir my-project/src --host 0.0.0.0 --port "${PORT:-8000}"
