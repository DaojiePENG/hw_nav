#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${LOVON_RKNN_CONVERT_ENV_NAME:-lovon-rknn-convert}"

"$ROOT_DIR/scripts/create_lovon_rk3588_env.sh" convert
conda run -n "$ENV_NAME" python "$ROOT_DIR/scripts/download_lovon_rk3588_models.py"
conda run -n "$ENV_NAME" python "$ROOT_DIR/scripts/download_lovon_rk3588_models.py" --verify-only
conda run -n "$ENV_NAME" python "$ROOT_DIR/scripts/convert_lovon_rk3588_models.py" "$@"
conda run -n "$ENV_NAME" python "$ROOT_DIR/scripts/verify_lovon_rk3588_bundle.py"

echo
echo "RK3588 models are ready under models/lovon_pro_rk3588/."
echo "Copy that directory to the same path on the board, including conversion_report.json."
