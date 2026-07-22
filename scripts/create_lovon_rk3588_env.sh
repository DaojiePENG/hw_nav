#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-}"
TOOLKIT_TAG="v2.3.2"
TOOLKIT_COMMIT="42aa1d426c0a9e0869b6374edba009f7208a1926"
CACHE_DIR="${LOVON_RKNN_TOOLKIT_DIR:-${HOME}/.cache/lovon/rknn-toolkit2-${TOOLKIT_TAG}}"

if [[ "$MODE" != "convert" && "$MODE" != "board" ]]; then
    echo "Usage: $0 {convert|board}" >&2
    exit 2
fi
if ! command -v conda >/dev/null 2>&1; then
    echo "Error: conda not found. Install Miniforge/Miniconda first." >&2
    exit 1
fi

if [[ ! -d "$CACHE_DIR/.git" ]]; then
    mkdir -p "$(dirname "$CACHE_DIR")"
    git clone --branch "$TOOLKIT_TAG" --depth 1 --filter=blob:none --sparse \
        https://github.com/airockchip/rknn-toolkit2.git "$CACHE_DIR"
fi
ACTUAL_COMMIT="$(git -C "$CACHE_DIR" rev-parse HEAD)"
if [[ "$ACTUAL_COMMIT" != "$TOOLKIT_COMMIT" ]]; then
    echo "Error: RKNN Toolkit commit mismatch: expected $TOOLKIT_COMMIT, got $ACTUAL_COMMIT" >&2
    exit 1
fi
git -C "$CACHE_DIR" sparse-checkout init --no-cone
git -C "$CACHE_DIR" sparse-checkout set --no-cone \
    /rknn-toolkit2/packages/x86_64/requirements_cp310-2.3.2.txt \
    /rknn-toolkit2/packages/x86_64/rknn_toolkit2-2.3.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl \
    /rknn-toolkit-lite2/packages/rknn_toolkit_lite2-2.3.2-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl

if [[ "$MODE" == "convert" ]]; then
    if [[ "$(uname -m)" != "x86_64" ]]; then
        echo "Error: RKNN conversion environment must run on x86_64 Linux." >&2
        exit 1
    fi
    ENV_NAME="${LOVON_RKNN_CONVERT_ENV_NAME:-lovon-rknn-convert}"
    if ! conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
        conda create --solver classic -y -n "$ENV_NAME" python=3.10 pip
    fi
    # RKNN Toolkit 2.3.2 imports pkg_resources, which was removed from
    # setuptools 81+.  Pin it before importing the proprietary wheel.
    conda run -n "$ENV_NAME" python -m pip install "setuptools<81"
    PACKAGE_DIR="$CACHE_DIR/rknn-toolkit2/packages/x86_64"
    conda run -n "$ENV_NAME" python -m pip install \
        -r "$PACKAGE_DIR/requirements_cp310-2.3.2.txt"
    # The upstream requirement is onnx>=1.16.1, but ONNX 1.22 removed the
    # onnx.mapping API still used by RKNN Toolkit 2.3.2.
    conda run -n "$ENV_NAME" python -m pip install onnx==1.16.1
    conda run -n "$ENV_NAME" python -m pip install \
        "$PACKAGE_DIR/rknn_toolkit2-2.3.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl" \
        huggingface-hub==0.33.4
    conda run -n "$ENV_NAME" python -c \
        "from rknn.api import RKNN; import importlib.metadata as m; print('rknn-toolkit2', m.version('rknn-toolkit2'))"
    echo "Conversion environment ready: $ENV_NAME"
else
    if [[ "$(uname -m)" != "aarch64" ]]; then
        echo "Error: board environment must be created on an aarch64 RK3588 host." >&2
        exit 1
    fi
    ENV_NAME="${LOVON_RK3588_ENV_NAME:-lovon-rk3588}"
    if ! conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
        conda create --solver classic -y -n "$ENV_NAME" python=3.10 pip
    fi
    conda run -n "$ENV_NAME" python -m pip install "setuptools<81"
    LITE_DIR="$CACHE_DIR/rknn-toolkit-lite2/packages"
    conda run -n "$ENV_NAME" python -m pip install \
        "$LITE_DIR/rknn_toolkit_lite2-2.3.2-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl"
    conda run -n "$ENV_NAME" python -m pip install \
        -r "$ROOT_DIR/requirements-lovon-pro-rk3588.txt"
    conda run -n "$ENV_NAME" python -m pip install --no-deps -e "$ROOT_DIR"
    conda run -n "$ENV_NAME" python -c \
        "from rknnlite.api import RKNNLite; import cv2, numpy; print('RKNNLite OK', cv2.__version__, numpy.__version__)"
    if [[ -r /sys/kernel/debug/rknpu/version ]]; then
        echo "RKNPU driver: $(tr -d '\n' </sys/kernel/debug/rknpu/version)"
    else
        echo "Warning: cannot read /sys/kernel/debug/rknpu/version; verify the BSP driver is >= 2.3.2 before benchmarking." >&2
    fi
    echo "Board environment ready: $ENV_NAME"
fi
