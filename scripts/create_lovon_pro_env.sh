#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${LOVON_PRO_ENV_NAME:-lovon-pro}"
PLATFORM="${1:-cpu}"

if ! command -v conda >/dev/null 2>&1; then
    echo "Error: conda not found. Install Miniconda/Miniforge first." >&2
    exit 1
fi

case "$PLATFORM" in
    cpu|cuda124|jetson) ;;
    *)
        echo "Usage: $0 {cpu|cuda124|jetson}" >&2
        exit 2
        ;;
esac

if ! conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
    conda create -y -n "$ENV_NAME" python=3.10.18 pip=25.1
fi

if [[ "$PLATFORM" == "cpu" ]]; then
    conda run -n "$ENV_NAME" python -m pip install \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.5.1 torchvision==0.20.1
elif [[ "$PLATFORM" == "cuda124" ]]; then
    conda run -n "$ENV_NAME" python -m pip install \
        --index-url https://download.pytorch.org/whl/cu124 \
        torch==2.5.1 torchvision==0.20.1
else
    cat <<'EOF'
Jetson 的 PyTorch wheel 与 JetPack/L4T 严格绑定。
请先按 NVIDIA Jetson 文档在 lovon-pro 环境中安装对应 torch/torchvision，
确认 `conda run -n lovon-pro python -c "import torch; print(torch.__version__)"` 成功后按回车继续。
EOF
    read -r
    conda run -n "$ENV_NAME" python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__)"
fi

conda run -n "$ENV_NAME" python -m pip install -r "$ROOT_DIR/requirements-lovon-pro-lock.txt"
conda run -n "$ENV_NAME" python -m pip install --no-deps -e "$ROOT_DIR"

echo
echo "Environment ready. Verify with:"
echo "  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n $ENV_NAME pytest -q tests/test_lovon_agent_pro.py"
echo "  conda run -n $ENV_NAME python scripts/smoke_test_lovon_pro.py --mock"
