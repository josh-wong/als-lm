#!/bin/bash
# ALS-LM Environment Setup Script
# Usage: bash setup.sh
# Creates venv, installs all dependencies, and validates the training environment.
# Safe to run multiple times (idempotent).
set -euo pipefail

# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

ok()   { echo -e "  ${GREEN}[OK]${NC} $1"; }
fail() { echo -e "  ${RED}[FAIL]${NC} $1"; }
warn() { echo -e "  ${YELLOW}[WARN]${NC} $1"; }
info() { echo -e "  ${BLUE}[INFO]${NC} $1"; }

echo ""
echo "=========================================="
echo "  ALS-LM Environment Setup"
echo "=========================================="
echo ""

# ---------------------------------------------------------------------------
# 1. Check system prerequisites
# ---------------------------------------------------------------------------
echo "Checking system prerequisites..."

if ! command -v nvidia-smi &>/dev/null; then
    fail "nvidia-smi not found. Install the NVIDIA GPU driver."
    exit 1
fi
ok "nvidia-smi found"

if [ ! -f /usr/local/cuda/bin/nvcc ]; then
    fail "nvcc not found at /usr/local/cuda/bin/nvcc. Install the CUDA toolkit."
    echo "  See: https://developer.nvidia.com/cuda-downloads"
    exit 1
fi
ok "nvcc found at /usr/local/cuda/bin/nvcc"

if ! command -v gcc &>/dev/null; then
    fail "gcc not found. Run: sudo apt install build-essential"
    exit 1
fi
ok "gcc found"

if ! python3 -c "import sysconfig; sysconfig.get_path('include')" &>/dev/null; then
    fail "Python 3 dev headers missing. Run: sudo apt install python3-dev"
    exit 1
fi
ok "Python 3 dev headers found"

echo ""

# ---------------------------------------------------------------------------
# 2. Set CUDA environment variables
# ---------------------------------------------------------------------------
echo "Setting CUDA environment variables..."

export PATH=/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}

ok "PATH includes /usr/local/cuda/bin"
ok "CUDA_HOME=$CUDA_HOME"
ok "LD_LIBRARY_PATH includes /usr/local/cuda/lib64"

echo ""

# ---------------------------------------------------------------------------
# 3. Detect and report system info
# ---------------------------------------------------------------------------
echo "System information:"

PYTHON_VER=$(python3 --version 2>&1)
CUDA_VER=$(nvcc --version 2>&1 | grep "release" || echo "unknown")
GCC_VER=$(gcc --version 2>&1 | head -1)
GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "unknown")
MEM_INFO=$(free -h 2>/dev/null | grep Mem || echo "unknown")

info "Python:    $PYTHON_VER"
info "CUDA:      $CUDA_VER"
info "gcc:       $GCC_VER"
info "GPU:       $GPU_INFO"
info "RAM:       $MEM_INFO"

# Warn if WSL2 RAM is less than 32GB
TOTAL_RAM_MB=$(free -m 2>/dev/null | awk '/^Mem:/ {print $2}' || echo "0")
if [ "$TOTAL_RAM_MB" -lt 32000 ] 2>/dev/null; then
    echo ""
    warn "WSL2 RAM is ${TOTAL_RAM_MB}MB (< 32GB)."
    warn "DeepSpeed CPU offloading needs substantial RAM for optimizer states."
    warn "Edit C:\\Users\\<you>\\.wslconfig and set memory=58GB (or higher),"
    warn "then run 'wsl --shutdown' from PowerShell and restart WSL2."
fi

echo ""

# ---------------------------------------------------------------------------
# 4. Create or reuse virtual environment
# ---------------------------------------------------------------------------
echo "Setting up Python virtual environment..."

if [ ! -d ".venv" ]; then
    info "Creating new venv at .venv/"
    python3 -m venv .venv
    ok "Virtual environment created"
else
    ok "Existing .venv/ found, reusing"
fi

# shellcheck disable=SC1091
source .venv/bin/activate
ok "Virtual environment activated"

info "Upgrading pip..."
pip install --upgrade pip --quiet
ok "pip upgraded"

echo ""

# ---------------------------------------------------------------------------
# 5. Install PyTorch with CUDA 12.8 support
# ---------------------------------------------------------------------------
echo "Installing PyTorch with CUDA 12.8 support..."
info "Using index URL: https://download.pytorch.org/whl/cu128"

pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
ok "PyTorch installed"

echo ""

# ---------------------------------------------------------------------------
# 6. Install DeepSpeed with pre-built CPUAdam
# ---------------------------------------------------------------------------
echo "Installing DeepSpeed with pre-built CPUAdam extension..."
info "Setting DS_BUILD_CPU_ADAM=1 for C++ extension compilation"

DS_BUILD_CPU_ADAM=1 pip install deepspeed==0.18.6 ninja
ok "DeepSpeed installed"

echo ""

# ---------------------------------------------------------------------------
# 7. Install remaining dependencies from requirements.txt
# ---------------------------------------------------------------------------
echo "Installing remaining dependencies from requirements.txt..."
info "PyTorch and DeepSpeed already installed; pip will skip them"

pip install -r requirements.txt
ok "All dependencies installed"

echo ""

# ---------------------------------------------------------------------------
# 8. Quick validation checks
# ---------------------------------------------------------------------------
echo "Running validation checks..."

if python -c "import torch; assert torch.cuda.is_available(); print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}')"; then
    ok "PyTorch CUDA validation passed"
else
    fail "PyTorch cannot access CUDA. Check GPU driver and CUDA toolkit."
    exit 1
fi

if python -c "import deepspeed; print(f'DeepSpeed {deepspeed.__version__}')"; then
    ok "DeepSpeed import validation passed"
else
    fail "DeepSpeed import failed."
    exit 1
fi

if python -c "import transformers; print(f'transformers {transformers.__version__}')"; then
    ok "transformers import validation passed"
else
    fail "transformers import failed."
    exit 1
fi

if python -c "import safetensors; print(f'safetensors {safetensors.__version__}')"; then
    ok "safetensors import validation passed"
else
    fail "safetensors import failed."
    exit 1
fi

echo ""

# ---------------------------------------------------------------------------
# 9. Run ds_report and save output
# ---------------------------------------------------------------------------
echo "Running DeepSpeed environment report..."

mkdir -p reports
ds_report 2>&1 | tee reports/ds_report.txt

echo ""

# Check that CPUAdam compiled successfully
if grep -q "cpu_adam.*YES" reports/ds_report.txt; then
    ok "CPUAdam C++ extension is available"
else
    fail "CPUAdam C++ extension is NOT available."
    echo ""
    echo "  Troubleshooting steps:"
    echo "  1. Ensure nvcc is in PATH: which nvcc"
    echo "  2. Ensure CUDA_HOME is set: echo \$CUDA_HOME"
    echo "  3. Ensure gcc is compatible: gcc --version (CUDA 12.8 supports up to gcc 14)"
    echo "  4. Ensure Python dev headers: dpkg -l | grep python3-dev"
    echo "  5. Try reinstalling: DS_BUILD_CPU_ADAM=1 pip install --force-reinstall deepspeed==0.18.6"
    echo ""
    echo "  Without CPUAdam, training will be 5-7x slower (Python fallback)."
    echo "  This makes 500M model training impractical (20-84 days instead of 4-12)."
    exit 1
fi

echo ""

# ---------------------------------------------------------------------------
# 10. Summary
# ---------------------------------------------------------------------------
echo "=========================================="
echo "  Environment Setup Complete"
echo "=========================================="
echo ""
ok "PyTorch $(python -c 'import torch; print(torch.__version__)') with CUDA $(python -c 'import torch; print(torch.version.cuda)')"
ok "DeepSpeed $(python -c 'import deepspeed; print(deepspeed.__version__)') with CPUAdam"
ok "transformers $(python -c 'import transformers; print(transformers.__version__)')"
ok "safetensors $(python -c 'import safetensors; print(safetensors.__version__)')"
ok "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
ok "ds_report saved to reports/ds_report.txt"
echo ""
info "Activate the environment with: source .venv/bin/activate"
info "Run the stress test with: python scripts/stress_test.py"
echo ""
