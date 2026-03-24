#!/bin/bash
set -e  # Exit on error

echo "========== Ultralytics Jetson Setup =========="

# --------------------------------------------------
# 1. CHECK SUDO
# --------------------------------------------------
sudo -v

# --------------------------------------------------
# 2. VERIFY JETSON + JETPACK VERSION
# --------------------------------------------------
if [ ! -f /etc/nv_tegra_release ]; then
    echo "❌ Not a Jetson device. Exiting."
    exit 1
fi

L4T_RELEASE=$(sed -n 's/.*R\([0-9]*\) (release).*/\1/p' /etc/nv_tegra_release)
L4T_REVISION=$(sed -n 's/.*REVISION: \([0-9.]*\).*/\1/p' /etc/nv_tegra_release)

if [[ "$L4T_RELEASE" != "36" || "$L4T_REVISION" != 5* ]]; then
    echo "❌ Unsupported JetPack version: R$L4T_RELEASE.$L4T_REVISION"
    echo "👉 Required: JetPack 6.2.x (L4T 36.5)"
    exit 1
fi

echo "✅ JetPack 6.2 detected"

# --------------------------------------------------
# 3. SYSTEM DEPENDENCIES
# --------------------------------------------------
echo "Installing system dependencies..."

sudo apt-get update
sudo apt-get install -y \
    python3-pip \
    python3-dev \
    python3-setuptools \
    python3-wheel \
    build-essential \
    libopenblas-dev \
    libjpeg-dev \
    zlib1g-dev

# --------------------------------------------------
# 4. FIX CUDA REPO + CUSPARSELT
# --------------------------------------------------
echo "Installing CUDA dependencies..."

wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/arm64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y libcusparselt0 libcusparselt-dev

# --------------------------------------------------
# 5. PYTHON BASE SETUP
# --------------------------------------------------
echo "Upgrading pip..."

python3 -m pip install --upgrade pip

# Remove conflicting torch versions
pip uninstall -y torch torchvision torchaudio || true

# --------------------------------------------------
# 6. INSTALL PYTORCH (JETSON-COMPATIBLE)
# --------------------------------------------------
echo "Installing PyTorch (Jetson optimized)..."

pip install \
https://github.com/ultralytics/assets/releases/download/v0.0.0/torch-2.5.0a0+872d972e41.nv24.08-cp310-cp310-linux_aarch64.whl

pip install \
https://github.com/ultralytics/assets/releases/download/v0.0.0/torchvision-0.20.0a0+afc54f7-cp310-cp310-linux_aarch64.whl

# --------------------------------------------------
# 7. CORE PYTHON LIBRARIES
# --------------------------------------------------
echo "Installing core dependencies..."

pip install numpy==1.26.4
pip install opencv-python==4.10.0.82
pip install pillow scipy matplotlib

# --------------------------------------------------
# 8. INSTALL ULTRALYTICS (AFTER TORCH)
# --------------------------------------------------
echo "Installing Ultralytics..."

pip install ultralytics

# --------------------------------------------------
# 9. OPTIONAL: ONNX + GPU RUNTIME
# --------------------------------------------------
echo "Installing ONNX + ONNXRuntime GPU..."

pip install onnx onnxslim

pip install \
https://github.com/ultralytics/assets/releases/download/v0.0.0/onnxruntime_gpu-1.23.0-cp310-cp310-linux_aarch64.whl

# --------------------------------------------------
# 10. VALIDATION
# --------------------------------------------------
echo "Validating installation..."

python3 - <<EOF
import torch
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

from ultralytics import YOLO
print("Ultralytics OK")
EOF

echo "========== ✅ INSTALLATION COMPLETE =========="
