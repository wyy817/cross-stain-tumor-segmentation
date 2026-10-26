#!/bin/bash
# ============================================================
# FYP fyp_new Environment Setup (Linux Server, A40 GPU)
# ============================================================

ENV_NAME=fyp_new
PROJECT_DIR=/path/to/FYP_S2        # [SERVER] Change to actual path, e.g. /home/user/FYP_S2
CONCH_DIR=${PROJECT_DIR}/CONCH

echo "[1/5] Creating conda environment: ${ENV_NAME} ..."
conda create -n ${ENV_NAME} python=3.10 -y
conda activate ${ENV_NAME}

echo "[2/5] Installing PyTorch 2.6 with CUDA 12.4 (A40 is Ampere, sm_86) ..."
# A40: CUDA 12.4 is fine; use cu124 or cu121
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

echo "[3/5] Installing mmsegmentation stack ..."
pip install -U openmim
mim install mmengine==0.10.5
mim install "mmcv>=2.1.0,<2.2.0"
mim install mmsegmentation==1.2.2

echo "[4/5] Installing CONCH and dependencies ..."
pip install timm==0.9.16 huggingface_hub transformers tokenizers
cd ${CONCH_DIR} && pip install -e . && cd ${PROJECT_DIR}

echo "[5/5] Installing other dependencies ..."
pip install scikit-image scipy pandas Pillow h5py matplotlib

echo ""
echo "============================================================"
echo " Environment [${ENV_NAME}] setup complete!"
echo " Run: python download_weights.py --token YOUR_HF_TOKEN"
echo "============================================================"
