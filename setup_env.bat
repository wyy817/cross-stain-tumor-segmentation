@echo off
:: ============================================================
:: FYP fyp_new Environment Setup (Windows, miniforge3)
:: RTX 5070 (Blackwell, sm_100) requires PyTorch with CUDA 12.6+
:: ============================================================

SET CONDA_ROOT=C:\path\to\miniforge3
SET ENV_NAME=fyp_new
SET PROJECT_DIR=D:\XJTLU\FYP_S2

CALL %CONDA_ROOT%\Scripts\activate.bat

echo [1/5] Creating conda environment: %ENV_NAME% ...
CALL conda create -n %ENV_NAME% python=3.10 -y
CALL conda activate %ENV_NAME%

echo [2/5] Installing PyTorch 2.6 with CUDA 12.6 (for RTX 5070 Blackwell) ...
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126

echo [3/5] Installing mmsegmentation stack ...
pip install -U openmim
mim install mmengine==0.10.5
mim install "mmcv>=2.1.0,<2.2.0"
mim install mmsegmentation==1.2.2

echo [4/5] Installing CONCH and dependencies ...
pip install timm==0.9.16 huggingface_hub transformers tokenizers
cd %PROJECT_DIR%\CONCH && pip install -e . && cd %PROJECT_DIR%

echo [5/5] Installing other dependencies ...
pip install scikit-image scipy pandas Pillow h5py matplotlib

echo.
echo ============================================================
echo  Environment [%ENV_NAME%] setup complete!
echo  CONCH weights: run  python download_weights.py  after
echo  getting HF token from https://huggingface.co/settings/tokens
echo  (Accept terms at: https://huggingface.co/MahmoodLab/CONCH)
echo ============================================================
pause
