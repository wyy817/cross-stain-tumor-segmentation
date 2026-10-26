#!/usr/bin/env bash
# ============================================================
# Server Environment Setup for FYP_S2
# Server: A40 GPU, CUDA 12.4, Linux
# Run from FYP_S2 root: bash scripts/server_setup.sh
# ============================================================
set -e

FYP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
echo "FYP_S2 root: $FYP_DIR"

# ── 1. Create conda environment ──────────────────────────────
echo ""
echo "=== Step 1: Create conda env fyp_new (Python 3.10) ==="
mamba create -n fyp_new python=3.10 -y

# Activate — must use conda run for scripted activation
CONDA_RUN="conda run -n fyp_new --no-capture-output"

# ── 2. PyTorch 2.6.0 + CUDA 12.4 ────────────────────────────
echo ""
echo "=== Step 2: Install PyTorch 2.6.0 + cu124 ==="
$CONDA_RUN pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124

# ── 3. mmengine ──────────────────────────────────────────────
echo ""
echo "=== Step 3: Install mmengine ==0.10.5 ==="
$CONDA_RUN pip install mmengine==0.10.5

# ── 4. mmcv (via openmim — auto-detects CUDA/torch version) ──
echo ""
echo "=== Step 4: Install mmcv ==2.2.0 via mim ==="
$CONDA_RUN pip install openmim
$CONDA_RUN mim install "mmcv==2.2.0"

# If mim fails to find a wheel for torch2.6+cu124, uncomment the fallback:
# $CONDA_RUN pip install mmcv==2.2.0 \
#     -f https://download.openmmlab.com/mmcv/dist/cu124/torch2.6/index.html

# ── 5. mmsegmentation ────────────────────────────────────────
echo ""
echo "=== Step 5: Install mmsegmentation ==1.2.2 ==="
$CONDA_RUN pip install mmsegmentation==1.2.2

# ── 6. Patch mmseg MMCV version check if needed ──────────────
# mmseg 1.2.2 checks mmcv <= '2.2.0'. If installed mmcv is 2.2.0
# this should be fine. Only uncomment if you see "MMCV" version error.
# MMSEG_INIT=$(conda run -n fyp_new python -c "import mmseg; print(mmseg.__file__)")
# sed -i "s/MMCV_MAX = '2.2.0'/MMCV_MAX = '2.3.0'/" "$MMSEG_INIT"

# ── 7. Other dependencies ────────────────────────────────────
echo ""
echo "=== Step 6: Install other dependencies ==="
$CONDA_RUN pip install \
    "numpy==1.26.4" \
    "timm==0.9.16" \
    "Pillow>=10.0" \
    "scikit-image>=0.21" \
    "tqdm" \
    "pandas"

# ── 8. Install CONCH (editable) ──────────────────────────────
echo ""
echo "=== Step 7: Install CONCH package ==="
$CONDA_RUN pip install -e "$FYP_DIR/CONCH"

# ── 9. Verify installation ───────────────────────────────────
echo ""
echo "=== Step 8: Verify ==="
$CONDA_RUN python - <<'EOF'
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

import mmengine; print(f"mmengine: {mmengine.__version__}")
import mmcv;     print(f"mmcv:     {mmcv.__version__}")
import mmseg;    print(f"mmseg:    {mmseg.__version__}")
import timm;     print(f"timm:     {timm.__version__}")
import numpy;    print(f"numpy:    {numpy.__version__}")

from conch.open_clip_custom import create_model_from_pretrained
print("CONCH import: OK")
EOF

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Next steps:"
echo "  1. Update CSV paths:  conda run -n fyp_new python $FYP_DIR/scripts/patch_for_server.py"
echo "  2. Run training E2:   conda run -n fyp_new python $FYP_DIR/scripts/02_train.py --config $FYP_DIR/configs/E2_conch_fpn.py"
echo "  3. Run training E3:   conda run -n fyp_new python $FYP_DIR/scripts/02_train.py --config $FYP_DIR/configs/E3_conch_fpn_aug.py"
