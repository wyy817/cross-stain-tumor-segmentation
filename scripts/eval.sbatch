#!/bin/bash
# ============================================================
# SLURM job script for cross-stain evaluation (all 3 experiments)
# Usage:
#   sbatch scripts/eval.sbatch
# Output:
#   logs/eval_<JOBID>.out
# ============================================================

#SBATCH --job-name=fyp_eval
#SBATCH --output=/path/to/FYP_S2/logs/eval_%j.out
#SBATCH --error=/path/to/FYP_S2/logs/eval_%j.err
#SBATCH --partition=aiaca40
#SBATCH --qos=4a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH --time=02:00:00

FYP_DIR=/path/to/FYP_S2
CONDA_BASE=/path/to/miniforge3

source $CONDA_BASE/etc/profile.d/conda.sh
conda activate fyp_new

cd $FYP_DIR
export PYTHONPATH=$FYP_DIR

echo "=============================="
echo "Job ID: $SLURM_JOB_ID  Node: $SLURMD_NODENAME"
echo "=============================="

# ── Find best checkpoints ──────────────────────────────────
E1_CKPT=$(ls $FYP_DIR/work_dirs/E1_resnet_fpn/best_mDice_iter_*.pth 2>/dev/null | tail -1)
E2_CKPT=$(ls $FYP_DIR/work_dirs/E2_conch_fpn/best_mDice_iter_*.pth 2>/dev/null | tail -1)
E3_CKPT=$(ls $FYP_DIR/work_dirs/E3_conch_fpn_aug/best_mDice_iter_*.pth 2>/dev/null | tail -1)

echo "E1 ckpt: $E1_CKPT"
echo "E2 ckpt: $E2_CKPT"
echo "E3 ckpt: $E3_CKPT"

# ── E1: ResNet-50 baseline ─────────────────────────────────
if [ -n "$E1_CKPT" ]; then
    echo ""
    echo "=== Evaluating E1: ResNet-50 ==="
    python scripts/03_evaluate.py \
        --config configs/E1_resnet_fpn.py \
        --ckpt "$E1_CKPT" \
        --out_json logs/eval_E1.json
else
    echo "[SKIP] E1 checkpoint not found"
fi

# ── E2: CONCH frozen + FPN ─────────────────────────────────
if [ -n "$E2_CKPT" ]; then
    echo ""
    echo "=== Evaluating E2: CONCH+FPN ==="
    python scripts/03_evaluate.py \
        --config configs/E2_conch_fpn.py \
        --ckpt "$E2_CKPT" \
        --out_json logs/eval_E2.json
else
    echo "[SKIP] E2 checkpoint not found"
fi

# ── E3: CONCH + FPN + StainAug ────────────────────────────
if [ -n "$E3_CKPT" ]; then
    echo ""
    echo "=== Evaluating E3: CONCH+FPN+StainAug ==="
    python scripts/03_evaluate.py \
        --config configs/E3_conch_fpn_aug.py \
        --ckpt "$E3_CKPT" \
        --out_json logs/eval_E3.json
else
    echo "[SKIP] E3 checkpoint not found"
fi

echo ""
echo "=============================="
echo "All evaluations done."
echo "=============================="
