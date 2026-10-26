#!/bin/bash
# ============================================================
# SLURM job script for FYP training
# Usage:
#   sbatch scripts/train.sbatch E2_conch_fpn
#   sbatch scripts/train.sbatch E3_conch_fpn_aug
#   sbatch scripts/train.sbatch E1_resnet_fpn
# ============================================================

#SBATCH --job-name=fyp_${1:-E2}
#SBATCH --output=/path/to/FYP_S2/logs/slurm_%x_%j.out
#SBATCH --error=/path/to/FYP_S2/logs/slurm_%x_%j.err
#SBATCH --partition=a40          # ← 改成你的实际分区名 (sinfo 查看)
#SBATCH --gres=gpu:1             # 1 块 GPU
#SBATCH --cpus-per-task=5        # 4 workers + 1 main process
#SBATCH --mem=32G
#SBATCH --time=12:00:00

# ── Paths ────────────────────────────────────────────────────
FYP_DIR=/path/to/FYP_S2
CONDA_BASE=/path/to/miniforge3

# ── Config from argument ──────────────────────────────────────
EXPERIMENT=${1:-E2_conch_fpn}
CONFIG=$FYP_DIR/configs/${EXPERIMENT}.py

echo "=============================="
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURMD_NODENAME"
echo "GPU:        $CUDA_VISIBLE_DEVICES"
echo "Experiment: $EXPERIMENT"
echo "Config:     $CONFIG"
echo "=============================="

# ── Activate conda ────────────────────────────────────────────
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate fyp_new

# ── Create log dir ────────────────────────────────────────────
mkdir -p $FYP_DIR/logs

# ── Run training ──────────────────────────────────────────────
cd $FYP_DIR
export PYTHONPATH=$FYP_DIR

python scripts/02_train.py --config $CONFIG

echo "=============================="
echo "Training finished: $EXPERIMENT"
echo "=============================="
