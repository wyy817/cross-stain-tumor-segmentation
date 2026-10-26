#!/bin/bash
#SBATCH --job-name=fyp_E3
#SBATCH --partition=aiaca40
#SBATCH --qos=4a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/path/to/FYP_S2/logs/E3_%j.out
#SBATCH --error=/path/to/FYP_S2/logs/E3_%j.err

set -e
FYP=/path/to/FYP_S2
mkdir -p "$FYP/logs"

echo "=== Job $SLURM_JOB_ID: E3 CONCH frozen+FPN+StainAug (private dataset) ==="
echo "Start: $(date)"

conda run -n fyp_new --no-capture-output \
    python "$FYP/scripts/02_train.py" \
    --config "$FYP/configs/E3_conch_fpn_aug.py"

echo "End: $(date)"
