#!/bin/bash
#SBATCH --job-name=fyp_E1
#SBATCH --partition=aiaca40
#SBATCH --qos=4a40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --output=/path/to/FYP_S2/logs/E1_%j.out
#SBATCH --error=/path/to/FYP_S2/logs/E1_%j.err

set -e
FYP=/path/to/FYP_S2
mkdir -p "$FYP/logs"

echo "=== Job $SLURM_JOB_ID: E1 ResNet50+FPN (private dataset) ==="
echo "Start: $(date)"

ml anaconda3
source activate /path/to/miniforge3/envs/fyp_new

python "$FYP/scripts/02_train.py" \
    --config "$FYP/configs/E1_resnet_fpn.py"

echo "End: $(date)"
