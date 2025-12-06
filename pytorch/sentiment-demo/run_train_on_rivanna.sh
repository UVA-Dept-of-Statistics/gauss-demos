#!/bin/bash

#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:2
#SBATCH --partition=gpu
#SBATCH -J "train_roberta_variant"
#SBATCH --mail-user=trb5me@virginia.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --account=ds-6040

module purge
module load miniforge/24.3.0-py3.11

ENV="$HOME/.conda/envs/pytorch_rivanna"
PY="$ENV/bin/python"

echo "=== DEBUG ==="
echo "Using python: $PY"
$PY - << 'EOF'
import sys, torch
print("python:", sys.executable)
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
EOF
echo "=============="

$PY train.py
