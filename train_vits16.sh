#!/bin/bash

#SBATCH --job-name=dinov3_vits16
#SBATCH --output=/home/hd/hd_hd/hd_fk313/logs/dinov3_vits16_%j.out
#SBATCH --error=/home/hd/hd_hd/hd_fk313/logs/dinov3_vits16_%j.err
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=02:00:00

mkdir -p /home/hd/hd_hd/hd_fk313/logs

echo "=============================="
echo "Job started on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "=============================="

# Load CUDA
module purge
module load devel/cuda/12.8

# Activate conda
source ~/.bashrc
conda activate dinov3

echo "Using Python: $(which python)"

python -c "import torch; print('Torch:', torch.__version__)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# Paths
REPO_DIR="/home/hd/hd_hd/hd_fk313/thesis/dinov3"
DATA_DIR="/home/hd/hd_hd/hd_fk313/patches_img_normal_tumor"
WEIGHTS="/home/hd/hd_hd/hd_fk313/thesis/dinov3/dinov3_vits16_pretrain.pth"
OUTPUT_DIR="/home/hd/hd_hd/hd_fk313/outputs/dinov3_vits16_${SLURM_JOB_ID}"

mkdir -p "$OUTPUT_DIR"
echo "Output directory: $OUTPUT_DIR"

cd "$REPO_DIR" || exit 1

echo "Repository:"
pwd

echo "Dataset:"
ls "$DATA_DIR" | head

echo "Starting DINOv3 ViT-S/16 Linear Probing..."
echo "Weights: $WEIGHTS"

python train_linear_probe.py \
  --data-dir "$DATA_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --repo-dir "$REPO_DIR" \
  --weights "$WEIGHTS" \
  --model-arch vits16 \
  --batch-size 32 \
  --epochs 2 \
  --lr 0.001 \
  --val-split 0.2 \
  --num-workers 4 \
  --seed 42

echo "Training finished."
