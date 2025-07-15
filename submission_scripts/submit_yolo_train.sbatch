#!/bin/bash
#SBATCH --job-name=yolo_train
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --gres=gpu:A6000:1
#SBATCH --time=24:00:00
#SBATCH --output=/home/kdoherty/logs/arthropod/detection/yolo_train_%j.out
#SBATCH --error=/home/kdoherty/logs/arthropod/detection/yolo_train_%j.err

# Activate mamba environment
source ~/.bashrc
mamba activate bugs
cd /home/kdoherty/arthropod-detection

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo ""

# Set environment variables
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Run the training script
echo "Starting YOLOv8 training..."
python 05_yolo_train.py

echo "Training completed at: $(date)"