#!/bin/bash
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=12:30:00
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --array=1-10
#SBATCH --err="hpc/logs/basic_ggnn_%a.info"
#SBATCH --output="hpc/logs/basic_ggnn_%a.info"
#SBATCH --job-name="basic_ggnn_%a"

# Setup Python Environment
module load Singularity
module load CUDA/10.2.89

# Start singularity instance
singularity run --nv main.simg -p gnnproject/analysis/train_ggnn_basic.py -a $SLURM_ARRAY_TASK_ID