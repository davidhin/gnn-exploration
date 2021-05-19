#!/bin/bash
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=12:30:00
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --array=1-10
#SBATCH --err="hpc/logs/bgv9_%a.info"
#SBATCH --output="hpc/logs/bgv9_%a.info"
#SBATCH --job-name="bgv9_%a"

# Setup Python Environment
module load Singularity
module load CUDA/10.2.89

# Start singularity instance
singularity exec --nv main.simg python gnnproject/analysis/train_ggnn_basic.py --variation cpg --model devign --batch_size 32