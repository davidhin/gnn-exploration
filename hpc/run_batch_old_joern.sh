#!/bin/bash
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=00:10:00
#SBATCH --mem=2GB
#SBATCH --array=1-200
#SBATCH --err="hpc/logs/batch_joern_%a.err"
#SBATCH --output="hpc/logs/batch_joern_%a.out"
#SBATCH --job-name="batch_joern"

# Setup Python Environment
module load Singularity
module load CUDA/10.2.89

# Start singularity instance
singularity run main.simg -p gnnproject/analysis/run_batch_old_joern.py -a $SLURM_ARRAY_TASK_ID