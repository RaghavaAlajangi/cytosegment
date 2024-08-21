#!/bin/bash -l
#SBATCH -o {SLURM_LOGS}/out.%j.log
#SBATCH -e {SLURM_LOGS}/err.%j.log
#SBATCH -D ./
#SBATCH -J {JOB_NAME}
# We need a GPU node
#SBATCH --partition=gpu
#SBATCH --constraint="gpu"
# request one GPU for model training
#SBATCH --gres=gpu:1
# We only need one node
#SBATCH --nodes=1
# Request one CPU per task
# We actually need all the CPUs we can get due to this issue 39 I think:
# https://gitlab.gwdg.de/maximilian.schloegel/dcml/-/issues/39
#SBATCH --cpus-per-task=16
# This is probably redundant, but we only have one task!
#SBATCH --ntasks-per-node=1
# Time required to  train one model
#SBATCH --time={MAX_TIME}
# Memory required per node. Default units are megabytes.
#SBATCH --mem={MAX_MEM}G
#SBATCH --mail-type=all
# Specify your mail address for receiving job notification
#SBATCH --mail-user={MAIL_ID}

# Import modules
module purge
module load anaconda/3/2021.11
module load cuda/11.6
module load cudnn/8.8.1
module load onnx/1.8.1
#Pytorch
module load pytorch/gpu-cuda-11.6/2.0.0
# Activate conda environment
conda activate cytosegment_env
# Train model
srun cytosegment {KWARGS}
