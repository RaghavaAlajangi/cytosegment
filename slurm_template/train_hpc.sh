#!/bin/bash -l
#SBATCH -o {{PATH_OUT}}/{{EXP_NAME}}/job_files/out.%j.log
#SBATCH -e {{PATH_OUT}}/{{EXP_NAME}}/job_files/err.%j.log
#SBATCH -D ./
#SBATCH -J {{JOB_NAME}}
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
# Initial tests showed about ~2:30h with bloody_bunny
#SBATCH --time={{MAX_TIME}}
# memory required per node. Default units are megabytes.
#SBATCH --mem={{MAX_MEM}}
#SBATCH --mail-type=all
#SBATCH --mail-user={{MAIL_ID}}


module purge
module load anaconda/3/2021.11
module load cuda/11.2
#Pytorch
module load pytorch/gpu-cuda-11.2/1.9.0

pip install virtualenv

## Remove existing venv (if it exists)
#if [ -d "venv" ]; then
#    rm -rf venv
#fi
# Create and activate a new venv
virtualenv --system-site-packages {{PATH_OUT}}/{{EXP_NAME}}/venv --python=python3.9.7
. {{PATH_OUT}}/{{EXP_NAME}}/venv/bin/activate

pip install albumentations
pip install h5py

srun python -m unet --params_path {{PARAMS_PATH}}
