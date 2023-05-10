#!/bin/bash -l
# Standard output and error:
#SBATCH -o out.%j
#SBATCH -e err.%j
# Initial working directory:
#SBATCH -D ./
# Job Name:
#SBATCH -J Tunable_UNet
# Queue:
#SBATCH --partition=gpu     # If using both GPUs of a node
# Node feature:
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:2        # If using both GPUs of a node
# Number of nodes and MPI tasks per node:
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-core=2
#SBATCH --cpus-per-task=36
#SBATCH --mem=100GB
#SBATCH --mail-type=all
#SBATCH --mail-user=raghava.alajangi@mpl.mpg.de
#
# wall clock limit
#SBATCH --time=24:00:00

module purge
module load anaconda/3/2021.11
module load cuda/11.2
#Pytorch
module load pytorch/gpu-cuda-11.2/1.9.0

pip install virtualenv
virtualenv --system-site-packages venv --python=python3.9.7
. venv/bin/activate

pip install albumentations
pip install h5py

echo "starting training .."

srun python -m unet --params_path "params/unet_params.yaml"

echo "Training finished!"
