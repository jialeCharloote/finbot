#!/bin/bash

#SBATCH --job-name=finetune222
#SBATCH --output=finetune2.txt
#SBATCH --error=finetune-error2.txt
#SBATCH --time=36:00:00
#SBATCH --mem=64G   
#SBATCH --ntasks-per-node=48
##SBATCH --cpus-per-task=1
#SBATCH --account=pi-dachxiu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=v100   # constraint job runs on a100 GPU use
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jialez@rcc.uchicago.edu


# Load modules
module load cuda/11.7

# Initialize conda
eval "$(conda shell.bash hook)"
conda activate pytorch


# Either install CuPy or modify the script
# pip install cupy-cuda117  # Uncomment if you want to install CuPy

# Run the script
python -u /home/jialez/finetunemodel/final.py