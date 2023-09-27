#!/bin/bash
# Submission script for Lucia
#SBATCH --job-name=ViTPose
#SBATCH --time=08:00:00 # hh:mm:ss
#
#SBATCH --ntasks=1
#SBATCH --gres="gpu:1"
#SBATCH --mem-per-cpu=30720 # 30GB
#SBATCH --partition=gpu
#
#SBATCH --mail-user=jerome.fink@unamur.be
#SBATCH --mail-type=ALL
#
#SBATCH --account=lsfb
#
#SBATCH --output=ViTPose_250.out

module purge
module load PyTorch

source ./venv/bin/activate
nvidia-smi
python poseVIT_classification.py \
 -l 250\
 -e Pose-VIT-250\
 -d /gpfs/projects/acad/lsfb/datasets/lsfb_v2/isol