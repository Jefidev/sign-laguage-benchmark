#!/bin/bash
# Submission script for Lucia
#SBATCH --job-name=ViTPose-500
#SBATCH --time=20:00:00 # hh:mm:ss
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres="gpu:1"
#SBATCH --mem-per-cpu=30720 # 30GB
#SBATCH --partition=gpu
#
#SBATCH --mail-user=jerome.fink@unamur.be
#SBATCH --mail-type=ALL
#
#SBATCH --account=lsfb
#
#SBATCH --output=./output/ViTPose_500.out

module purge
module load PyTorch

source ./venv/bin/activate
pip install -r requirements.txt
nvidia-smi
python poseVIT_classification.py \
 -l 500\
 -e Pose-VIT-500\
 -d /gpfs/projects/acad/lsfb/datasets/lsfb_v2/isol