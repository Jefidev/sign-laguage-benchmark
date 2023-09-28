#!/bin/bash
# Submission script for Lucia
#SBATCH --job-name=ViTPose-debug
#SBATCH --time=08:00:00 # hh:mm:ss
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
#SBATCH --output=./output/test-run.out

module purge
module load PyTorch

source ./venv/bin/activate
pip install -r requirements.txt
nvidia-smi
python test-run.py \
 -l 250\
 -e test-lucia\
 -d /gpfs/projects/acad/lsfb/datasets/lsfb_v2/isol \
 --dry-run