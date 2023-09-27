#!/bin/bash
# Submission script for Lucia
#SBATCH --job-name=download_data
#SBATCH --time=08:00:00 # hh:mm:ss
#
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=4096 # 4GB
#SBATCH --partition=batch
#
#SBATCH --mail-user=jerome.fink@unamur.be
#SBATCH --mail-type=ALL
#
#SBATCH --account=lsfb
#
#SBATCH --output=download_isol.out

module purge
module load PyTorch

source ./venv/bin/activate
python download_isol_lucia.py
