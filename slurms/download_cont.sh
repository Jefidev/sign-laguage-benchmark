#!/bin/bash
# Submission script for Lucia
#SBATCH --job-name=download_data
#SBATCH --time=08:00:00 # hh:mm:ss
#
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=8192 # 4GB
#SBATCH --partition=batch
#
#SBATCH --mail-user=jerome.fink@unamur.be
#SBATCH --mail-type=ALL
#
#SBATCH --account=lsfb
#
#SBATCH --output=download_cont.out

module purge
module load PyTorch

source ./venv/bin/activate
python download_cont_lucia.py
