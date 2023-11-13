#!/bin/bash
#SBATCH --mem=32g
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --partition=cpu
#SBATCH --account=bbnp-delta-cpu
#SBATCH --job-name=wesad_gaussian_filtering_ecg_combining
#SBATCH --time=48:00:00
#SBATCH --constraint="scratch"
#SBATCH -o ./job_logs/job_%x_%j.out
#SBATCH --mail-user=alkurdi2@illinois.edu
#SBATCH --mail-type="END" 
module list
# ACTIVATE ANACONDA
eval "$(conda shell.bash hook)"
conda activate fb_code
echo "current conda environment is "$CONDA_DEFAULT_ENV
echo "job is starting on `hostname`"
cd /projects/bbnp/Vansh/
echo " "
echo " ********************** "
echo " *** combing files  *** "
echo " ********************** "
echo " "
date
srun python just_combine.py
echo "Done"
date