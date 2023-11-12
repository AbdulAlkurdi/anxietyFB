#!/bin/bash -l
$ cat job.slurm
#SBATCH --mem=32g
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=3  
#SBATCH --partition=cpu
#SBATCH --account=bbnp-delta-cpu
#SBATCH --job-name=wesad_gaussian_filtering_ecg
#SBATCH --time=48:00:00      # hh:mm:ss for the job
#SBATCH --constraint="scratch"
#SBATCH --gpus-per-node=0
#SBATCH -output="job_logs/job_%x_%j.out"
#SBATCH --mail-user=alkurdi2@illinois.edu
#SBATCH --mail-type="END" 
 
module list
conda activate fb_code

echo "job is starting on `hostname`"
cd /projects/bbnp/fb_code/

subject=$1
snr=$2
n_samples=$3
WINDOW_IN_SECONDS=$4
stride=$5

echo " "
echo " **************************************************** "
echo " *** subject=$subject				*** "
echo " *** snr=$snr					*** "
echo " *** n_samples=$n_samples				*** "
echo " *** WINDOW_IN_SECONDS=$WINDOW_IN_SECONDS		*** "
echo " *** stride=$stride				*** "
echo " **************************************************** "
echo " "
date

srun python noise-multiparallel.py $subject $snr $n_samples $WINDOW_IN_SECONDS $stride


echo "Done"
date

