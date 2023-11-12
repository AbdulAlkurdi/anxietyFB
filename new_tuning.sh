#/bin/bash
stride = 10
WINDOW_IN_SECONDS = 60
n_samples = 10

for subject in 2 3 4 5 6 7 8 9 10 11 13 14 15 16 17; do

    for snr in 0.0001 0.001 0.01 0.05 0.1 0.15 0.2 0.3 0.4 0.5 0.6; do
        #echo job.sbatch $subject $snr $n_samples $WINDOW_IN_SECONDS $stride 
        sbatch job.sbatch $subject $snr $n_samples $WINDOW_IN_SECONDS $stride 
    done;
done;