

for snr in SNRS = .01 .05 .3 ; do #.1 .2 .4, i can potentially add more
    sbatch gn_e2ejob.sbatch $subject $snr $n_samples $WINDOW_IN_SECONDS $stride 
    sleep 0.25 # pause to be nice to the scheduler 
done;

