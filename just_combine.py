import pandas as pd
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import logging
now = datetime.now().strftime('%Y-%m-%d')
import time

loadPath = 'data/WESAD'
savePath = 'data/GN-WESAD'
subject_feature_path =  '/subject_feats'

snrs = [ 0.0001,  0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6] 
subject_ids = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
n_samples = 10 

def process_subject_file(snr, n_i, s):
    file_path = f'{savePath}/n_{n_i}/snr_{snr}{subject_feature_path}/S{s}_feats.csv'
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, index_col=0)
        df['subject'] = s
        return df
    return None

def combine_noiZ_files(subjects):
    now = datetime.now().strftime('%Y-%m-%d')
    logging.basicConfig(level=logging.INFO, filename=now+'-combine.log', filemode='w', force=True)

    for snr in snrs:
        for n_i in range(n_samples):
            # Parallelize file reading
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(process_subject_file, snr, n_i, s) for s in subjects]
                df_list = [future.result() for future in futures if future.result() is not None]

            if df_list:
                df = pd.concat(df_list)
                df['label'] = df[['0', '1', '2']].idxmax(axis=1)
                df.drop(['0', '1', '2'], axis=1, inplace=True)
                df.reset_index(drop=True, inplace=True)
                
                df.to_csv(f'{savePath}/n_{n_i}/snr_{snr}{subject_feature_path}/{now}_feats_filt.csv')
                logging.info('-' * 20)
                logging.info(f'Saved file to: {savePath}/n_{n_i}/snr_{snr}{subject_feature_path}/{now}_feats_filt.csv')
                
                counts = df['label'].value_counts()
                logging.info('Number of samples per class:')
                ('baseline: {0[1]}; stress: {1[1]}; amusement: {2[1]} '.format(*list(zip(counts.index, counts.values))))
    logging.info('all done!')


combine_noiZ_files(subject_ids)
# took 19m 30s to run