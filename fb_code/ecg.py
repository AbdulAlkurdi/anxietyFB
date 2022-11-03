import heartpy as hp
import pandas as pd_old
import dask as pd
import utils

# Calculating window statistics for ECGs
def get_window_stats_ecg(data, label=-1):
    wd, m = hp.process(data['ECG'].dropna().reset_index(drop=True), 700)
    return {'bpm': m['bpm'], 'ibi': m['ibi'], 'sdnn': m['sdnn'], 'sdsd': m['sdsd'], 
            'rmssd': m['rmssd'], 'pnn20': m['pnn20'], 'pnn50': m['pnn50']}

def compute_features_chest(ch_data_dict, labels, norm_type=None):
    ecg_df = pd_old.DataFrame(ch_data_dict['ECG'], columns=['ECG'])
    
    # Adding index for combination due to different sampling frequencies
    ecg_df.index = [(1 / utils.fs_dict['ECG']) * i for i in range(len(ecg_df))]
    
    # Change index to datetime
    ecg_df.index = pd_old.to_datetime(ecg_df.index, unit='s')
    
    return ecg_df