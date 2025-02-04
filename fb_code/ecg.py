import heartpy as hp
import pandas as pd_old
import dask as pd
import utils

def get_window_stats_ecg(data, label=-1, norm_type=None):
    '''extracts features from ecgs

    Function that scales passed data so that it has specified lower 
    and upper bounds.
    
    Parameters
    ----------
    data : dict
        Physiological signal dictionary

    Returns
    -------
    out : dict
        Contains ECG features like BPM, HRV, etc.
    '''
    wd, m = hp.process(data['ECG'].dropna().reset_index(drop=True), 700)
    return {'bpm': m['bpm'], 'ibi': m['ibi'], 'sdnn': m['sdnn'], 'sdsd': m['sdsd'], 
            'rmssd': m['rmssd'], 'pnn20': m['pnn20'], 'pnn50': m['pnn50']}

def get_ecg_data(e4_data_dict, norm_type=None):
    '''returns normalized ecg data from physiological signal dict

    Function that extracts ecg data from the physiological signal dict and
    normalizes the signal based on the sampling frequency of the ECG.
    
    Parameters
    ----------
    e4_data_dict : dict
        Physiological signal dictionary

    Returns
    -------
    ecg_df : pandas dataframe
        Pandas dataframe with ecg data normalized by sampling frequency
    '''
    # Convert ecg data into dictionary
    ecg_df = pd_old.DataFrame(e4_data_dict['ECG'], columns=['ECG'])
    
    # Adding index for combination due to different sampling frequencies
    ecg_df.index = [(1 / utils.fs_dict['ECG']) * i for i in range(len(ecg_df))]
    
    # Change index to datetime
    ecg_df.index = pd_old.to_datetime(ecg_df.index, unit='s')
    
    return ecg_df