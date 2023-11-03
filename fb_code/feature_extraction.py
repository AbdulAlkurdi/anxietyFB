import os
import utils
import respiration
import ecg
import pandas as pd_old
import dask as pd
import numpy as np
import neurokit2 as nk
import pickle
from datetime import datetime


WINDOW_IN_SECONDS = 60
stride = 0.25
label_dict = {'baseline': 1, 'stress': 2, 'amusement': 0}
int_to_label = {1: 'baseline', 2: 'stress', 0: 'amusement'}
feat_names = None
savePath = 'data'
subject_feature_path = '/WESAD/subject_feats'

# Make file paths if they don't exist
if not os.path.exists(savePath):
    os.makedirs(savePath)
if not os.path.exists(savePath + subject_feature_path):
    os.makedirs(savePath + subject_feature_path)

class SubjectData:
    """
    Class: SubjectData
    /*
    This class is used to
    */
    """

    def __init__(self, main_path, subject_number):
        '''initializes data for a given subject
    
        Parameters
        ----------
        main_path : string
            path under which all subject folders reside

        subject_number : int
            number of the subject we are initializing

        '''

        self.name = f'S{subject_number}'
        self.subject_keys = ['signal', 'label', 'subject']
        self.signal_keys = ['chest', 'wrist']
        self.chest_keys = ['ACC', 'ECG', 'EMG', 'EDA', 'Temp', 'Resp']
        self.wrist_keys = ['ACC', 'BVP', 'EDA', 'TEMP']
        with open(os.path.join(main_path, self.name) + '/' + self.name + '.pkl', 'rb') as file:
            self.data = pickle.load(file, encoding='latin1')
        self.labels = self.data['label']

    def get_wrist_and_chest_data(self):
        '''extracts wrist data from data dictionary

        Function that extracts wrist data from the data dictionary, which
        contains the physiological signals that have been extracted from the
        .pkl subject file.
        
        Parameters
        ----------

        Returns
        -------
        data : dict
            wrist data

        '''

        data = self.data['signal']['wrist']
        data.update({'ACC_C': self.data['signal']['chest']['ACC'],
                     'ECG': self.data['signal']['chest']['ECG'],
                     'EDA_C': self.data['signal']['chest']['EDA'],
                     'EMG_C': self.data['signal']['chest']['EMG'],
                     'Resp_C': self.data['signal']['chest']['Resp'],
                     'Temp_C': self.data['signal']['chest']['Temp']})
        return data

# Computes features for wrist
def compute_features(e4_data_dict, labels, norm_type=None):
    '''
    This should be run in fb_code environemnt and not in TFwesad
    '''

    # Dataframes for each sensor type
    eda_df = pd_old.DataFrame(e4_data_dict['EDA'], columns=['EDA'])
    bvp_df = pd_old.DataFrame(e4_data_dict['BVP'], columns=['BVP'])
    acc_df = pd_old.DataFrame(e4_data_dict['ACC'], columns=['ACC_x', 'ACC_y', 'ACC_z'])
    temp_df = pd_old.DataFrame(e4_data_dict['TEMP'], columns=['TEMP'])
    label_df = pd_old.DataFrame(labels, columns=['label'])
    resp_df = pd_old.DataFrame(e4_data_dict['Resp_C'], columns=['Resp_C'])
    acc_c_df = pd_old.DataFrame(e4_data_dict['ACC_C'], columns=['ACC_x_C', 'ACC_y_C', 'ACC_z_C'])
    ecg_c_df = pd_old.DataFrame(e4_data_dict['ECG'], columns=['ECG'])
    #eda_c_df = pd_old.DataFrame(e4_data_dict['EDA_C'], columns=['EDA_C'])
    #emg_c_df = pd_old.DataFrame(e4_data_dict['EMG_C'], columns=['EMG_C'])
    resp_c_df = pd_old.DataFrame(e4_data_dict['Resp_C'], columns=['Resp_C'])
    #temp_c_df = pd_old.DataFrame(e4_data_dict['Temp_C'], columns=['Temp_C'])

    # Filter EDA
    eda_df['EDA'] = utils.butter_lowpass_filter(eda_df['EDA'], 1.0, utils.fs_dict['EDA'], 6)
    #eda_c_df['EDA_C'] = utils.butter_lowpass_filter(eda_c_df['EDA_C'], 1.0, utils.fs_dict['chest'], 6)
    eda_data = nk.eda_phasic(nk.standardize(eda_df['EDA']), sampling_rate=utils.fs_dict['EDA'])
    eda_df['EDA_SCR'] = eda_data['EDA_Phasic']
    eda_df['EDA_SCL'] = eda_data['EDA_Tonic']
    #eda_data_c = nk.eda_phasic(nk.standardize(eda_c_df['EDA_C']), sampling_rate=utils.fs_dict['chest'])
    #eda_c_df['EDA_SCR_C'] = eda_data_c['EDA_Phasic']
    #eda_c_df['EDA_SCL_C'] = eda_data_c['EDA_Tonic']
    
    # Filter ACM
    for _ in acc_df.columns:
        acc_df[_] = utils.filterSignalFIR(acc_df.values)
    for _ in acc_c_df.columns:
        acc_c_df[_] = utils.filterSignalFIR(acc_c_df.values)

    # Adding indices for combination due to differing sampling frequencies
    eda_df.index = [(1 / utils.fs_dict['EDA']) * i for i in range(len(eda_df))]
    bvp_df.index = [(1 / utils.fs_dict['BVP']) * i for i in range(len(bvp_df))]
    acc_df.index = [(1 / utils.fs_dict['ACC']) * i for i in range(len(acc_df))]
    temp_df.index = [(1 / utils.fs_dict['TEMP']) * i for i in range(len(temp_df))]
    label_df.index = [(1 / utils.fs_dict['label']) * i for i in range(len(label_df))]
    resp_df.index = [(1 / utils.fs_dict['Resp']) * i for i in range(len(resp_df))]
    acc_c_df.index = [(1 / utils.fs_dict['chest']) * i for i in range(len(acc_c_df))]
    ecg_c_df.index = [(1 / utils.fs_dict['chest']) * i for i in range(len(ecg_c_df))]
    #eda_c_df.index = [(1 / utils.fs_dict['chest']) * i for i in range(len(eda_c_df))]
    #emg_c_df.index = [(1 / utils.fs_dict['chest']) * i for i in range(len(emg_c_df))]
    resp_c_df.index = [(1 / utils.fs_dict['chest']) * i for i in range(len(resp_c_df))]
    #temp_c_df.index = [(1 / utils.fs_dict['chest']) * i for i in range(len(temp_c_df))]

    # Change indices to datetime
    eda_df.index = pd_old.to_datetime(eda_df.index, unit='s')
    bvp_df.index = pd_old.to_datetime(bvp_df.index, unit='s')
    temp_df.index = pd_old.to_datetime(temp_df.index, unit='s')
    acc_df.index = pd_old.to_datetime(acc_df.index, unit='s')
    label_df.index = pd_old.to_datetime(label_df.index, unit='s')
    resp_df.index = pd_old.to_datetime(resp_df.index, unit='s')
    acc_c_df.index = pd_old.to_datetime(acc_c_df.index, unit='s')
    ecg_c_df.index = pd_old.to_datetime(ecg_c_df.index, unit='s')
    #eda_c_df.index = pd_old.to_datetime(eda_c_df.index, unit='s')
    #emg_c_df.index = pd_old.to_datetime(emg_c_df.index, unit='s')
    resp_c_df.index = pd_old.to_datetime(resp_c_df.index, unit='s')
    #temp_c_df.index = pd_old.to_datetime(temp_c_df.index, unit='s')

    # Getting ECG features
    ecg_df = ecg.get_ecg_data(e4_data_dict, norm_type=None)
        
    # Combined dataframe
    df = eda_df.join(bvp_df, how='outer')
    df = df.join(temp_df, how='outer')
    df = df.join(acc_df, how='outer')
    df = df.join(label_df, how='outer')
    df = df.join(ecg_df, how='outer')
    #df = df.join(eda_c_df, how='outer')
    df = df.join(acc_c_df, how='outer')
    #df = df.join(emg_c_df, how='outer')
    df = df.join(resp_c_df, how='outer')
    #df = df.join(temp_c_df, how='outer')
    df['label'] = df['label'].fillna(method='bfill')
    df.reset_index(drop=True, inplace=True)

    if norm_type == 'std':
        # std norm
        df = (df - df.mean()) / df.std()
    elif norm_type == 'minmax':
        # minmax norm
        df = (df - df.min()) / (df.max() - df.min())

    # Groupby
    grouped = df.groupby('label')
    baseline = grouped.get_group(1)
    stress = grouped.get_group(2)
    amusement = grouped.get_group(3)
    return grouped, baseline, stress, amusement


def get_samples(data, n_windows, label):
    global feat_names
    global WINDOW_IN_SECONDS
    
    samples = []
    # Using label freq (700 Hz) as our reference frequency due to it being the largest
    # and thus encompassing the lesser ones in its resolution.
    window_len = utils.fs_dict['label'] * WINDOW_IN_SECONDS

    for i in range(n_windows):
        # Get window of data
        w = data[window_len * i: window_len * (i + 1)]

        # Add/Calc rms acc
        w = pd_old.concat([utils.get_net_accel(w, 'wrist'), w])
        cols = list(w.columns)
        cols[0] = 'net_acc'
        w.columns = cols
        w = pd_old.concat([utils.get_net_accel(w, 'chest'), w])
        cols = list(w.columns)
        cols[0] = 'net_acc_C'
        w.columns = cols
        
        # Calculate stats for window
        wstats = utils.get_window_stats(data=w, label=label)
        
        # Calculate stats for window (ECG)
        wstats_ecg = ecg.get_window_stats_ecg(data=w, label=label)
        
        # Seperating sample and label
        x = pd_old.DataFrame(wstats).drop('label', axis=0)
        y = x['label'][0]
        x.drop('label', axis=1, inplace=True)
        
        feat_names = None
        if feat_names is None:
            feat_names = []
            for row in x.index:
                for col in x.columns:
                    feat_names.append('_'.join([str(row), str(col)]))

        # Populate sample df
        wdf = pd_old.DataFrame(x.values.flatten()).T
        wdf.columns = feat_names
        wdf = pd_old.concat([wdf, pd_old.DataFrame({'label': y}, index=[0])], axis=1)
        
        # Add BVP feature
        wdf['BVP_peak_freq'] = utils.get_peak_freq(w['BVP'].dropna())
        
        # Add more features here: ACC (w and c)
        wdf['net_acc_abs_integral'] = utils.get_absolute_integral(w['net_acc'].dropna())
        wdf['ACC_x_abs_integral'] = utils.get_absolute_integral(w['ACC_x'].dropna())
        wdf['ACC_y_abs_integral'] = utils.get_absolute_integral(w['ACC_y'].dropna())
        wdf['ACC_z_abs_integral'] = utils.get_absolute_integral(w['ACC_z'].dropna())
        wdf['net_acc_C_abs_integral'] = utils.get_absolute_integral(w['net_acc_C'].dropna())
        wdf['ACC_x_C_abs_integral'] = utils.get_absolute_integral(w['ACC_x_C'].dropna())
        wdf['ACC_y_C_abs_integral'] = utils.get_absolute_integral(w['ACC_y_C'].dropna())
        wdf['ACC_z_C_abs_integral'] = utils.get_absolute_integral(w['ACC_z_C'].dropna())
        wdf['ACC_x_peak_freq'] = utils.get_peak_freq(w['ACC_x'].dropna())
        wdf['ACC_y_peak_freq'] = utils.get_peak_freq(w['ACC_y'].dropna())
        wdf['ACC_z_peak_freq'] = utils.get_peak_freq(w['ACC_z'].dropna())
        wdf['ACC_x_C_peak_freq'] = None if len(w['ACC_x_C'].dropna()) == 0 else utils.get_peak_freq(w['ACC_x_C'].dropna())
        wdf['ACC_y_C_peak_freq'] = None if len(w['ACC_y_C'].dropna()) == 0 else utils.get_peak_freq(w['ACC_y_C'].dropna())
        wdf['ACC_z_C_peak_freq'] = None if len(w['ACC_z_C'].dropna()) == 0 else utils.get_peak_freq(w['ACC_z_C'].dropna())
        
        # ECG
        for key in wstats_ecg.keys():
            wdf['ECG_'+key] = wstats_ecg[key]
        
        # EDA(w and c)
        wdf['EDA_slope'] = utils.get_slope(w['EDA'].dropna())
        wdf['EDA_C_slope'] = None if len(w['ACC_z_C'].dropna()) == 0 else utils.get_slope(w['EDA_C'].dropna())
        wdf['EDA_drange'] = utils.get_dynamic_range(w['EDA'].dropna())
        wdf['EDA_C_drange'] = None if len(w['EDA_C'].dropna()) == 0 else utils.get_dynamic_range(w['EDA_C'].dropna())

        # EMG(c)
        wdf['EMG_drange'] = utils.get_dynamic_range(w['EMG_C'].dropna())
        wdf['EMG_abs_integral'] = utils.get_absolute_integral(w['EMG_C'].dropna())

        # RESP(c)
        if len(w['Resp_C'].dropna()) > 0:
            wdf['Resp_C_rate'], wdf['Resp_C_Inhal_mean'], wdf['Resp_C_Inhal_std'], wdf['Resp_C_Exhal_mean'], wdf['Resp_C_Exhal_std'], wdf['Resp_C_I/E'] = respiration.get_resp_features(w['Resp_C'].dropna())

        # TEMP(w and c)
        wdf['TEMP_drange'] = utils.get_dynamic_range(w['TEMP'].dropna())
        wdf['TEMP_C_drange'] = None if len(w['Temp_C'].dropna()) == 0 else utils.get_dynamic_range(w['Temp_C'].dropna())
        wdf['TEMP_slope'] = utils.get_slope(w['TEMP'].dropna())
        wdf['TEMP_C_slope'] = None if len(w['Temp_C'].dropna()) == 0 else utils.get_slope(w['Temp_C'].dropna())
        
        samples.append(wdf)

    return pd_old.concat(samples)

def make_patient_data(subject_id):
    global savePath
    global WINDOW_IN_SECONDS

    # Make subject data object for Sx
    subject = SubjectData(main_path='data/WESAD', subject_number=subject_id)

    # Empatica E4 data - now with resp
    e4_data_dict = subject.get_wrist_and_chest_data()
    
    # norm type
    norm_type = None

    # The 3 classes we are classifying
    grouped, baseline, stress, amusement = compute_features(e4_data_dict, subject.labels, norm_type)

    # Get windows
    #n_baseline_wdws = int(len(baseline) / (utils.fs_dict['label'] * WINDOW_IN_SECONDS)) # these windows have no overlap
    #n_stress_wdws = int(len(stress) / (utils.fs_dict['label'] * WINDOW_IN_SECONDS)) # these windows have no overlap
    #n_amusement_wdws = int(len(amusement) / (utils.fs_dict['label'] * WINDOW_IN_SECONDS)) # these windows have no overlap
    n_baseline_wdws = len(range(0,len(baseline) - WINDOW_IN_SECONDS*64+1,int(stride*64)))
    n_stress_wdws = len(range(0,len(stress) - WINDOW_IN_SECONDS*64+1,int(stride*64)))
    n_amusement_wdws = len(range(0,len(amusement) - WINDOW_IN_SECONDS*64+1,int(stride*64)))

    # Get samples
    baseline_samples = get_samples(baseline, n_baseline_wdws, 1)
    stress_samples = get_samples(stress, n_stress_wdws, 2)
    amusement_samples = get_samples(amusement, n_amusement_wdws, 0)

    all_samples = pd_old.concat([baseline_samples, stress_samples, amusement_samples])
    all_samples = pd_old.concat([all_samples.drop('label', axis=1), pd_old.get_dummies(all_samples['label'])], axis=1)
    # Save file as csv
    all_samples.to_csv(f'{savePath}{subject_feature_path}/S{subject_id}_feats.csv')

    subject = None

def combine_files(subjects):
    df_list = []
    for s in subjects:
        df = pd_old.read_csv(f'{savePath}{subject_feature_path}/S{s}_feats.csv', index_col=0)
        df['subject'] = s
        df_list.append(df)

    df = pd_old.concat(df_list)

    df['label'] = (df['0'].astype(str) + df['1'].astype(str) + df['2'].astype(str)).apply(lambda x: x.index('1'))
    df.drop(['0', '1', '2'], axis=1, inplace=True)

    df.reset_index(drop=True, inplace=True)

    #df.to_csv(f'{savePath}/may14_feats4.csv')
    now = datetime.today().strftime('%Y-%m-%d--%H-%M-%p')
    df.to_csv(f'{savePath}/WESAD/subject_feats/{now}_feats.csv')

    counts = df['label'].value_counts()
    print('Number of samples per class:')
    for label, number in zip(counts.index, counts.values):
        print(f'{int_to_label[label]}: {number}')


if __name__ == '__main__':

    subject_ids = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]

    for patient in subject_ids:
        print(f'Processing data for S{patient}...')
        make_patient_data(patient)

    combine_files(subject_ids)
    print('Processing complete.')
