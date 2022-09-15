import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import find_peaks
import scipy.signal as scisig
import scipy.stats
import biosppy
import neurokit2 as nk
#import cvxEDA

# E4 (wrist) Sampling Frequencies
fs_dict = {'ACC': 32, 'BVP': 64, 'EDA': 4, 'TEMP': 4, 'label': 700, 'chest': 700}
WINDOW_IN_SECONDS = 30
label_dict = {'baseline': 1, 'stress': 2, 'amusement': 0}
int_to_label = {1: 'baseline', 2: 'stress', 0: 'amusement'}
feat_names = None
savePath = 'data'
subject_feature_path = '/subject_feats'

if not os.path.exists(savePath):
    os.makedirs(savePath)
if not os.path.exists(savePath + subject_feature_path):
    os.makedirs(savePath + subject_feature_path)

# cvxEDA
'''
def eda_stats(y):
    Fs = fs_dict['EDA']
    yn = (y - y.mean()) / y.std()
    [r, p, t, l, d, e, obj] = cvxEDA.cvxEDA(yn, 1. / Fs)
    return [r, p, t, l, d, e, obj]
'''

class SubjectData:

    def __init__(self, main_path, subject_number):
        self.name = f'S{subject_number}'
        self.subject_keys = ['signal', 'label', 'subject']
        self.signal_keys = ['chest', 'wrist']
        self.chest_keys = ['ACC', 'ECG', 'EMG', 'EDA', 'Temp', 'Resp']
        self.wrist_keys = ['ACC', 'BVP', 'EDA', 'TEMP']
        with open(os.path.join(main_path, self.name) + '/' + self.name + '.pkl', 'rb') as file:
            self.data = pickle.load(file, encoding='latin1')
        self.labels = self.data['label']

    def get_data(self):
        data = self.data['signal']['wrist']
        data.update({'ACC_C': self.data['signal']['chest']['ACC'],
                     'ECG_C': self.data['signal']['chest']['ECG'],
                     'EDA_C': self.data['signal']['chest']['EDA'],
                     'EMG_C': self.data['signal']['chest']['EMG'],
                     'Resp_C': self.data['signal']['chest']['Resp'],
                     'Temp_C': self.data['signal']['chest']['Temp']
                     })
        return data

    '''
    def get_chest_data(self):
        return self.data['signal']['chest']
        return data

    def extract_features(self):  # only wrist
        results = \
            {
                key: get_statistics(self.get_wrist_data()[key].flatten(), self.labels, key)
                for key in self.wrist_keys
            }
        return results
    '''


# https://github.com/MITMediaLabAffectiveComputing/eda-explorer/blob/master/load_files.py
def butter_lowpass(cutoff, fs, order=5):
    # Filtering Helper functions
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = scisig.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    # Filtering Helper functions
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = scisig.lfilter(b, a, data)
    return y

def get_slope(series):
    linreg = scipy.stats.linregress(np.arange(len(series)), series )
    slope = linreg[0]
    return slope

def get_window_stats(data, label=-1):
    mean_features = np.mean(data)
    std_features = np.std(data)
    min_features = np.amin(data)
    max_features = np.amax(data)

    features = {'mean': mean_features, 'std': std_features, 'min': min_features, 'max': max_features,
                'label': label}
    return features


def get_net_accel(data):
    return (data['ACC_x'] ** 2 + data['ACC_y'] ** 2 + data['ACC_z'] ** 2).apply(lambda x: np.sqrt(x))

def get_net_accel_C(data):
    return (data['ACC_x_C'] ** 2 + data['ACC_y_C'] ** 2 + data['ACC_z_C'] ** 2).apply(lambda x: np.sqrt(x))

def get_peak_freq(x):
    f, Pxx = scisig.periodogram(x, fs=8)
    psd_dict = {amp: freq for amp, freq in zip(Pxx, f)}
    peak_freq = psd_dict[max(psd_dict.keys())]
    return peak_freq

def get_absolute_integral(x):
    return np.sum(np.abs(x))

def get_dynamic_range(x):
    return np.max(x) / np.min(x)

def get_resp_features(resp_data):
    resp_rate, filtered, zeros, resp_rate_ts, resp_rate = biosppy.signals.resp.resp(resp_data, sampling_rate=700, show=False)
    extremas, values = biosppy.signals.tools.find_extrema(signal=filtered, mode='both')
    inhal_durations = []
    exhal_durations = []
    last_index = 0
    for i in range(len(extremas)):
        if values[i] * values[last_index] < 0:
            if values[last_index] < 0:
                inhal_durations.append((extremas[i] - extremas[last_index]) / 700)
            else:
                exhal_durations.append((extremas[i] - extremas[last_index]) / 700)
            last_index = i
    return np.mean(resp_rate), np.mean(inhal_durations), np.std(inhal_durations), np.mean(exhal_durations), np.std(exhal_durations), np.sum(inhal_durations) / np.sum(exhal_durations)
    

        
# https://github.com/MITMediaLabAffectiveComputing/eda-explorer/blob/master/AccelerometerFeatureExtractionScript.py
def filterSignalFIR(eda, cutoff=0.4, numtaps=64):
    f = cutoff / (fs_dict['ACC'] / 2.0)
    FIR_coeff = scisig.firwin(numtaps, f)

    return scisig.lfilter(FIR_coeff, 1, eda)


def compute_features(data_dict, labels, norm_type=None):
    # Dataframes for each sensor type
    eda_df = pd.DataFrame(data_dict['EDA'], columns=['EDA'])
    bvp_df = pd.DataFrame(data_dict['BVP'], columns=['BVP'])
    acc_df = pd.DataFrame(data_dict['ACC'], columns=['ACC_x', 'ACC_y', 'ACC_z'])
    temp_df = pd.DataFrame(data_dict['TEMP'], columns=['TEMP'])
    label_df = pd.DataFrame(labels, columns=['label'])

    acc_c_df = pd.DataFrame(data_dict['ACC_C'], columns=['ACC_x_C', 'ACC_y_C', 'ACC_z_C'])
    ecg_c_df = pd.DataFrame(data_dict['ECG_C'], columns=['ECG_C'])
    eda_c_df = pd.DataFrame(data_dict['EDA_C'], columns=['EDA_C'])
    emg_c_df = pd.DataFrame(data_dict['EMG_C'], columns=['EMG_C'])
    resp_c_df = pd.DataFrame(data_dict['Resp_C'], columns=['Resp_C'])
    temp_c_df = pd.DataFrame(data_dict['Temp_C'], columns=['Temp_C'])


    # Filter EDA
    eda_df['EDA'] = butter_lowpass_filter(eda_df['EDA'], 1.0, fs_dict['EDA'], 6)
    eda_c_df['EDA_C'] = butter_lowpass_filter(eda_c_df['EDA_C'], 1.0, fs_dict['chest'], 6)

    eda_data = nk.eda_phasic(nk.standardize(eda_df['EDA']), sampling_rate=fs_dict['EDA'])
    eda_df['EDA_SCR'] = eda_data['EDA_Phasic']
    eda_df['EDA_SCL'] = eda_data['EDA_Tonic']
    eda_data_c = nk.eda_phasic(nk.standardize(eda_c_df['EDA_C']), sampling_rate=fs_dict['chest'])
    eda_c_df['EDA_SCR_C'] = eda_data_c['EDA_Phasic']
    eda_c_df['EDA_SCL_C'] = eda_data_c['EDA_Tonic']
    
    # Filter ACM
    for _ in acc_df.columns:
        acc_df[_] = filterSignalFIR(acc_df.values)
    for _ in acc_c_df.columns:
        acc_c_df[_] = filterSignalFIR(acc_c_df.values)

    # Adding indices for combination due to differing sampling frequencies
    eda_df.index = [(1 / fs_dict['EDA']) * i for i in range(len(eda_df))]
    bvp_df.index = [(1 / fs_dict['BVP']) * i for i in range(len(bvp_df))]
    acc_df.index = [(1 / fs_dict['ACC']) * i for i in range(len(acc_df))]
    temp_df.index = [(1 / fs_dict['TEMP']) * i for i in range(len(temp_df))]
    label_df.index = [(1 / fs_dict['label']) * i for i in range(len(label_df))]
    acc_c_df.index = [(1 / fs_dict['chest']) * i for i in range(len(acc_c_df))]
    ecg_c_df.index = [(1 / fs_dict['chest']) * i for i in range(len(ecg_c_df))]
    eda_c_df.index = [(1 / fs_dict['chest']) * i for i in range(len(eda_c_df))]
    emg_c_df.index = [(1 / fs_dict['chest']) * i for i in range(len(emg_c_df))]
    resp_c_df.index = [(1 / fs_dict['chest']) * i for i in range(len(resp_c_df))]
    temp_c_df.index = [(1 / fs_dict['chest']) * i for i in range(len(temp_c_df))]

    # Change indices to datetime
    eda_df.index = pd.to_datetime(eda_df.index, unit='s')
    bvp_df.index = pd.to_datetime(bvp_df.index, unit='s')
    temp_df.index = pd.to_datetime(temp_df.index, unit='s')
    acc_df.index = pd.to_datetime(acc_df.index, unit='s')
    label_df.index = pd.to_datetime(label_df.index, unit='s')
    acc_c_df.index = pd.to_datetime(acc_c_df.index, unit='s')
    ecg_c_df.index = pd.to_datetime(ecg_c_df.index, unit='s')
    eda_c_df.index = pd.to_datetime(eda_c_df.index, unit='s')
    emg_c_df.index = pd.to_datetime(emg_c_df.index, unit='s')
    resp_c_df.index = pd.to_datetime(resp_c_df.index, unit='s')
    temp_c_df.index = pd.to_datetime(temp_c_df.index, unit='s')
    
    # Combined dataframe - not used yet
    df = eda_df.join(bvp_df, how='outer')
    df = df.join(temp_df, how='outer')
    df = df.join(acc_df, how='outer')
    df = df.join(label_df, how='outer')
    df = df.join(acc_c_df, how='outer')
    df = df.join(ecg_c_df, how='outer')
    df = df.join(eda_c_df, how='outer')
    df = df.join(emg_c_df, how='outer')
    df = df.join(resp_c_df, how='outer')
    df = df.join(temp_c_df, how='outer')
    df['label'] = df['label'].fillna(method='bfill')
    df.reset_index(drop=True, inplace=True)

    if norm_type is 'std':
        # std norm
        df = (df - df.mean()) / df.std()
    elif norm_type is 'minmax':
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
    window_len = fs_dict['label'] * WINDOW_IN_SECONDS

    for i in range(n_windows):
        # Get window of data
        w = data[window_len * i: window_len * (i + 1)]

        # Add/Calc rms acc
        # w['net_acc'] = get_net_accel(w)
        w = pd.concat([w, get_net_accel(w)])
        #w.columns = ['net_acc', 'ACC_x', 'ACC_y', 'ACC_z', 'BVP',
          #           'EDA', 'EDA_phasic', 'EDA_smna', 'EDA_tonic', 'TEMP',
            #         'label']
        cols = list(w.columns)
        cols[0] = 'net_acc'
        w.columns = cols

        w = pd.concat([w, get_net_accel_C(w)])
        #w.columns = ['net_acc', 'ACC_x', 'ACC_y', 'ACC_z', 'BVP',
          #           'EDA', 'EDA_phasic', 'EDA_smna', 'EDA_tonic', 'TEMP',
            #         'label']
        cols = list(w.columns)
        cols[0] = 'net_acc_C'
        w.columns = cols
        
        # Calculate stats for window
        wstats = get_window_stats(data=w, label=label)

        # Seperating sample and label
        x = pd.DataFrame(wstats).drop('label', axis=0)
        y = x['label'][0]
        x.drop('label', axis=1, inplace=True)

        if feat_names is None:
            feat_names = []
            for row in x.index:
                for col in x.columns:
                    feat_names.append('_'.join([row, col]))

        # sample df
        wdf = pd.DataFrame(x.values.flatten()).T
        wdf.columns = feat_names
        wdf = pd.concat([wdf, pd.DataFrame({'label': y}, index=[0])], axis=1)
        
        # More feats
        wdf['BVP_peak_freq'] = get_peak_freq(w['BVP'].dropna())

        # Add more features here
        # ACC (w and c)
        wdf['net_acc_abs_integral'] = get_absolute_integral(w['net_acc'].dropna())
        wdf['ACC_x_abs_integral'] = get_absolute_integral(w['ACC_x'].dropna())
        wdf['ACC_y_abs_integral'] = get_absolute_integral(w['ACC_y'].dropna())
        wdf['ACC_z_abs_integral'] = get_absolute_integral(w['ACC_z'].dropna())
        wdf['net_acc_C_abs_integral'] = get_absolute_integral(w['net_acc_C'].dropna())
        wdf['ACC_x_C_abs_integral'] = get_absolute_integral(w['ACC_x_C'].dropna())
        wdf['ACC_y_C_abs_integral'] = get_absolute_integral(w['ACC_y_C'].dropna())
        wdf['ACC_z_C_abs_integral'] = get_absolute_integral(w['ACC_z_C'].dropna())
        wdf['ACC_x_peak_freq'] = get_peak_freq(w['ACC_x'].dropna())
        wdf['ACC_y_peak_freq'] = get_peak_freq(w['ACC_y'].dropna())
        wdf['ACC_z_peak_freq'] = get_peak_freq(w['ACC_z'].dropna())
        wdf['ACC_x_C_peak_freq'] = get_peak_freq(w['ACC_x_C'].dropna())
        wdf['ACC_y_C_peak_freq'] = get_peak_freq(w['ACC_y_C'].dropna())
        wdf['ACC_z_C_peak_freq'] = get_peak_freq(w['ACC_z_C'].dropna())

        # EDA(w and c)
        wdf['EDA_slope'] = get_slope(w['EDA'].dropna())
        wdf['EDA_C_slope'] = get_slope(w['EDA_C'].dropna())
        wdf['EDA_drange'] = get_dynamic_range(w['EDA'].dropna())
        wdf['EDA_C_drange'] = get_dynamic_range(w['EDA_C'].dropna())

        # EMG(c)
        wdf['EMG_drange'] = get_dynamic_range(w['EMG_C'].dropna())
        wdf['EMG_abs_integral'] = get_absolute_integral(w['EMG_C'].dropna())
        # RESP(c)
        wdf['Resp_C_rate'], wdf['Resp_C_Inhal_mean'], wdf['Resp_C_Inhal_std'], wdf['Resp_C_Exhal_mean'], wdf['Resp_C_Exhal_std'], wdf['Resp_C_I/E'] = get_resp_features(w['Resp_C'].dropna())

        # TEMP(w and c)
        wdf['TEMP_drange'] = get_dynamic_range(w['TEMP'].dropna())
        wdf['TEMP_C_drange'] = get_dynamic_range(w['Temp_C'].dropna())
        wdf['TEMP_slope'] = get_slope(w['TEMP'].dropna())
        wdf['TEMP_C_slope'] = get_slope(w['Temp_C'].dropna())

        samples.append(wdf)

    return pd.concat(samples)


def make_patient_data(subject_id):
    global savePath
    global WINDOW_IN_SECONDS

    # Make subject data object for Sx
    subject = SubjectData(main_path='data/WESAD', subject_number=subject_id)

    # Empatica E4 data - now with resp
    data_dict = subject.get_data()

    # norm type
    norm_type = None

    # The 3 classes we are classifying
    grouped, baseline, stress, amusement = compute_features(data_dict, subject.labels, norm_type)

    # print(f'Available windows for {subject.name}:')
    n_baseline_wdws = int(len(baseline) / (fs_dict['label'] * WINDOW_IN_SECONDS))
    n_stress_wdws = int(len(stress) / (fs_dict['label'] * WINDOW_IN_SECONDS))
    n_amusement_wdws = int(len(amusement) / (fs_dict['label'] * WINDOW_IN_SECONDS))
    # print(f'Baseline: {n_baseline_wdws}\nStress: {n_stress_wdws}\nAmusement: {n_amusement_wdws}\n')

    #
    baseline_samples = get_samples(baseline, n_baseline_wdws, 1)
    # Downsampling
    # baseline_samples = baseline_samples[::2]
    stress_samples = get_samples(stress, n_stress_wdws, 2)
    amusement_samples = get_samples(amusement, n_amusement_wdws, 0)

    all_samples = pd.concat([baseline_samples, stress_samples, amusement_samples])
    all_samples = pd.concat([all_samples.drop('label', axis=1), pd.get_dummies(all_samples['label'])], axis=1)
    # Selected Features
    # all_samples = all_samples[['EDA_mean', 'EDA_std', 'EDA_min', 'EDA_max',
    #                          'BVP_mean', 'BVP_std', 'BVP_min', 'BVP_max',
    #                        'TEMP_mean', 'TEMP_std', 'TEMP_min', 'TEMP_max',
    #                        'net_acc_mean', 'net_acc_std', 'net_acc_min', 'net_acc_max',
    #                        0, 1, 2]]
    # Save file as csv (for now)
    all_samples.to_csv(f'{savePath}{subject_feature_path}/S{subject_id}_feats_4.csv')

    # Does this save any space?
    subject = None


def combine_files(subjects):
    df_list = []
    for s in subjects:
        df = pd.read_csv(f'{savePath}{subject_feature_path}/S{s}_feats_4.csv', index_col=0)
        df['subject'] = s
        df_list.append(df)

    df = pd.concat(df_list)

    df['label'] = (df['0'].astype(str) + df['1'].astype(str) + df['2'].astype(str)).apply(lambda x: x.index('1'))
    df.drop(['0', '1', '2'], axis=1, inplace=True)

    df.reset_index(drop=True, inplace=True)

    df.to_csv(f'{savePath}/may14_feats4.csv')

    counts = df['label'].value_counts()
    print('Number of samples per class:')
    for label, number in zip(counts.index, counts.values):
        print(f'{int_to_label[label]}: {number}')


if __name__ == '__main__':

    #subject_ids = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
    subject_ids = [2]

    for patient in subject_ids:
        print(f'Processing data for S{patient}...')
        make_patient_data(patient)

    combine_files(subject_ids)
    print('Processing complete.')