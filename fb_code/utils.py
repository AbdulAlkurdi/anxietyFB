import pandas as pd
import scipy.signal as scisig
import scipy.stats
import numpy as np
import pickle
import os
# TODO: Make this a parameter in the future
fs_dict = {'ACC': 32, 'BVP': 64, 'EDA': 4, 'TEMP': 4, 'label': 700, 'Resp': 700, 'ECG': 700, 'chest': 700} # Frequency dictionary for WESAD data


def get_net_accel(data, part):
    '''get net acceleration

    Function that computes net acceleration for wrist data.

    Parameters
    ----------
    data : pandas dataframe
        data with physiological signals

    Returns
    -------
    out : float
        net acceleration
    '''
    return (data['ACC_x'] ** 2 + data['ACC_y'] ** 2 + data['ACC_z'] ** 2).apply(lambda x: np.sqrt(x)) if part == 'wrist' \
            else (data['ACC_x_C'] ** 2 + data['ACC_y_C'] ** 2 + data['ACC_z_C'] ** 2).apply(lambda x: np.sqrt(x))

def butter_lowpass(cutoff, fs, order=5):
    '''lowpass filter

    Function that creates a low pass filter to be used on smoothing signals.

    Reference: https://github.com/MITMediaLabAffectiveComputing/eda-explorer/blob/master/load_files.py

    Parameters
    ----------
    data : pandas dataframe
        data with physiological signals

    Returns
    -------
    b : float
        filter parameter
    a : float
        filter parameter
    '''
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = scisig.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    '''apply lowpass filter to data

    Function that applies a low pass filter to smooth a signal.

    Parameters
    ----------
    data : pandas dataframe
        data with physiological signals
    cutoff : float
        lowpass filter smoothing parameter
    fs : float
        lowpass filter smoothing parameter
    order : float
        lowpass filter smoothing parameter
        defualt : 5
    Returns
    -------
    y : pandas dataframe or series, or numpy array
        filtered signal
    '''
    b, a = butter_lowpass(cutoff, fs, order=order)
    return scisig.lfilter(b, a, data)

def get_slope(series):
    '''get slope

    Function that computes the slope of a 1-d dataset

    Parameters
    ----------
    series : pandas series
        the series upon which we wish to compute the slope

    Returns
    -------
    slope : float
        slope of the 1d dataset
    '''
    return scipy.stats.linregress(np.arange(len(series)), series )[0]

def get_window_stats(data, label=-1):
    '''get window stats

    Function that computes window stats of a given dataset (i.e., mean,
    std, min, max, etc.).

    Parameters
    ----------
    data : pandas dataframe or series
        the data upon which we wish to compute the window stats
    label : int
        label for the given window stats
        default : -1 
    Returns
    -------
    out : dict
        dictionary containing the window statistics
    '''
    return {'mean': np.mean(data), 'std': np.std(data), 'min': np.amin(data), 'max': np.amax(data),'label': label}

def get_absolute_integral(x):
    '''get absolute integral

    Function that computes the absolute integral of a dataset.

    Parameters
    ----------
    x : pandas series or pandas dataframe
        the series/dataframe upon which we wish to compute the slope

    Returns
    -------
    out : float
        absolute integral of the dataset
    '''
    return np.sum(np.abs(x))

def get_dynamic_range(x):
    '''get dynamic range

    Function that computes the dynamic range of a dataset.

    Parameters
    ----------
    x : pandas series or pandas dataframe
        the series/dataframe upon which we wish to compute the dynamic range

    Returns
    -------
    out : float
        dynamic range of the dataset
    '''
    return np.max(x) / np.min(x)

def get_peak_freq(x):
    '''get peak frequency

    Function that computes the peak frequency of a dataset.

    Reference: https://en.wikipedia.org/wiki/Periodogram

    Parameters
    ----------
    x : pandas series or pandas dataframe
        the series/dataframe upon which we wish to compute the peak frequency

    Returns
    -------
    peak_freq : float
        peak frequency of the dataset
    '''
    f, Pxx = scisig.periodogram(x, fs=8)
    psd_dict = {amp: freq for amp, freq in zip(Pxx, f)}
    peak_freq = psd_dict[max(psd_dict.keys())]
    return peak_freq

def filterSignalFIR(eda, cutoff=0.4, numtaps=64):
    '''filter signal using finite impulse response (FIR)

    Function that filters a signal using the FIR method

    Reference: https://github.com/MITMediaLabAffectiveComputing/eda-explorer/blob/master/AccelerometerFeatureExtractionScript.py

    Parameters
    ----------
    eda : dict
        data to filter
    cutoff : float
        parameter in FIR filter
        default : 0.4
    numtaps : float
        parameter in FIR filter
        default : 64

    Returns
    -------
    peak_freq : float
        peak frequency of the dataset
    '''
    f = cutoff / (fs_dict['ACC'] / 2.0)
    FIR_coeff = scisig.firwin(numtaps, f)
    return scisig.lfilter(FIR_coeff, 1, eda)

def get_f1_spec(cm):
    # Extracting True Positives, False Positives, and True Negatives
    TP = cm[1, 1]
    FP = cm[0, 1] + cm[2, 1]
    TN = cm[0, 0] + cm[0, 2] + cm[2, 0] + cm[2, 2]

    # Calculating Precision, Recall
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FP)

    # Calculating F1 Score
    F1 = 2 * (Precision * Recall) / (Precision + Recall)

    # Calculating Specificity
    Specificity = TN / (TN + FP)

    return F1, Specificity
def calculate_binary_metrics(cm):
    # Extracting True Positives, False Positives, and True Negatives
    TP = cm[1, 1]  # True Positives (Stress)
    FP = cm[0, 1] + cm[2, 1]  # False Positives (Not Stress classified as Stress)
    TN = cm[0, 0] + cm[2, 2]  # True Negatives (Not Stress)
    FN = cm[1, 0] + cm[1, 2]  # False Negatives (Stress classified as Not Stress)

    # Calculating Precision, Recall
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)

    # Calculating F1 Score
    F1 = 2 * (Precision * Recall) / (Precision + Recall)

    return F1

def ammend_results_table(gn_wesad_acc, GN_cm_cr_dict, wesad_acc, WESAD_cm_cr_dict, cases):
    force_update = False
    for case in cases:
        if case == 'WESAD':
            wesad_file_wanted = '/mnt/d/Users/alkurdi/data/WESAD/wesad_models_results-win60stride1_wbinaryf1.csv'
            #if (os.path.isfile(wesad_file_wanted)) & (not force_update):
                #logging.info('WESAD binary f1 results table found. Loading...')
                #wesad_acc = pd.read_csv(wesad_file_wanted,index_col=0,)
            #else:
            #logging.info('Running WESAD case...')
            #with open('/mnt/d/Users/alkurdi/data/WESAD/cm_cr_dict.pickle', 'rb') as handle:
            #    WESAD_cm_cr_dict = pickle.load(handle)
            #wesad_acc = pd.read_csv(
            #    '/mnt/d/Users/alkurdi/data/WESAD/wesad_models_results-win60stride1_wcm_wcr.csv',
            #    index_col=0,)
            wesad_acc['Binary F1'] = None
            wesad_acc['Specificity'] = None                
            for i, cm in enumerate(WESAD_cm_cr_dict['cm']):
                cm = cm['Confusion Matrix']

                _, specificity = get_f1_spec(cm)
                binary_f1_score = calculate_binary_metrics(cm)
                
                _,_, model = WESAD_cm_cr_dict['cr'][i]['id']
                
                if binary_f1_score is not None:
                    wesad_acc.loc[ wesad_acc['Model'] == model,
                                    'Binary F1'] = binary_f1_score
                if specificity is not None:
                    wesad_acc.loc[ wesad_acc['Model'] == model,
                                    'Specificity'] = specificity
            
            wesad_acc.to_csv(wesad_file_wanted)
            print('wesad_acc saved to: data/WESAD/wesad_models_results-win60stride1_wbinaryf1.csv')
        if case == 'GN-WESAD':
            gn_wesad_wanted = '/mnt/d/Users/alkurdi/data/GN-WESAD/GN_wesad_models_results_wbinaryf1.csv'
            #if (os.path.isfile(gn_wesad_wanted) and not force_update):
                
                #logging.info('GN-WESAD binary f1 results table found. Loading...')
            #    gn_wesad_acc = pd.read_csv(gn_wesad_wanted, index_col=0,)
            #else:
                #logging.info('Running GN-WESAD case...')
            #    with open(
            #        '/mnt/d/Users/alkurdi/data/GN-WESAD/cm_cr_dict.pickle', 'rb'
            #    ) as handle:
            #        GN_cm_cr_dict = pickle.load(handle)
            #tgt_file = '/mnt/d/Users/alkurdi/data/GN-WESAD/GN_wesad_models_results_wcm_wcr.csv'
            #gn_wesad_acc = pd.read_csv(tgt_file, index_col=0)
            gn_wesad_acc['Binary F1'] = None

            for i, cm in enumerate(GN_cm_cr_dict['cm']):
                cm = cm['Confusion Matrix']
                binary_f1_score = calculate_binary_metrics(cm)
                _, specificity = get_f1_spec(cm)
                snr , n_i , model = GN_cm_cr_dict['cr'][i]['id']
                if binary_f1_score is not None:
                    gn_wesad_acc.loc[
                        (gn_wesad_acc['Model'] == model)
                        & (gn_wesad_acc['SNR'] == snr)
                        & (gn_wesad_acc['n_i'] == (n_i)),
                        'Binary F1'] = float(binary_f1_score)
                if specificity is not None:
                    gn_wesad_acc.loc[
                        (gn_wesad_acc['Model'] == model)
                        & (gn_wesad_acc['SNR'] == snr)
                        & (gn_wesad_acc['n_i'] == (n_i)),
                        'Specificity'] = specificity
            gn_wesad_acc.to_csv( gn_wesad_wanted)
            print('gn_wesad_acc saved to: data/GN-WESAD/GN_wesad_models_results_wbinaryf1.csv')
        if case == 'PR-WESAD':
            #logging.info('Running PR-WESAD case...')
            loadPath = '/mnt/d/Users/alkurdi/data/PR-WESAD'
    return wesad_acc, gn_wesad_acc


