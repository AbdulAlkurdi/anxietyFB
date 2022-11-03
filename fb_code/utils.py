import dask as pd
import scipy.signal as scisig
import scipy.stats
import numpy as np

# TODO: Make this a parameter in the future
# Frequency dictionary for WESAD data
fs_dict = {'ACC': 32, 'BVP': 64, 'EDA': 4, 'TEMP': 4, 'label': 700, 'Resp': 700, 'ECG': 700, 'chest': 700}


def get_net_accel(data):
    """
    Function: Computes net acceleration (used for wrist data)

    :param:
        data (DataFrame): data with physiological signals

    :return
        net_acceleration (float): Net acceleration
    """
    return (data['ACC_x'] ** 2 + data['ACC_y'] ** 2 + data['ACC_z'] ** 2).apply(lambda x: np.sqrt(x))

def get_net_accel_C(data):
    """
    Function: Computes net acceleration (used for chest data)

    :param:
        data (DataFrame): data with physiological signals

    :return
        net_acceleration (float): Net acceleration
    """
    return (data['ACC_x'] ** 2 + data['ACC_y'] ** 2 + data['ACC_z'] ** 2).apply(lambda x: np.sqrt(x))

# https://github.com/MITMediaLabAffectiveComputing/eda-explorer/blob/master/load_files.py
def butter_lowpass(cutoff, fs, order=5):
    """
    Function: Low-pass filter used to smooth signals

    :param:
        cutoff
        fs
        order

    :return
        b
        a
    """
    # Filtering Helper functions
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = scisig.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    """
    Function: Computes net acceleration

    :param:

    :return
    """
    # Filtering Helper functions
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = scisig.lfilter(b, a, data)
    return y

def get_slope(series):
    """
    Function: Computes net acceleration

    :param:

    :return
    """
    linreg = scipy.stats.linregress(np.arange(len(series)), series )
    slope = linreg[0]
    return slope

def get_window_stats(data, label=-1):
    """
    Function: Computes net acceleration

    :param:

    :return
    """
    mean_features = np.mean(data)
    std_features = np.std(data)
    min_features = np.amin(data)
    max_features = np.amax(data)

    features = {'mean': mean_features, 'std': std_features, 'min': min_features, 'max': max_features,
                'label': label}
    return features

def get_absolute_integral(x):
    """
    Function: Computes net acceleration

    :param:

    :return
    """
    return np.sum(np.abs(x))

def get_dynamic_range(x):
    """
    Function: Computes net acceleration

    :param:

    :return
    """
    return np.max(x) / np.min(x)

def get_peak_freq(x):
    """
    Function: Computes net acceleration

    :param:

    :return
    """
    f, Pxx = scisig.periodogram(x, fs=8)
    psd_dict = {amp: freq for amp, freq in zip(Pxx, f)}
    peak_freq = psd_dict[max(psd_dict.keys())]
    return peak_freq

# https://github.com/MITMediaLabAffectiveComputing/eda-explorer/blob/master/AccelerometerFeatureExtractionScript.py
def filterSignalFIR(eda, cutoff=0.4, numtaps=64):
    """
    Function: Computes net acceleration

    :param:

    :return
    """
    f = cutoff / (fs_dict['ACC'] / 2.0)
    FIR_coeff = scisig.firwin(numtaps, f)

    return scisig.lfilter(FIR_coeff, 1, eda)