import dask as pd
import scipy.signal as scisig
import scipy.stats
import numpy as np

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
    return (data['ACC_x'] ** 2 + data['ACC_y'] ** 2 + data['ACC_z'] ** 2).apply(lambda x: np.sqrt(x)) if part == 'wrist' else \
        (data['ACC_x_C'] ** 2 + data['ACC_y_C'] ** 2 + data['ACC_z_C'] ** 2).apply(lambda x: np.sqrt(x))

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

    Note: uses a periodogram (https://en.wikipedia.org/wiki/Periodogram)

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
    """
    Function: Computes net acceleration
    https://github.com/MITMediaLabAffectiveComputing/eda-explorer/blob/master/AccelerometerFeatureExtractionScript.py
    :param:

    :return
    """
    f = cutoff / (fs_dict['ACC'] / 2.0)
    FIR_coeff = scisig.firwin(numtaps, f)

    return scisig.lfilter(FIR_coeff, 1, eda)