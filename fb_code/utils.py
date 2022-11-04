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