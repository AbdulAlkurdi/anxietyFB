# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 16:34:09 2020

@author: Abdul Alkurdi
"""
import numpy as np
from biosppy.signals import ecg
from scipy import signal
import pandas as pd
import matplotlib.pyplot as plt
import time
from ctypes import CDLL, POINTER
from ctypes import c_size_t, c_double

def time_measurements(RR,x):
        # Parameters: RR: RR intervals in ms
        #        x: threshold in ms, typically: 50
        # return: a dict holds the time domain measurements: pnnx, rmssd (please check the definition of pnnx, rmssd on the table1 above)

        # IRI: the difference between RR intervals    

        IRI = np.abs(np.diff(RR))
        n=np.sum(IRI > x)
        pnnx=(n/len(IRI))*100

        rmssd = np.sqrt(np.mean(IRI ** 2))

        return dict(zip(['pnnx', 'rmssd'],
                        [pnnx, rmssd]))

def freq_measurements(RR, fs, bands, method='periodogram', plotting=True):
        '''Estimatation of the power spectral density (PSD).
        Parameters: RR: RR intervals in ms
                    fs: sampling frequency of the RR signal (not the ecg)
                    method: method used to estimate the power spectral density (PSD).
                    plotting: True if you want to plot the PSD.
        return: a dict holds some freq domain measurements.
                very low frequency (vlf), low frequency (lf), and high frequency (hf) components. 
                LF/HF ratio (lf_hf),  normalized vlues of the high and low frequencies (lfnu, hfnu)
        (please check the definitions in this paper: 
        the Malik, M., Bigger, J. T., Camm, A. J., Kleiger, R. E., Malliani, A., Moss, A. J., & Schwartz, P. J. (1996). Heart rate variability: Standards of measurement, physiological interpretation, and clinical use. European heart journal, 17(3), 354-381.)

        f: Array of sample frequencies.
        Pxx_den: Power spectral density or power spectrum of x.

        For implementation functions, refer to: 
                         https://docs.scipy.org/doc/scipy-0.13.0/reference/generated/scipy.signal.periodogram.html
                        https://docs.scipy.org/doc/scipy-0.13.0/reference/generated/scipy.signal.welch.html
                        https://docs.scipy.org/doc/numpy-1.9.2/reference/generated/numpy.trapz.html#numpy.trapz '''
        # using two methods:
        if method == 'periodogram':
            f, Pxx_den = signal.periodogram(x=RR, fs=fs)
        elif method == 'welch':
            f, Pxx_den = signal.welch(x=RR, fs=fs)
        # finding the indices of each band.
        vlf_indices = np.logical_and(f>= bands['vlf'][0], f < bands['vlf'][1])
        lf_indices = np.logical_and(f>= bands['lf'][0], f < bands['lf'][1])
        hf_indices = np.logical_and(f>= bands['hf'][0], f < bands['hf'][1])
        # integrate the power spectral density at each band.
        vlf = np.trapz(y=Pxx_den[vlf_indices], x=f[vlf_indices])
        lf = np.trapz(y=Pxx_den[lf_indices], x=f[lf_indices])
        hf = np.trapz(y=Pxx_den[hf_indices], x=f[hf_indices])

        total_power = vlf + lf + hf
        lf_hf = lf / hf
        lfnu = (lf / (total_power - vlf)) * 100
        hfnu = (hf / (total_power - vlf)) * 100
        return dict(zip(['total_power', 'vlf', 'lf', 'hf', 'lf_hf', 'lfnu','hfnu'],
                        [total_power, vlf, lf, hf, lf_hf, lfnu, hfnu]))

def freq_ratio( ecg, fs, method='welch', factor = 1):
    '''
    To run this with WESAD, you can send ECG or PPG signal only. Time to execute is exponential wrt vector length.
    I do not need to run more than 60 seconds of data (WESAD ecg fs is 700 and e4 ppg fs is 64)
    Notes
    - Heart Rate Variability (HRV) represents the fluctuations between two consecutive beats in the heart rate record.
    - Heart Rate Variability (HRV) is based on the estimation of the distance between two consecutive beats which often called NN intervals or RR intervals.  
    - Variation in the heart rate can be evaluated using two main methods: time-domain and Frequency domain. 
    - Some of these measures are derived from direct estimation of the RR interval. Others are derived from the differences between RR intervals.
    - Many of the measures correlate closely with others.
    - The method selected should correspond to the aim of each study.
    - The key part before applying any of these methods is to accurately estimate the RR intervals for a given signal. 
    For references: 
    - Malik, M., Bigger, J. T., Camm, A. J., Kleiger, R. E., Malliani, A., Moss, A. J., & Schwartz, P. J. (1996).  Heart rate variability: Standards of measurement, physiological interpretation, and clinical use.  European heart journal, 17(3), 354-381.
    -  An Efficient Algorithm for Automatic Peak Detection in Noisy Periodic and Quasi-Periodic Signals.
    - https://github.com/rhenanbartels/hrv
    This code was made by and made available by Ahmad Algaraawi.  
    '''
   # print("Start")
    fs = int(np.round(fs/factor))
 #   print("A")
    ## Initialization
    # N: total No. of samples
    # L: number of scales
    # alpha: constant factor 
    # M: Local Maxima Scalogram matrix (initialize with random numbers)
    # M_reshaped: Local Maxima Scalogram matrix, reshaped to the scales of interest.
    # k: scale (frequency resolution)
    N =  ecg.shape[0] 
   # print(N) 
   # print("B")                         
    L = int(2*fs)  
   # print("C")     
    alpha =10;  
   # print("D")                              
    M = alpha + np.abs(np.random.randn(L,N))
   # print(M)
  #  print("E")
    M_reshaped =0
     
    #print("M: ") 
    #print(M)
  #  print("F")

    so_file = "./peakFinder.so"
    peakFinder = CDLL(so_file)
  #  print("G")

    ND_POINTER_2 = np.ctypeslib.ndpointer(dtype=np.float64,
                                      ndim=2,
                                      flags="C")

  #  print("H")
    ND_POINTER_1 = np.ctypeslib.ndpointer(dtype=np.float64,
                                      ndim=1,
                                      flags="C")

# define the prototype
  #  print("End")
    peakFinder.print_matrix.argtypes = [ND_POINTER_2, c_size_t, c_size_t, ND_POINTER_1 ]
    ecg_resize = ecg[:, 0]
    peakFinder.print_matrix(M, M.shape[0], M.shape[1], ecg_resize )
  #  print("ecg shape")
   # print(ecg.shape)
    
    gamma = np.sum(M, axis=1)
    chosen = np.argmin(gamma)

    M_reshaped = M[0:chosen,:]
   # print("M_reshaped")
   # print(M_reshaped)

    standard = np.std(M_reshaped, axis=0)

    peakLocations = np.where(standard==0)[0]

   # print("peak locations")
   # print(peakLocations)

    peakLocations_time = peakLocations/fs
    RR = np.diff(peakLocations_time) * 1000
   # print("rr:")
   # print(RR)
    time_dict = time_measurements(RR, 50)

    bands = {'vlf': (0, 0.04), 'lf': (0.04, 0.15), 'hf': (0.15, 0.4)}
    if method =='welch':
      pack = freq_measurements(RR, 1.0, bands, method='welch')
    else:
      pack = freq_measurements(RR, 1.0, bands)
    
    if False:
        sample = 1000
        ax =plt.figure(figsize=(8,8))
        plt.imshow(M_reshaped[:,:sample], aspect=sample/100);
        plt.title('Local Maxima Scalogram matrix (LMS)')
        plt.ylabel('scale (k)')
        plt.xlabel('sample')
    return pack, ecg, RR, time_dict, M_reshaped, M


    '''
    loop_start_time = time.time()
    
    for k in range(0, int(L)): 
        for i in range(k+1, N-k-1):
            if (ecg[i]>ecg[i-k-1] and ecg[i]>ecg[i+k+1]):
                M[k,i]=0
    peakFinder.print_matrix(M, M.shape[0], M.shape[1], ecg_resize )
    loop_end_time = time.time()
    loop_time = loop_end_time - loop_start_time
    '''
   # print("loop time")
  #  print(loop_time)
'''
    with open('/projects/bbyn/lli4/feature-based/results/160writeHDF.csv', 'a') as file:
        file.write("double for loop time:")
     #   file.write("REAL")
        file.write(str(loop_time))
        file.write("\n")
        '''
    # gamma: array of shape L. 
    #        by summing all the columns, you get a vector that contains 
    #        the information about the scale-dependant distribution of zeros (local maxima)
    # chosen: number of chosen scales (rows), which is index of the global minimum 
    #         This represents the scale with the most local maxima.
    
    
    # M_reshaped: Local Maxima Scalogram matrix, reshaped to the scales of interest.
    # standard: shape N 
    #           the standard deviation of the reshaped M matrix in the x-axis.
    
    # peakLocations: ecg peaks locations in samples.
    # peakLocations_time: ecg peaks locations in time (s)
    # RR intervals in ms
    
   
   # measurement_start = time.time()
    
   # measurement_end = time.time()
   # measurement_time = measurement_end - measurement_start
  #  print("measurement time")
   # print(measurement_time)
'''
    with open('/projects/bbyn/lli4/feature-based/results/160writeHDF.csv', 'a') as file:
        file.write("time_measurement time:")
      #  file.write("REAL")
        file.write(str(measurement_time))
        file.write("\n")
'''
    
      #  freq1_start = time.time()
        
      #  freq1_end = time.time()
      #  freq1_time = freq1_end - freq1_start
       # print("freq 1 time")
       # print(freq1_time)
        
      
    

def analyze_ecg(series, fs, seg_len = 30):
    #freq_hybrid returns pack, ecg
    BS_signal_analysis = ecg.ecg(signal=series, sampling_rate=fs, show=True)
    pack,  ecg_out  = freq_ratio_hybrid(series, fs=fs, RR=BS_signal_analysis['rpeaks'])
    return BS_signal_analysis, pack, ecg_out

def freq_ratio_hybrid( ecg, fs, RR, method, factor = 1):
    '''
    Notes
    - Heart Rate Variability (HRV) represents the fluctuations between two consecutive beats in the heart rate record.
    - Heart Rate Variability (HRV) is based on the estimation of the distance between two consecutive beats which often called NN intervals or RR intervals.  
    - Variation in the heart rate can be evaluated using two main methods: time-domain and Frequency domain. 
    - Some of these measures are derived from direct estimation of the RR interval. Others are derived from the differences between RR intervals.
    - Many of the measures correlate closely with others.
    - The method selected should correspond to the aim of each study.
    - The key part before applying any of these methods is to accurately estimate the RR intervals for a given signal. 
    For references: 
    - Malik, M., Bigger, J. T., Camm, A. J., Kleiger, R. E., Malliani, A., Moss, A. J., & Schwartz, P. J. (1996).  Heart rate variability: Standards of measurement, physiological interpretation, and clinical use.  European heart journal, 17(3), 354-381.
    -  An Efficient Algorithm for Automatic Peak Detection in Noisy Periodic and Quasi-Periodic Signals.
    - https://github.com/rhenanbartels/hrv

    This code was made by and made available by Ahmad Algaraawi. Modified by Abdul Alkurdi to use an alternative method to spectography 
    '''
    bands = {'vlf': (0, 0.04), 'lf': (0.04, 0.15), 'hf': (0.15, 0.4)}
    fs = int(np.round(fs/factor))
    time_dict = time_measurements(RR, 50)
    if method =='welch':
        pack = freq_measurements(RR, 1.0, bands, method='welch')
    else:
        pack = freq_measurements(RR, 1.0, bands)
    return pack, ecg, time_dict

def freq_ratio_fast(ecg, fs, method='welch', factor=1):
    fs = int(np.round(fs / factor))
    N = ecg.shape[0]
    L = int( fs/2)
    alpha = 10
    M = alpha + np.abs(np.random.randn(L, N))

    # Vectorization
    #for k in range(1, L):
    #    ecg_shifted_right = np.roll(ecg, -k - 1)
    #    ecg_shifted_left = np.roll(ecg, k + 1)
    #    M[k, k + 1:N - k - 1] *= (ecg[k + 1:N - k - 1] <= ecg_shifted_right[k + 1:N - k - 1]) | \
    #                            (ecg[k + 1:N - k - 1] <= ecg_shifted_left[k + 1:N - k - 1])
    # Using NumPy array operations for faster computation
    # Define the maximum offset for comparison
    max_offset = 50

    for k in range(0, int(L)): 
        for i in range(k+1, N-k-1):
            if (ecg[i]>ecg[i-k-1] and ecg[i]>ecg[i+k+1]):
                M[k,i]=0
    print('6th try')

    gamma = np.sum(M, axis=1)
    chosen = np.argmin(gamma)
    M_reshaped = M[0:chosen, :]
    standard = np.std(M_reshaped, axis=0)
    peakLocations = np.where(standard == 0)[0]
    peakLocations_time = peakLocations / fs
    RR = np.diff(peakLocations_time) * 1000
    time_dict = time_measurements(RR, 50)
    bands = {'vlf': (0, 0.04), 'lf': (0.04, 0.15), 'hf': (0.15, 0.4)}
    if method =='welch':
        pack = freq_measurements(RR, 1.0, bands, method='welch')
    else:
        pack = freq_measurements(RR, 1.0, bands)
    if False:
        sample = 1000
        ax =plt.figure(figsize=(8,8))
        plt.imshow(M_reshaped[:,:sample], aspect=sample/100);
        plt.title('Local Maxima Scalogram matrix (LMS)')
        plt.ylabel('scale (k)')
        plt.xlabel('sample')
    return pack, ecg, RR, time_dict, M_reshaped, M