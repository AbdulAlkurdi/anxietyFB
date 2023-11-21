'''
This version includes the following changes:
- using multiprocessing pool for adding noise, computing features and getting samples
- using global logger
- 
'''
import numpy as np
import os
import pickle
import dask as pd
import pandas as pd_old
from datetime import datetime
import warnings
import utils
from feature_extraction import SubjectData, compute_features, get_samples
from datetime import datetime
from multiprocessing import Pool
import sys
from loggerwrapper import GLOBAL_LOGGER

# To ignore all warnings:
warnings.filterwarnings("ignore", module="numpy")
warnings.filterwarnings("ignore", module="scipy")
starting = datetime.today().strftime('%Y-%m-%d--%H-%M-%p')

label_dict = {'baseline': 1, 'stress': 2, 'amusement': 0}
int_to_label = {1: 'baseline', 2: 'stress', 0: 'amusement'}
feat_names = None
loadPath = '/mnt/d/Users/alkurdi/data/WESAD'
savePath = '/mnt/d/Users/alkurdi/data/GN-WESAD'
subject_feature_path = '/subject_feats'
onedrive = '/mnt/d/Users/alkurdi/data/GN-WESAD'

subject_ids = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
snrs = [ 0.0001,  0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 1, 2] #0.00001,
#snrs = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.6]
#snrs = [0.5, 0.6]
n_samples = 10 # Number of samples to generate for each SNR
if not os.path.exists(savePath):
    os.makedirs(savePath)
if not os.path.exists(savePath + subject_feature_path):
    os.makedirs(savePath + subject_feature_path)
# Make directories for each SNR and each sample
for n_i in range(n_samples):
    if not os.path.exists(savePath + '/n_'+str(n_i)):
        os.makedirs(savePath + '/n_'+str(n_i))
    for snr in snrs:
        if not os.path.exists(savePath + '/n_'+str(n_i)+'/snr_'+str(snr)):
            os.makedirs(savePath + '/n_'+str(n_i)+'/snr_'+str(snr))
        if not os.path.exists(savePath + '/n_'+str(n_i)+'/snr_'+str(snr)+ subject_feature_path):
            os.makedirs(savePath + '/n_'+str(n_i)+'/snr_'+str(snr)+ subject_feature_path)
        for subject_id in subject_ids:
            if not os.path.exists(savePath + '/n_'+str(n_i)+'/snr_'+str(snr)+ '/S'+str(subject_id)):
                os.makedirs(savePath + '/n_'+str(n_i)+'/snr_'+str(snr)+ '/S'+str(subject_id))
            if not os.path.exists(onedrive + '/n_'+str(n_i)+'/snr_'+str(snr)+ '/S'+str(subject_id)):
                os.makedirs(onedrive + '/n_'+str(n_i)+'/snr_'+str(snr)+ '/S'+str(subject_id))
def calculate_param(signal, noise_type, signal_to_noise_ratio):
    """
    Function: Calculates the parameters of a continuous probability density function given
    our desired signal to noise ratio and signal.
    
    :param:
        signal (array or ndarray): Our signal
        noise_type (string): Probability distribution of our noise (i.e., Gaussian)
        signal_to_noise_ratio (float): Desired signal to noise ratio
    
    :return
        sigma (float): sigma of the gaussian
    """
    if noise_type == 'Gaussian': return (signal**2).mean()/signal_to_noise_ratio
    else: return -1
def gaussian_homoskedastic(signal_name, signal, signal_to_noise_ratio=None):
    """
    Constructs a homoskedastic gaussian probability density function, samples noise from it,
    then adds noise to the signal
    
    :param:
        signal_name (string): The name of the signal (i.e., ECG)
        signal (array or ndarray): The signal we wish to add noise to
        signal_to_noise_ratio (float): [default: None] If specified, our desired SNR.
        
    :return
        noisy_signal: The signal after we have added noise to it
    """
    x_new = None
    if signal_name == 'ACC':
        alpha = 0.5
        mu = 0
        # Noise X Axis
        x_axis = signal[:,0]
        sigma = calculate_param(x_axis, 'Gaussian', signal_to_noise_ratio)
        s = np.random.normal(mu, sigma, 1000)
        x_axis_new = np.copy(x_axis)
        for i in range(len(x_axis_new)):
            x_axis_new[i] += float(np.random.normal(mu, sigma, 1))
        # Noise Y Axis
        y_axis = signal[:,1]
        sigma = calculate_param(y_axis, 'Gaussian', signal_to_noise_ratio)
        s = np.random.normal(mu, sigma, 1000)
        y_axis_new = np.copy(y_axis)
        for i in range(len(y_axis_new)):
            y_axis_new[i] += float(np.random.normal(mu, sigma, 1))
        # Noise Z Axis
        z_axis = signal[:,2]
        sigma = calculate_param(z_axis, 'Gaussian', signal_to_noise_ratio)
        s = np.random.normal(mu, sigma, 1000)
        z_axis_new = np.copy(z_axis)
        for i in range(len(z_axis_new)):
            z_axis_new[i] += float(np.random.normal(mu, sigma, 1))
        # Put together noisy signal
        x_new = np.zeros((len(signal), 3))
        x_new[:,0] = x_axis_new
        x_new[:,1] = y_axis_new
        x_new[:,2] = z_axis_new
        return (x_new, sigma)
    else: 
        # Store original shape
        original_shape = signal.shape
        # Caveat: some signals like ACC have three axes
        # Flatten signal to be 1d
        x = np.ravel(signal)
        # Calculate mean and Standard deviation
        alpha = 0.5
        mu = 0
        sigma = calculate_param(x, 'Gaussian', signal_to_noise_ratio)
        # Add noise
        x_new = x + np.random.normal(mu, sigma, (len(x),))
        return (np.array(x_new).reshape(original_shape), sigma)


def add_noise(data, subject_id, snr, n_i):
    """
    Function: Adds noise to the WESAD data, stored in the specified root directory.
    
    :param:
        data
    :return:
        noised data
    """
    noisy_data = {}
    signals_list = data.keys()
    for signal_name in signals_list: 
        #print(f'data before sending:{data[signal_name]}')
        x_gaussian_homoskedastic, _ = gaussian_homoskedastic(signal_name, data[signal_name] , snr)
        noisy_data[signal_name] = x_gaussian_homoskedastic
    # pickle subject files
    with open(onedrive + '/n_'+str(n_i)+'/snr_'+str(snr)+ '/S'+str(subject_id)+'/S'+str(subject_id)+'.pkl', 'wb') as dest:
        pickle.dump(noisy_data, dest) 
    return noisy_data
def call_add_noise_windows_samples_then_combine(data_dict, subject_id, snr, n_i, labels ):
    noisy_dict = add_noise(data_dict, subject_id, snr, n_i)
    # The 3 classes we are classifying
    grouped, baseline, stress, amusement = compute_features(noisy_dict, labels, norm_type=None)
    print('baseline: ',len(baseline),'stress: ',len(stress),'amusement: ',len(amusement))
    # Get windows
    n_baseline_wdws = len(range(0,len(baseline) - WINDOW_IN_SECONDS*utils.fs_dict['label']+1,int(stride*utils.fs_dict['label'])))
    n_stress_wdws = len(range(0,len(stress) - WINDOW_IN_SECONDS*utils.fs_dict['label']+1,int(stride*utils.fs_dict['label'])))
    n_amusement_wdws = len(range(0,len(amusement) - WINDOW_IN_SECONDS*utils.fs_dict['label']+1,int(stride*utils.fs_dict['label'])))        
    # Get samples
    baseline_samples = get_samples(baseline, n_baseline_wdws, 1)
    stress_samples = get_samples(stress, n_stress_wdws, 2)
    amusement_samples = get_samples(amusement, n_amusement_wdws, 0)
    all_samples = pd_old.concat([baseline_samples, stress_samples, amusement_samples])
    all_samples = pd_old.concat([all_samples.drop('label', axis=1), pd_old.get_dummies(all_samples['label'])], axis=1)
    #noisy_dict[n_i] = all_samples
    all_samples.to_csv(f'{savePath}/n_{n_i}/snr_{snr}/subject_feats/S{subject_id}_feats.csv')
    return all_samples
def make_patient_data_wnoise(subject_id, snr, n_samples): #this makes data for 1 patient, 1 snr and all samples 
    #global savePath
    #global WINDOW_IN_SECONDS
    #global stride
    # Make subject data object for Sx 
    subject = SubjectData(main_path=loadPath, subject_number=subject_id)
    # Empatica E4 data - now with resp
    data_dict = subject.get_wrist_and_chest_data()
    noisy_dict = {}
    with Pool() as pool:
        # add noise
        pool.starmap(call_add_noise_windows_samples_then_combine, [(data_dict, subject_id, snr, n_i, subject.labels) for n_i in range(n_samples)])
def combine_noiZ_files(subjects):
    df_list = []
    for snr in snrs:
        for n_i in range(n_samples):
            for s in subjects:
                df = pd_old.read_csv(f'{savePath}/n_{n_i}/snr_{snr}{subject_feature_path}/S{subject_id}_feats.csv', index_col=0)
                df['subject'] = s
                df_list.append(df)
            df = pd_old.concat(df_list)
            df['label'] = (df['0'].astype(str) + df['1'].astype(str) + df['2'].astype(str)).apply(lambda x: x.index('1'))
            df.drop(['0', '1', '2'], axis=1, inplace=True)
            df.reset_index(drop=True, inplace=True)
            now = datetime.today().strftime('%Y-%m-%d--%H-%M-%p')
            df.to_csv(f'{savePath}/n_{n_i}/snr_{snr}{subject_feature_path}/{now}_feats.csv')
            print('Saved file to: ',f'{savePath}/n_{n_i}/snr_{snr}{subject_feature_path}/{now}_feats.csv')
    counts = df['label'].value_counts()
    print('Number of samples per class:')
    for label, number in zip(counts.index, counts.values):
        print(f'{int_to_label[label]}: {number}')
def find_incomplete_raws(onedrive):
    #bads are the ones that do not have the gaussian-modified data.  
    bads = []
    bad_snrs = []
    bad_subjects = []
    bad_ns = []
    completed_snrs = []
    for n_i in range(n_samples):
        for snr in snrs:
            for subject_id in subject_ids:
                try: 
                    a = os.listdir(f'{onedrive}/n_{n_i}/snr_{str(snr)}/S{subject_id}')
                    a[0]
                    completed_snrs.append(snr)
                except:
                    bads.append(f'n_{n_i}/snr_{snr}/S{subject_id}')
                    bad_snrs.append(snr)
                    bad_subjects.append(subject_id)
                    bad_ns.append(n_i)
    bad_snrs = sorted(set(bad_snrs))
    bad_subjects = sorted(set(bad_subjects))
    bad_ns = sorted(set(bad_ns))
    completed_snrs = sorted(set(completed_snrs))
    #printing after checking
    now = datetime.today().strftime('%Y-%m-%d--%H-%M-%p')
    GLOBAL_LOGGER.info(str(now)+ '. completed snrs : '+str(completed_snrs))
    GLOBAL_LOGGER.info(f'{now}. incomplete snrs :{bad_snrs}')
    return bad_snrs
if __name__ == '__main__':
    global WINDOW_IN_SECONDS
    global stride
    global snrs_arg
    global subject_id_arg
    subject_id_arg, snrs_arg, n_samples, WINDOW_IN_SECONDS, stride  = sys.argv
    snrs_arg = [snrs_arg]
    subject_id_arg = [subject_id_arg]
    # update snrs, subject_ids and n_samples based on what is already completed
    snrs = find_incomplete_raws(onedrive)
    now = datetime.today().strftime('%Y-%m-%d--%H-%M-%p')
    GLOBAL_LOGGER.info(f'{now}. snrs: {snrs_arg}.')
    with Pool() as pool:
        pool.starmap(make_patient_data_wnoise, [subject_id_arg, snrs_arg, n_samples])
    #combine_noiZ_files(subject_ids) # i am running multiple snrs at the same time on separate terminals, so i will combine them later
    now = datetime.today().strftime('%Y-%m-%d--%H-%M-%p')
    GLOBAL_LOGGER.info(f'Processing for {n_samples} samples for snr: {snrs_arg} participant: {subject_id_arg} for is complete.')