import numpy as np
import os
import pickle
import dask as pd
import pandas as pd_old

rootdir = '/Users/samuelschapiro/Desktop/Research/HCDL/fb_code/data/WESAD'

def calculate_param(signal, noise_type, signal_to_noise_ratio):
    """
    Function: Calculates the parameters of a continuous probability density function given
    our desired signal to noise ratio and signal.
    
    :param:
        signal (array or ndarray): Our signal
        noise_type (string): Probability distribution of our noise (i.e., Gaussian)
        signal_to_noise_ratio (float): Desired signal to noise ratio
    
    :return
    (for now, just)
        sigma (float): sigma of the gaussian
    """
    
    return (signal**2).mean()/signal_to_noise_ratio

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


def write_file(path, file_name, data):
    """
    Function: Writes filename to its specified path and then dumps data in .pkl format to that file.
    
    :param:
        path (string): Where you want to write (create) the file
        file_name (string): What you want to name the file you're creating
        data (array or ndarray or pd.DataFrame): What you want inside the file you're creating
    
    """
    
    filename = file_name + '.pkl'
    with open(os.path.join(path, filename), 'wb') as dest:
        pickle.dump(data, dest)

def add_noise(rootdir, snrs, body_parts=['wrist', 'chest'],
              signals={'wrist': ['ACC', 'BVP', 'EDA', 'TEMP'],'chest': ['ECG', 'Temp', 'EMG', 'Resp', 'ACC']}):
    """
    Function: Adds noise to the WESAD data, stored in the specified root directory
    
    :param:
        rootdir (string): The root directory from which to read the WESAD data.
        body_parts (list): [Default: '[wrist', 'chest']] Body parts from which to read the WESAD data
        signals (dict): [Default: see above] The physiological signals each body part has
    :return
        None
    """
    snr=snrs
    patients = []
    patients_with_noise1 = []
    patients_with_noise2 = []
    patients_with_noise3 = []
    patients_with_noise4 = []
    patients_with_noise5 = []
    patient_idx = 0
    # Iterate through each patient's folder and construct a df with all patient data
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            # If this is a .pkl file then it's the synchronized features/labels
            # and we want to serialize the file
            if '.pkl' in file and 'gauss' not in file: 
                # For each signal to noise ratio we want to test
                # Serialize
                patients.append(pd.read_pickle(subdir + '/' + file))
                patients_with_noise1.append(pd.read_pickle(subdir + '/' + file))
                patients_with_noise2.append(pd.read_pickle(subdir + '/' + file))
                patients_with_noise3.append(pd.read_pickle(subdir + '/' + file))
                patients_with_noise4.append(pd.read_pickle(subdir + '/' + file))                
                patients_with_noise5.append(pd.read_pickle(subdir + '/' + file))
                for body_part in body_parts:
                    # Add noise
                    for sgl in signals[body_part]:
                        # Get signal
                        signal = patients[patient_idx]['signal'][body_part][sgl]

                        # Get gaussian homoskedastic noise1
                        x_gaussian_homoskedastic1, sigma1 = gaussian_homoskedastic(sgl, signal, snr[0])
                        patients_with_noise1[patient_idx]['signal'][body_part][sgl] = x_gaussian_homoskedastic1

                        # Get gaussian homoskedastic noise2
                        x_gaussian_homoskedastic2, sigma2 = gaussian_homoskedastic(sgl, signal, snr[1])
                        patients_with_noise2[patient_idx]['signal'][body_part][sgl] = x_gaussian_homoskedastic2
                        
#                         # Get gaussian homoskedastic noise3
#                         x_gaussian_homoskedastic3, sigma3 = gaussian_homoskedastic(sgl, signal, snr[2])
#                         patients_with_noise3[patient_idx]['signal'][body_part][sgl] = x_gaussian_homoskedastic3
                        
#                         # Get gaussian homoskedastic noise2
#                         x_gaussian_homoskedastic4, sigma4 = gaussian_homoskedastic(sgl, signal, snr[3])
#                         patients_with_noise4[patient_idx]['signal'][body_part][sgl] = x_gaussian_homoskedastic4
                        
#                         # Get gaussian homoskedastic noise3
#                         x_gaussian_homoskedastic5, sigma5 = gaussian_homoskedastic(sgl, signal, snr[4])
#                         patients_with_noise5[patient_idx]['signal'][body_part][sgl] = x_gaussian_homoskedastic5
                        
                # Export each noise type
                write_file(subdir, file.split('.')[0]+'_gauss_homo_snr_'+str(snr[0]), 
                           patients_with_noise1[patient_idx])
                
                # Export each noise type
                write_file(subdir, file.split('.')[0]+'_gauss_homo_snr_'+str(snr[1]), 
                           patients_with_noise2[patient_idx])
                
#                 # Export each noise type
#                 write_file(subdir, file.split('.')[0]+'_gauss_homo_snr_'+str(snr[2]), 
#                            patients_with_noise3[patient_idx])
                
#                 # Export each noise type
#                 write_file(subdir, file.split('.')[0]+'_gauss_homo_snr_'+str(snr[3]), 
#                            patients_with_noise4[patient_idx])
                
#                 # Export each noise type
#                 write_file(subdir, file.split('.')[0]+'_gauss_homo_snr_'+str(snr[4]), 
#                            patients_with_noise5[patient_idx])
                
                # Increment patient index
                patient_idx += 1