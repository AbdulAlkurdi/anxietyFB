from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from multiprocessing import Pool
import time
iterations = 10000
global yells
yells = 0
def process_item(item):
    # Your processing code goes here
    #print(f"Processing {item}")
    # Replace the following line with your actual processing code
    time.sleep(0.0005)
    return item * item

def star_process(item1, item2, item3):
    global yells
    print(f'subject id :{item1}, snr: {item2}, sample: {item3}')
    yells = yells+1

#%%

# Example usage of ProcessPoolExecutor to parallelize a for loop
startTime = datetime.now()
with ProcessPoolExecutor() as executor:
    # Submit tasks to the process pool
    
    future_to_item = {executor.submit(process_item, item): item for item in range(iterations)}
    
    # Process as tasks complete
    for future in as_completed(future_to_item):
        item = future_to_item[future]
        try:
            result = future.result()
            # Do something with the result if needed
            #print(f"Result of item {item}: {result}")
        except Exception as exc:
            print(f"Item {item} generated an exception: {exc}")

print('total time for ProcessPoolExecutor parallel ',datetime.now() - startTime)

#%%

startTime = datetime.now()
#for i in range(iterations):
#    result2 = process_item(i)
print('total time for loop',datetime.now() - startTime)


#%%
startTime = datetime.now()
with Pool() as pool:
    #result3 = process_item
    results3 = pool.map(process_item, range(iterations))

print('total time for multiprocessing parallel ',datetime.now() - startTime)


#%%

subject_ids = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
snrs = [ 0.0001,  0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6] #0.00001,
#snrs = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.6]
#snrs = [0.5, 0.6]
n_samples = 10 # Number of samples to generate for each SNR

with Pool() as pool:
    
    pool.starmap(star_process,  [(subject, snr, n_samples) for subject in subject_ids
                                                     for snr in snrs])
    
print(f'total yells {yells}')



#%%

#%%
from feature_extraction import SubjectData, compute_features, get_samples
import pandas as pd_old
import utils

savePath = 'data/WESAD'
loadPath = savePath 
WINDOW_IN_SECONDS = 60
stride = 10
def add_noise(data, subject_id, snr, n_i):
    return data
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
    all_samples.to_csv(f'{savePath}/n_{n_i}/snr_{snr}/subject_feats/S{subject_id}_feats_testparallel.csv')
    return all_samples

def make_patient_data_wnoise(subject_id, snr, n_samples): #this makes data for 1 patient, 1 snr and all samples 
    global savePath
    global WINDOW_IN_SECONDS
    global stride

    # Make subject data object for Sx 
    subject = SubjectData(main_path=loadPath, subject_number=subject_id)
    # Empatica E4 data - now with resp
    data_dict = subject.get_wrist_and_chest_data()
    noisy_dict = {}
    # add noise
    for n_i in range(n_samples):
        all_samples = call_add_noise_windows_samples_then_combine(data_dict, subject_id, snr, n_i, subject.labels)  

    # add noise
    pool.starmap(call_add_noise_windows_samples_then_combine, [(data_dict, subject_id, snr, n_i) for n_i in range(n_samples)])

    pool.starmap(star_process,  [(subject, snr, n_samples) for subject in subject_ids])
