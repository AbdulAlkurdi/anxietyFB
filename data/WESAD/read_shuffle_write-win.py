import os, re, itertools, fnmatch
from datetime import datetime
#from scipy import signal
from time import time
import concurrent.futures
import pandas as pd
import logging
#import pickle
now = datetime.now()

subject_ids = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
snrs = [ 0.01, 0.05, 0.3,0.1, 0.15, 0.2, 0.4, 0.5, 0.6, 0.001, 0.0001] # dones = [ 0.01, 0.05, 0.3]
n_i = [9,8,7,6,5,4,3,2,1,0]## next is [ 6] [7, 8, 9] # dones = [0,1,2,3,4, 5,6, 7, 8 , 9] 
done = itertools.product(snrs, n_i, subject_ids)#snrs , n_i, subject_ids, [factor])
i_max = len(snrs)*len(subject_ids)*len(n_i) 


stng = "fixed_resampled140hz_S*.pkl"
dir_root = 'D:/Users/alkurdi/data/GN-WESAD/'

subject_ids = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
snrs = [ 0.01, 0.05, 0.3,0.1, 0.15, 0.2, 0.4, 0.5, 0.6, 0.001, 0.0001] # dones = [ 0.01, 0.05, 0.3]
n_i = [9,8,7,6,5,4,3,2,1,0]## next is [ 6] [7, 8, 9] # dones = [0,1,2,3,4, 5,6, 7, 8 , 9] 
i_max = len(snrs)*len(subject_ids)*len(n_i) 

def get_incomplete(factor):
    comples = list(itertools.product(snrs, n_i, subject_ids, [factor]))
    combis = []
    i = 0
    for root, dirnames, filenames in os.walk(dir_root):
        for filename in fnmatch.filter(filenames, stng):
            i += 1; #print(root)
            broken_root = re.split(r'/',root)
            broken_root.append(re.split(r'\\',broken_root[-1]))
            golden_ticket = broken_root[-1]; #print(golden_ticket)
            a, b, c = golden_ticket
            n = a.split('_')[1]
            snr = b.split('_')[1]
            subject_id = c[1:] ; #print(f'n_i: {n_i} | snr: {snr} | subject_id: {subject_id}')
            combi = (float(snr), int(n), int(subject_id), factor)
            combis.append(combi)
    print(f'Number of files: {i} | len combi {len(combis)} | len of comples {len(comples)}')
    print('now lets be smart about it and find the missing ones')
    missing = []
    for c in comples:
        if c not in combis:
            missing.append(c)
    print(f'len of missing {len(missing)}')
    '''
    at 10:48pm 11/18/2023 len of missing was 906
    at 11:11pm 11/18/2023 len of missing was 388
    '''
    return missing
#missing = get_incomplete(factor = 5)

def verify():
    df_counter = [['ecg len', 'bvp len', 'eda len', 'acc len', 'temp len', 'resp len', 'emg len', 'eda len', 'temp len', 'acc len', 'bvp len', 'subject']]
    big_start_time = time()
    for root, dirnames, filenames in os.walk(dir_root):
        for filename in fnmatch.filter(filenames, stng):
            small_start_time = time()
            #print(os.path.join(root, filename))
            broken_root = re.split(r'/',root)
            broken_root.append(re.split(r'\\',broken_root[-1]))
            golden_ticket = broken_root[-1]; #print(golden_ticket)
            a, b, c = golden_ticket
            n = a.split('_')[1]
            snr = b.split('_')[1]
            subject_id = c[1:] ; #print(f'n_i: {n_i} | snr: {snr} | subject_id: {subject_id}')
            #print if pickle file is available
            #print(os.path.exists(os.path.join(root, filename)),os.path,os.path.join(root, filename))
            df = pd.read_pickle(os.path.join(root, filename))
            #print(len(df['signal']['chest']['ECG']), len(df['signal']['wrist']['BVP']))
            df_counter.append([len(df['signal']['chest']['ECG']), len(df['signal']['wrist']['BVP']), len(df['signal']['chest']['EDA']), len(df['signal']['chest']['ACC']), len(df['signal']['chest']['Temp']), len(df['signal']['chest']['Resp']), len(df['signal']['chest']['EMG']), len(df['signal']['chest']['EDA']), len(df['signal']['chest']['Temp']), len(df['signal']['wrist']['ACC']), len(df['signal']['wrist']['BVP']), df['subject']])
            print(f'finished {n} {snr} {subject_id}  iteration in {round(time()-small_start_time,2)} seconds')
    df_counter = pd.DataFrame(df_counter)
    df_counter.to_csv('fixed_resampled140hz_counter.csv')
    print(f'finished in {round(time()-big_start_time,2)} seconds, which is {round((time()-big_start_time)/len(df_counter),2)} seconds per iteration')

def read_shuffle_write(snr,  n_i, subject_id, factor = 5):
    sesh_id =[n_i,snr,subject_id]
    #from_path = 'D:/Users/alkurdi/data/'
    #to_path = 'D:/Users/alkurdi/data/'
    from_path = 'D:/Users/alkurdi/data'
    to_path = 'D:/Users/alkurdi/data'
    gn_path = from_path+'/GN-WESAD'
    start_time = time()
    wesad_path = from_path+'/WESAD'
    msg = f'starting  n_i: {n_i}; snr: {snr}, id: {subject_id}. iteration'
    logging.info(msg)
    sesh_path = '/n_'+str(n_i)+'/snr_'+str(snr)+'/S'+str(subject_id)
    load_ws_path = wesad_path + '/S'+str(subject_id) + '/S'+str(subject_id)+'.pkl'
    load_gn_path =  gn_path + sesh_path + '/S'+str(subject_id)+'.pkl'
    factor = factor
    savePath = gn_path + sesh_path + f'/fixed_resampled140hz_S{subject_id}.pkl'
    if os.path.exists(f'{savePath}'):
        msg = savePath+'n_i: '+ str(n_i)+ '; snr: '+ str(snr)+ '; subject_id: '+str(subject_id)+ ' already exist.'
        logging.info(msg)
    else:
        logging.info(f'n_i: {n_i}, snr: {snr}, subject_id: {subject_id} does not exist...but FEAR NOT!, we will create it now')    
        logging.info(f'{sesh_id} unpickling ws {load_ws_path}')
        try: 
            gn_df = pd.read_pickle(load_gn_path)
            logging.info(f'{sesh_id} unpickling gn from {load_gn_path}')
        except Exception as error:
            logging.warn(f'{sesh_id} gn file not found at {load_gn_path}, error: {error}')
            return
        try:
            ws_df = pd.read_pickle(load_ws_path)
        except Exception as error:
            logging.warn(f'{sesh_id} ws file not found at {load_ws_path}, error: {error}')
            return
        logging.info(f'{sesh_id} ws unpickled {load_ws_path}')
        logging.debug(f'{sesh_id} gn unpickled {gn_df.keys()}')
        fixed_gn_df = {}; fixed_gn_df['signal'] = {}
        wrist = {}; chest = {}
        new_len = int(len(ws_df['signal']['chest']['ECG'])/factor)
        logging.debug(f'{sesh_id} resampling and playing with chest')
        fixed_gn_df['label'] = signal.resample(ws_df['label'],new_len)
        chest['ACC'] = signal.resample(gn_df['ACC'],new_len) #gn_df['ACC'] #= #
        chest['ECG'] = signal.resample(gn_df['ECG'],new_len) #= gn_df['ECG']
        chest['EMG'] = signal.resample(gn_df['EMG_C'],int(new_len/100)+1)#= gn_df['EMG_C'] 
        chest['EDA'] = signal.resample(gn_df['EDA_C'],int(new_len/100)+1)#= gn_df['EDA_C']
        chest['Temp'] = signal.resample(gn_df['Temp_C'],int(new_len/100)+1)#= gn_df['Temp_C']
        chest['Resp'] = signal.resample(gn_df['Resp_C'],new_len)#gn_df['Resp_C'] #
        logging.debug(f'{sesh_id} resampling and playing with wrist')
        wrist['ACC'] = signal.resample(gn_df['ACC'],new_len) #= gn_df['ACC']
        wrist['BVP'] = gn_df['BVP'] # signal.resample(gn_df['BVP'],new_len) #= 
        wrist['EDA'] = gn_df['EDA'] #signal.resample(gn_df['EDA'],new_len) #= 
        wrist['TEMP'] = gn_df['TEMP'] #= signal.resample(gn_df['TEMP'],new_len) # 
        logging.debug(f'{sesh_id} combining and saving')
        fixed_gn_df['signal']['chest'] = chest 
        fixed_gn_df['signal']['wrist'] = wrist
        fixed_gn_df['subject'] = ws_df['subject']
        logging.info(f'{sesh_id}...starting to pkl')
        with open(f'{savePath}', 'wb') as f:
            pickle.dump(fixed_gn_df, f)
        logging.info(f'{sesh_id}...finished {n_i} {snr} {subject_id}  iteration in {round(time()-start_time,2)} seconds')
        #return [savePath, fixed_gn_df] #} {snr} {subject_id}  iterations in ', time()-start_time, 'seconds'
def main(missing):
    start_time = time()
    #logging.debug(f'combo: {list[combo]}')
    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(read_shuffle_write, i[0], i[1], i[2], i[3]) for i in missing]
        doner, not_doner = concurrent.futures.wait(futures, return_when='ALL_COMPLETED')
        logging.debug(f'doner assigning futures, now waiting for them to finish')
        for future in doner:
            results = future.result()
    # we'll let each process write its own file
    #for savePath, df in results:
    #    pd.to_pickle(df, savePath)
    num_its = len(n_i)*len(snrs)*len(subject_ids)
    logging.info(f'finished {num_its} iterations in {round(time()-start_time,2)} seconds')
    logging.info(f'an average of {(time()-start_time)/num_its} seconds per iteration')
if __name__ == '__main__':
    factor = 5  #reducing the bigger (ECG) sampling rate by a factor of 5 snr,  n_i, subject_id
    # it's safe to go down to 10 ref. dziezec et al.
    missing = get_incomplete(factor)
    main(missing)
    verify()

