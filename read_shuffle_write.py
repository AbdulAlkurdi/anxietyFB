import os
from datetime import datetime
from scipy import signal
from time import time
import concurrent.futures
import itertools
import pandas as pd
import logging
import pickle
now = datetime.now()
#home= '/mnt/c/Users/alkurdi/Downloads/WESAD'
home = 'D:/Users/alkurdi/data' #windows version
#home = '/mnt/d/Users/alkurdi/data/'
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename= f'{home}/logs/{now.strftime("%H_%M_%S")}_{os.path.basename(__file__)[0:-3]}.log' )
logging.getLogger().addHandler(logging.StreamHandler())

subject_ids = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
snrs = [ 0.01, 0.05, 0.3]#,0.1, 0.15, 0.2,0.0001, 0.001, 0.4, 0.5, 0.6] 
n_i = [5,6]#[0,1,2,3,4] # next is [ 6] [7, 8, 9] 
done = itertools.product([0.1, 0.05,0.3], [0,1,3,4], subject_ids)#snrs , n_i, subject_ids, [factor])
i_max = len(snrs)*len(subject_ids)*len(n_i) 

def read_shuffle_write(snr,  n_i, subject_id, factor = 5):
    sesh_id =[n_i,snr,subject_id]
    gn_path = home+'/GN-WESAD'
    start_time = time()
    wesad_path = home+'/WESAD'
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
        except:
            logging.warn(f'{sesh_id} gn file not found at {load_gn_path}')
            return
        ws_df = pd.read_pickle(load_ws_path)
        logging.info(f'{sesh_id} ws unpickled {load_ws_path}')
        logging.info(f'{sesh_id} unpickling gn from {savePath}/S{subject_id}.pkl')
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
def main():
    start_time = time()
    factor = 5 #reducing the bigger (ECG) sampling rate by a factor of 5 snr,  n_i, subject_id
    combo = itertools.product(snrs , n_i, subject_ids, [factor]) # itertools.product(list(range(n_samples)), snrs, subject_ids, factor)
    logging.debug(f'combo: {list[combo]}')
    with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(read_shuffle_write, i[0], i[1], i[2], i[3]) for i in combo]
        done, not_done = concurrent.futures.wait(futures, return_when='ALL_COMPLETED')
        logging.debug(f'done assigning futures, now waiting for them to finish')
        for future in done:
            results = future.result()
    # we'll let each process write its own file
    #for savePath, df in results:
    #    pd.to_pickle(df, savePath)
    num_its = len(n_i)*len(snrs)*len(subject_ids)
    logging.info(f'finished {num_its} iterations in {round(time()-start_time,2)} seconds')
    logging.info(f'an average of {(time()-start_time)/num_its} seconds per iteration')
if __name__ == '__main__':
    main()