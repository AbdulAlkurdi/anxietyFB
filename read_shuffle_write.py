import os
#import time
from datetime import datetime
from scipy import signal
from time import time
import concurrent.futures
import itertools
import pandas as pd
import logging
import pickle
now = datetime.now()
home= '/mnt/c/Users/alkurdi/Desktop/Vansh'
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', filename= f'{home}/logs/{now.strftime("%H_%M_%S")}.log' )

subject_ids = [17]#, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
snrs = [ 0.01]#, 0.05,  0.3,0.1, 0.15, 0.2,0.0001, 0.001, 0.4, 0.5, 0.6] # this is what we ran #0.00001,
n_samples = 1 #10

i_max = len(snrs)*len(subject_ids)*n_samples

def read_shuffle_write(n_i, snr, subject_id):
    sesh_id =[n_i,snr,subject_id]
    gn_path = '/mnt/d/Users/alkurdi/data/GN-WESAD'
    start_time = time()
    wesad_path = '/mnt/c/Users/alkurdi/Desktop/Vansh/data/WESAD'
    msg = f'starting  n_i: {n_i}; snr: {snr}, id: {subject_id}. iteration'
    logging.info(msg)
    sesh_path = '/n_'+str(n_i)+'/snr_'+str(snr)+'/S'+str(subject_id)
    load_ws_path = wesad_path + '/S'+str(subject_id) + '/S'+str(subject_id)+'.pkl'
    load_gn_path =  gn_path + sesh_path + '/S'+str(subject_id)+'.pkl'
    
    savePath = gn_path + sesh_path + f'/fixed_resampled170hz64_S{subject_id}.pkl'
    if False: #os.path.exists(f'{savePath}'):
        msg = 'n_i: '+ str(n_i)+ '; snr: '+ str(snr)+ '; subject_id: '+str(subject_id)+ ' already exist.'
        logging.info(msg)
        #return 
    else:
        logging.info(f'n_i: {n_i}, snr: {snr}, subject_id: {subject_id} does not exist')
        logging.info('but fear not, we will create it now')    
        logging.info(f'{sesh_id} unpickling ws {load_ws_path}')
        #with open(f'{loadPath}/S{subject_id}/S{subject_id}.pkl', 'rb') as f:
        #    ws_df = pickle.load(f)
        try: 
            gn_df = pd.read_pickle(load_gn_path)
        except:
            logging.info(f'{sesh_id} gn file not found at {load_gn_path}')
            return
        ws_df = pd.read_pickle(load_ws_path)
        logging.info(f'{sesh_id} ws unpickled {load_ws_path}')
        logging.info(f'{sesh_id} unpickling gn from {savePath}/S{subject_id}.pkl')
        
        
        
        logging.info(f'{sesh_id} gn unpickled {gn_df.keys()}')

        fixed_gn_df = {}; fixed_gn_df['signal'] = {}
        wrist = {}; chest = {}
        new_len = int(len(ws_df['signal']['chest']['ECG'])/5)
        logging.info(f'{sesh_id} resampling and playing with chest')
        fixed_gn_df['label'] = signal.resample(ws_df['label'],new_len)
        chest['ACC'] = signal.resample(gn_df['ACC'],new_len) #= gn_df['ACC']
        chest['ECG'] = signal.resample(gn_df['ECG'],new_len) #= gn_df['ECG']
        chest['EMG'] = signal.resample(gn_df['EMG_C'],new_len)#= gn_df['EMG_C'] 
        chest['EDA'] = signal.resample(gn_df['EDA_C'],new_len)#= gn_df['EDA_C']
        chest['Temp'] = signal.resample(gn_df['Temp_C'],new_len)#= gn_df['Temp_C']
        chest['Resp'] = signal.resample(gn_df['Resp_C'],new_len)#= gn_df['Resp_C']
        logging.info(f'{sesh_id} resampling and playing with wrist')
        wrist['ACC'] = signal.resample(gn_df['ACC'],new_len) #= gn_df['ACC']
        wrist['BVP'] = signal.resample(gn_df['BVP'],new_len) #= gn_df['BVP']
        wrist['EDA'] = signal.resample(gn_df['EDA'],new_len) #= gn_df['EDA']
        wrist['TEMP'] = signal.resample(gn_df['TEMP'],new_len) #= gn_df['TEMP'] 
        logging.info(f'{sesh_id} combining and saving')
        print('chest', chest)
        print('wrist', wrist)
        fixed_gn_df['signal']['chest'] = chest 
        fixed_gn_df['signal']['wrist'] = wrist
        fixed_gn_df['subject'] = ws_df['subject']
        print('fixed gn keys', fixed_gn_df.keys())
        print('fixed gn signal keys', fixed_gn_df['signal'].keys())
        #return 
        logging.info(f'{sesh_id}...starting to pkl')
        with open(f'{savePath}', 'wb') as f:
            pickle.dump(fixed_gn_df, f)
        
        logging.info(f'{sesh_id}...finished {n_i} {snr} {subject_id}  iterations in {time()-start_time} seconds')
        return [savePath, fixed_gn_df] #} {snr} {subject_id}  iterations in ', time()-start_time, 'seconds'
def main():
    combo = itertools.product(list(range(n_samples)), snrs, subject_ids)
    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
        futures = [executor.submit(read_shuffle_write, i[0], i[1], i[2]) for i in combo]
        done, not_done = concurrent.futures.wait(futures, return_when='ALL_COMPLETED')
        for future in done:
            results = future.result()
    #for savePath, df in results:
    #    pd.to_pickle(df, savePath)
if __name__ == '__main__':
    main()

        
