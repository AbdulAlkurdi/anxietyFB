'''
This function reads the WESAD data, shuffles it, and writes it to a new file.
It is intended as a fix to the GN-WESAD data that was not created correctly following WESAD
format. 
'''
import os
from datetime import datetime
from time import time
import concurrent.futures
import itertools
import pandas as pd
import logging
import pickle
now = datetime.now()


#home = 'D:/Users/alkurdi/data' #windows version
#home = '/mnt/d/Users/alkurdi/data'
home = '/mnt/c/Users/alkurdi/Desktop/Vansh/data'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    filename= f'{home}/{now.strftime("%Y-%M-%d-%H-%M-%S")}_{os.path.basename(__file__)[0:-3]}.log' )
#logging.getLogger().addHandler(logging.StreamHandler())
logging.info('besmellah main')

subject_ids = [2, 3, 4, 5, 6]#, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17] #[2]#
snrs = [0.15,.2]#[0.4,0.1,0.01,0.15,0.05,0.3,0.2,0.0001,0.001,0.5,0.6]  #[0.6]#0.6, 0.01,0.4]#,
n_i = [5]#[0,1,2,3,4,5,6,7,8,9] #done[5] #[0]#
i_max = len(snrs)*len(subject_ids)*len(n_i)
#print(list(done))
subject_data = {}
#wesad_path =  'C:/Users/alkurdi/Downloads/WESAD'
wesad_path = '/mnt/c/Users/alkurdi/Downloads/WESAD'
logging.info(' starting to load all wesad dataset')
for subject in subject_ids:
    load_ws_path = wesad_path + '/S'+str(subject) + '/S'+str(subject)+'.pkl'
    subject_data[subject] = pd.read_pickle(load_ws_path)
logging.info(' loaded all wesad dataset')

def read_shuffle_write(snr,  n_i, subject_id, factor ):
    """
    Reads data from a file, shuffles and organizes it, and then writes the organized data to a new file.
    It's needed because the GN-WESAD data was not created correctly following WESAD format. sorry :(
    Args:
        snr (int): Signal-to-noise ratio.
        n_i (int): Number of iterations.
        subject_id (int): Subject ID.
        factor (int, optional): Factor for shuffling. Defaults to 5.

    Returns:
        None
    """
    x = factor/2
    x = int(x)
    sesh_id = [n_i, snr, subject_id]
    logging.info('%s', sesh_id)
    gn_path = home+'/GN-WESAD'
    msg = 'starting  n_i: %s; snr: %s, id: %s. iteration' % (n_i, snr, subject_id)
    logging.info(msg)
    sesh_path = f'/n_{n_i}/snr_{snr}/S{subject_id}'
    load_gn_path = f'{gn_path}{sesh_path}/S{subject_id}.pkl'
    save_path = f'{gn_path}{sesh_path}/organized_S{subject_id}.pkl'
    if os.path.exists(f'{save_path}') and False:
        msg = save_path+' n_i: '+ str(n_i)+ '; snr: '+ str(snr)+ '; subject_id: '+str(subject_id)+ ' already exist.'
        logging.info(msg)
    else:
        logging.info('n_i: %s, snr: %s, subject_id: %s does not exist...but FEAR NOT!, we will create it now', n_i, snr, subject_id)    
        try: 
            gn_df = pd.read_pickle(load_gn_path)
        except FileNotFoundError:
            logging.warning('%s gn file not found at %s', sesh_id, load_gn_path)
            return
        ws_df = subject_data[subject_id]
        logging.info('%s unpickling gn from %s/S%d.pkl', sesh_id, load_gn_path, subject_id)
        logging.debug('%s gn unpickled %s', sesh_id, gn_df.keys())
        fixed_gn_df = {}
        fixed_gn_df['signal'] = {}
        wrist = {}
        chest = {}
        logging.debug('%s resampling and playing with chest', sesh_id)
        fixed_gn_df['label'] = ws_df['label']  # signal.resample(ws_df['label'], new_len)
        chest['ACC'] = gn_df['ACC_C']  # gn_df['ACC'] #= #
        chest['ECG'] = gn_df['ECG']  # = gn_df['ECG']
        chest['EMG'] = gn_df['EMG_C']  # = signal.resample(gn_df['EMG_C'], int(new_len/100)+1)#
        chest['EDA'] = gn_df['EDA_C']  # signal.resample(gn_df['EDA_C'], int(new_len/100)+1)#
        chest['Temp'] = gn_df['Temp_C']  # = signal.resample(gn_df['Temp_C'], int(new_len/100)+1)#
        chest['Resp'] = gn_df['Resp_C']  # signal.resample(gn_df['Resp_C'], new_len)#
        logging.debug('%s resampling and playing with wrist', sesh_id)
        wrist['ACC'] = gn_df['ACC']  # =signal.resample(gn_df['ACC'], new_len) #
        wrist['BVP'] = gn_df['BVP'] # signal.resample(gn_df['BVP'],new_len) #= 
        wrist['EDA'] = gn_df['EDA'] #signal.resample(gn_df['EDA'],new_len) #= 
        wrist['TEMP'] = gn_df['TEMP'] #= signal.resample(gn_df['TEMP'],new_len) # 
        logging.debug('%s combining and saving', sesh_id)
        fixed_gn_df['signal']['chest'] = chest 
        fixed_gn_df['signal']['wrist'] = wrist
        fixed_gn_df['subject'] = ws_df['subject']
        logging.info('%s...starting to pkl', sesh_id)
        start_time_inner = time()  # Define start_time_inner variable
        with open(f'{save_path}', 'wb') as f:
            pickle.dump(fixed_gn_df, f)
        logging.info('%s...finished %s %s %s iteration in %s seconds', sesh_id, n_i, snr, subject_id, round(time()-start_time_inner,2))
        #return [save_path, fixed_gn_df] #} {snr} {subject_id}  iterations in ', time()-start_time, 'seconds'
logging.debug('done defining read_shuffle_write')

if __name__ == '__main__':
    start_time = time()
    factor = 1 
    combo = itertools.product(snrs, n_i, subject_ids, [factor])
    logging.info('besmellah main')
    with concurrent.futures.ProcessPoolExecutor(max_workers=12) as executor:
        futures = [executor.submit(read_shuffle_write, i[0], i[1], i[2], i[3]) for i in combo]
        logging.debug('left futures')
        done, not_done = concurrent.futures.wait(futures, return_when='ALL_COMPLETED')
        logging.debug('done assigning futures, now waiting for them to finish')
        for future in done:
            future.result()

    num_its = len(n_i) * len(snrs) * len(subject_ids)
    logging.info('finished %s iterations in %s seconds', num_its, round(time()-start_time,2))
    logging.info('an average of %s seconds per iteration', (time()-start_time)/num_its)

logging.info(' all done')
