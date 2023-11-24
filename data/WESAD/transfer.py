#done = product ([ 0.01, 0.05, 0.3],[0,1,2,3,4],[0,1,3,4], subject_ids)#snrs , n_i, subject_ids, [factor])

import os
from datetime import datetime
from scipy import signal
from time import time
import concurrent.futures
import itertools
import logging
import pickle
now = datetime.now()
import shutil 
home = '' #windows version
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename= f'{home}logs/{now.strftime("%H_%M_%S")}.log' )
logging.getLogger().addHandler(logging.StreamHandler())

subject_ids = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
snrs = [0.6, 0.01, 0.05, 0.3,0.1, 0.15, 0.2,0.0001, 0.001, 0.4, 0.5, 0.6] 
n_i = [1,2,3]#,0,4,5,6, 7, 8, 9]#[] # next is [ 6] [7, 8, 9] 
tots = len(snrs)*len(subject_ids)*len(n_i)
done = itertools.product(snrs, n_i, subject_ids)#snrs , n_i, subject_ids, [factor])
from_path = 'C:/Users/alkurdi/Downloads/WESAD/GN-WESAD'
to_path = 'D:/Users/alkurdi/data/GN-WESAD'
print('from: ', os.listdir(from_path))
print('to: ', os.listdir(to_path),', in', to_path)

i = 0


def delete_empty_folders(root):
   for dirpath, dirnames, filenames in os.walk(root, topdown=False):
      for dirname in dirnames:
         full_path = os.path.join(dirpath, dirname)
         if not os.listdir(full_path): 
            os.rmdir(full_path)
delete_empty_folders(from_path)
#print(*list(done), sep='\n')
for snr, n_i, subject_id in done:
    print(f'{snr}, {n_i}, {subject_id}')
    sesh_path = '/n_'+str(n_i)+'/snr_'+str(snr)+'/S'+str(subject_id)
    feat_path = '/n_'+str(n_i)+'/snr_'+str(snr)+'/subject_feats'
    delete_empty_folders(from_path+sesh_path)
    delete_empty_folders(from_path+feat_path)
    delete_empty_folders(from_path+'/n_'+str(n_i)+'/snr_'+str(snr))
    try:
        for file in os.listdir(from_path+sesh_path):
            #print('file: ', file)
            try:
                shutil.move(f'{from_path}{sesh_path}/{file}',f'{to_path}{sesh_path}/{file}')
            except Exception as error:
                continue
                #print("BIG BROBLEM!!!!!! An exception occurred:", error)
    except:
        #print('empty dir', snr, n_i, subject_id)
        delete_empty_folders(from_path+sesh_path)
    
    #print(from_path+feat_path)
    try:
        for file in os.listdir(from_path+feat_path):
            #print('file: ', file)
            try:
                shutil.move(f'{from_path}{feat_path}/{file}',f'{to_path}{feat_path}/{file}')
            except Exception as error:
                #print("BIG BROBLEM!!!!!! An exception occurred:", error)
                continue
    except:
        #print('empty dir', snr, n_i, subject_id)
        delete_empty_folders(from_path+feat_path)
    try:
        os.mkdir(to_path+sesh_path)
    except:
        #print('dir exists\n')
        continue
        
    try:
        continue
        #print(f'from_path : {from_path}{sesh_path} that has {os.listdir(from_path+sesh_path)}\n')    
    except:
        continue
        #print(f'from_path : {from_path}{sesh_path} that has {os.listdir(from_path+sesh_path)}\n') 
        
    try:
        continue
        #print(f'to_path : {to_path}{sesh_path} that has {os.listdir(to_path+sesh_path)}\n')    
    except:
        #print(f'from_path : {from_path}{sesh_path} that has {os.listdir(from_path+sesh_path)}\n') 
        continue         
    
    #print(f'from: {from_path}{sesh_path}/S{subject_id}.pkl')
    #print(f'to: {to_path}{sesh_path}/S{subject_id}.pkl')
    try:
        shutil.move(f'{from_path}{feat_path}',f'{to_path}{feat_path}')
    except Exception as error:
        #print("BIG BROBLEM!!!!!! An exception occurred:", error) # An exception occurred: division by zero
        #print('from: ',from_path+feat_path)
        #print('to: ', to_path+feat_path)
        #print('from: ', os.listdir(from_path+feat_path))
        continue
        try:
            for file in os.listdir(from_path+feat_path):
                print('file: ', file)
                try:
                    shutil.move(f'{from_path}{feat_path}/{file}',f'{to_path}{feat_path}/{file}')
                except Exception as error:
                    #print("BIG BROBLEM!!!!!! An exception occurred:", error)
                    continue
        except Exception as error:
            #print("BIG BROBLEM!!!!!! An exception occurred:", error)
            continue
    try:
        for file in os.listdir(f'{from_path}{sesh_path}'):
            #print(f'from: {from_path}{sesh_path}/{file}')
            shutil.move(f'{from_path}{sesh_path}/file',f'{to_path}{sesh_path}/file')
    except:
        #print('fixedS does not exist')
        continue
    os.isfile(f'{from_path}{sesh_path}/fixed_resampled140hz_S{subject_id}.pkl')
    try:
        shutil.move(f'{from_path}{sesh_path}/fixed_resampled140hz_S{subject_id}.pkl',f'{to_path}{sesh_path}/fixed_resampled140hz_S{subject_id}.pkl')
    except:
        #print('fixed_resampled140hz_S does not exist')
        continue
    try:    
        shutil.move(f'{from_path}{sesh_path}/S{subject_id}.pkl',f'{to_path}{sesh_path}/S{subject_id}.pkl')
        print('Success, It nice. Boi.')
    except Exception as error:
        # handle the exception
        if os.path.isfile(f'{from_path}{sesh_path}/S{subject_id}.pkl') and os.path.isfile(f'{to_path}{sesh_path}/S{subject_id}.pkl'):
            #print('both files exist')
            #print('REMOVING FROM FROM_PATH')
            os.remove(f'{from_path}{sesh_path}/S{subject_id}.pkl')

        if not os.path.isfile(f'{from_path}{sesh_path}/S{subject_id}.pkl'):
            #print('from_file does not exist')
            #print("BIG BROBLEM!!!!!! An exception occurred:", error) # An exception occurred: division by zero
            continue
        elif not os.path.isfile(f'{to_path}{sesh_path}/S{subject_id}.pkl'):
            print('to file does not exist')
            #print("BIG BROBLEM!!!!!! An exception occurred:", error) # An exception occurred: division by zero
        elif not os.path.isdir(f'{to_path}{sesh_path}'):
            #print('to dir does not exist')
            #print("BIG BROBLEM!!!!!! An exception occurred:", error) # An exception occurred: division by zero
            continue
        elif not os.path.isdir(f'{from_path}{sesh_path}'):
            #print('from dir does not exist')
            #print("BIG BROBLEM!!!!!! An exception occurred:", error) # An exception occurred: division by zero
            continue
        else:
            #print("BIG BROBLEM!!!!!! An exception occurred:", error) # An exception occurred: division by zero
    
            delete_empty_folders(f'{from_path}{sesh_path}')
            delete_empty_folders(f'{to_path}{sesh_path}')
