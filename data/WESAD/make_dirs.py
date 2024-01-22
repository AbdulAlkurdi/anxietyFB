import os
#savePath = 'C:/Users/alkurdi/Downloads/WESAD/GN-WESAD'
savePath = 'D:/Users/alkurdi/data/GN-WESAD'
n_samples = [0,1,2,3,4, 5,6, 7, 8 , 9]#
subject_ids = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
snrs = [ 0.01, 0.05, 0.3,0.1, 0.15, 0.2, 0.4, 0.5, 0.6, 0.001, 0.0001]
for n_i in n_samples:
    if not os.path.exists(savePath + '/n_'+str(n_i)):
        os.makedirs(savePath + '/n_'+str(n_i))
    for snr in snrs:
        if not os.path.exists(savePath + '/n_'+str(n_i)+'/snr_'+str(snr)):
            os.makedirs(savePath + '/n_'+str(n_i)+'/snr_'+str(snr))
        if not os.path.exists(savePath + '/n_'+str(n_i)+'/snr_'+str(snr)+'/subject_feats'):
            os.makedirs(savePath + '/n_'+str(n_i)+'/snr_'+str(snr)+'/subject_feats')
        if not os.path.exists(savePath + '/n_'+str(n_i)+'/subject_feats'):
            os.makedirs(savePath + '/n_'+str(n_i)+'/subject_feats')
        '''
        for subject_id in subject_ids:
            if not os.path.exists(savePath + '/n_'+str(n_i)+'/snr_'+str(snr)+ '/S'+str(subject_id)):
                os.makedirs(savePath + '/n_'+str(n_i)+'/snr_'+str(snr)+ '/S'+str(subject_id))
        '''    