import pandas as pd
import numpy as np
import json
import pickle
from scipy.io import wavfile
import os


def read_hx(participant_day_filepath, date):
    '''
    takes in the participant filepath and the date of the data. it reads ecg, br, accx, accy, accz,    
    fhex = 'record_265679'
    '''
    # read raw ECG file; ECG_I.wav only
    # change wavefile pathway
    path = participant_day_filepath 

    raw_ECG = wavfile.read(path+'/ECG_I.wav')
    #settings = {}
    #settings['fs'] = 256 # sampling rate

    # ECG
    raw_ECG = pd.DataFrame(data = raw_ECG[1])
    ecg = 0.0064 * raw_ECG #get correct magnitude of ECG
    ecg.rename(columns = {0: 'ECG'}, inplace = True)
    # Opening JSON file and return it as dictionary
    # change file pathway
    f = open(path+'/info.json')
    date_info = json.load(f)

    # BR 
    raw_br = wavfile.read(path+'/breathing_rate.wav')
    raw_br = pd.DataFrame(data = raw_br[1])
    br = 1.0000 * raw_br
    br.rename(columns = {0: 'breathing_rate'}, inplace = True)

    # acc x
    raw_accX = wavfile.read(path+'/acceleration_X.wav')
    raw_accX = pd.DataFrame(data = raw_accX[1])
    accx = 1.0000 * raw_accX 
    accx.rename(columns = {0: 'Acc_X'}, inplace = True)
    # acc y
    raw_accY = wavfile.read(path+'/acceleration_Y.wav')
    raw_accY = pd.DataFrame(data = raw_accY[1])
    accy = 1.0000 * raw_accY 
    accy.rename(columns = {0: 'Acc_Y'}, inplace = True)
    # acc z
    raw_accZ = wavfile.read(path+'/acceleration_Z.wav')
    raw_accZ = pd.DataFrame(data = raw_accZ[1])
    accz = 1.0000 * raw_accZ 
    accz.rename(columns = {0: 'Acc_Z'}, inplace = True)


    # Add timestamp to Hex (ECG & ACC) signal
    t0_ecg = list(date_info.values())[0]/256
    ecg['Timestamp'] = list(range(0, len(raw_ECG),1))
    ecg['Timestamp'] = ecg['Timestamp'].apply(lambda x: x/256+t0_ecg)
    #ecg['Timestamp'] = ecg['Timestamp'].str.get(0)
    ecg['Second'] = ecg['Timestamp']
    ecg = ecg.set_index('Timestamp')
    ecg['Second'] = ecg['Second'].apply(lambda x: x-ecg.index[0])
    ecg = ecg.reset_index()
    #ecg.columns = ['Heart rate', 'Second']

    t0_br = list(date_info.values())[0]/256
    br['Timestamp'] = list(range(0, len(raw_br),1))
    br['Timestamp'] = br['Timestamp'].apply(lambda x: x/1+t0_br)
    br['Second'] = br['Timestamp']
    br = br.set_index('Timestamp')
    br['Second'] = br['Second'].apply(lambda x: x-br.index[0])
    br = br.reset_index()

    t0_acc = list(date_info.values())[0]/256
    accx['Timestamp'] = list(range(0, len(raw_accX),1))
    accx['Timestamp'] = accx['Timestamp'].apply(lambda x: x/64+t0_acc)
    accx['Second'] = accx['Timestamp']
    accx = accx.set_index('Timestamp')
    accx['Second'] = accx['Second'].apply(lambda x: x-accx.index[0])
    accx = accx.reset_index()

    accy['Timestamp'] = list(range(0, len(raw_accY),1))
    accy['Timestamp'] = accy['Timestamp'].apply(lambda x: x/64+t0_acc)
    accy['Second'] = accy['Timestamp']
    accy = accy.set_index('Timestamp')
    accy['Second'] = accy['Second'].apply(lambda x: x-accy.index[0])
    accy = accy.reset_index()

    accz['Timestamp'] = list(range(0, len(raw_accZ),1))
    accz['Timestamp'] = accz['Timestamp'].apply(lambda x: x/64+t0_acc)
    accz['Second'] = accz['Timestamp']
    accz = accz.set_index('Timestamp')
    accz['Second'] = accz['Second'].apply(lambda x: x-accz.index[0])
    accz = accz.reset_index()
    
    data_dict = {'Date':date, 'ECG':ecg, 'BR':br, 'BR':br, 'accx':accx, 'accy':accy,'accz':accz}
    return data_dict

def read_E4(participant_filepath, date):
    '''
    usage:
        filepath=r'/home/maxinehe/Downloads/' + fE4
        a=read_E4(filepath, '230429')
        fE4 = 'A04BA8_230429-142458'
        fe4 = '230429-142458'
        
    '''
    filepath = participant_filepath
    # HR data -- started 10 seconds later than other metrics
    hr = pd.read_csv(filepath+str('/HR.csv'), header = None)
    # clean up HR file
    start_time = hr.values[0]
    hr_samp_rate = hr.values[1]
    hr = hr.drop(labels = [0, 1], axis = 0, inplace = False)
    hr['Timestamp'] = list(range(0, len(hr),1))
    hr['Timestamp'] = hr['Timestamp'].apply(lambda x: x/hr_samp_rate+start_time)
    hr['Timestamp'] = hr['Timestamp'].str.get(0)
    hr['Second'] = hr['Timestamp']
    hr = hr.set_index('Timestamp')
    hr['Second'] = hr['Second'].apply(lambda x: x-hr.index[0])
    hr.columns = ['Heart rate', 'Second']
    hr = hr.reset_index(inplace=False)
    
    # EDA data
    eda = pd.read_csv(filepath+str('/EDA.csv'), header = None)
    start_time = eda.values[0]
    eda_samp_rate = eda.values[1]
    eda = eda.drop(labels = [0, 1], axis = 0, inplace = False)
    eda['Timestamp'] = list(range(0, len(eda),1))
    eda['Timestamp'] = eda['Timestamp'].apply(lambda x: x/eda_samp_rate+start_time)
    eda['Timestamp'] = eda['Timestamp'].str.get(0)
    eda['Second'] = eda['Timestamp']
    eda = eda.set_index('Timestamp')
    eda['Second'] = eda['Second'].apply(lambda x: x-eda.index[0])
    eda.columns = ['EDA', 'Second']
    eda = eda.reset_index(inplace=False)

    temp = pd.read_csv(filepath+str('/TEMP.csv'), header = None)
    # clean up TEMP file
    start_time = temp.values[0]
    temp_samp_rate = temp.values[1]
    temp = temp.drop(labels = [0, 1], axis = 0, inplace = False)
    temp['Timestamp'] = list(range(0, len(temp),1))
    temp['Timestamp'] = temp['Timestamp'].apply(lambda x: x/temp_samp_rate+start_time)
    temp['Timestamp'] = temp['Timestamp'].str.get(0)
    temp['Second'] = temp['Timestamp']
    temp = temp.set_index('Timestamp')
    temp['Second'] = temp['Second'].apply(lambda x: x-temp.index[0])
    temp.columns = ['Temp', 'Second']
    temp = temp.reset_index(inplace=False)
    
    try:
        ibi = pd.read_csv(filepath+str('/IBI.csv'), header = None) # no correction of timestamp needed
        ibi = ibi.drop(labels = [0, 1], axis = 0, inplace = False)
        ibi.columns = ['Second', 'IBI']
    except:
        print('no IBI data available')
        ibi = pd.DataFrame()
    
    bvp = pd.read_csv(filepath+str('/BVP.csv'), header = None)
    start_time = bvp.values[0]
    bvp_samp_rate = bvp.values[1]
    bvp = bvp.drop(labels = [0, 1], axis = 0, inplace = False)
    bvp['Timestamp'] = list(range(0, len(bvp),1))
    bvp['Timestamp'] = bvp['Timestamp'].apply(lambda x: np.round(x/bvp_samp_rate+start_time, 2))
    bvp['Timestamp'] = bvp['Timestamp'].str.get(0)
    bvp['Second'] = bvp['Timestamp']
    bvp = bvp.set_index('Timestamp')
    bvp['Second'] = bvp['Second'].apply(lambda x: x-bvp.index[0])
    bvp.columns = ['BVP', 'Second']
    bvp = bvp.reset_index(inplace=False)
    
    acc = pd.read_csv(filepath+str('/ACC.csv'), header = None)
    start_time = acc.values[0,0]
    acc_samp_rate = acc.values[1,0]
    acc = acc.drop(labels = [0, 1], axis = 0, inplace = False)
    acc['Timestamp'] = list(range(0, len(acc),1))
    acc['Timestamp'] = acc['Timestamp'].apply(lambda x: x/acc_samp_rate+start_time)
    acc['Second'] = acc['Timestamp']
    acc = acc.set_index('Timestamp')
    acc['Second'] = acc['Second'].apply(lambda x: x-acc.index[0])
    acc.columns = ['Acceleration_X','Acceleration_Y','Acceleration_Z','Second']
    acc = acc.reset_index(inplace=False)
    
    data_dict = {'Date':date, 'HR':hr, 'EDA':eda, 'TEMP':temp, 'IBI':ibi, 'BVP':bvp,'ACC':acc}
    return data_dict 

def E4sync_offset(a):

    eda = a['EDA']
    temp = a['TEMP']
    bvp = a['BVP']
    acc = a['ACC']

    
    t_eda = eda['Timestamp'].iat[-1]
    t_temp = temp['Timestamp'].iat[-1]
    t_bvp = bvp['Timestamp'].iat[-1]
    t_acc = acc['Timestamp'].iat[-1]
    t0 = 0

    print(t_eda < t_temp and t_eda < t_bvp and t_eda < t_acc)
    print(t_temp < t_eda and t_temp < t_bvp and t_temp < t_acc)
    print(t_bvp < t_eda and t_bvp < t_temp and t_bvp < t_acc)
    print(t_acc < t_eda and t_acc < t_temp and t_acc < t_bvp)

    if t_eda < t_temp and t_eda < t_bvp and t_eda < t_acc :
        print(1)
        t1_loc = temp.loc[temp['Timestamp'] == round(t_eda, 2)]
        t2_loc = bvp.loc[bvp['Timestamp'] == round(t_eda, 2)]
        t3_loc = acc.loc[acc['Timestamp'] == round(t_eda, 2)]
        offset_eda = eda
        offset_temp = temp.truncate(before = t0, after = t1_loc.index[0])
        offset_bvp = bvp.truncate(before = t0, after = t2_loc.index[0])
        offset_acc = acc.truncate(before = t0, after = t3_loc.index[0])
        
    elif t_temp < t_eda and t_temp < t_bvp and t_temp < t_acc :
        print(2)
        t1_loc = eda.loc[eda['Timestamp'] == round(t_temp, 0)]
        t2_loc = bvp.loc[bvp['Timestamp'] == round(t_temp, 2)]
        t3_loc = acc.loc[acc['Timestamp'] == round(t_temp, 2)]
        offset_eda = eda.truncate(before = t0, after = t1_loc.index[0])
        offset_temp = temp
        offset_bvp = bvp.truncate(before = t0, after = t2_loc.index[0])
        offset_acc = acc.truncate(before = t0, after = t3_loc.index[0])
        
    elif t_bvp < t_eda and t_bvp < t_temp and t_bvp < t_acc :
        print(3)
        t1_loc = eda.loc[eda['Timestamp'] == round(t_bvp, 0)]
        t2_loc = temp.loc[temp['Timestamp'] == round(t_bvp, 0)]
        t3_loc = acc.loc[acc['Timestamp'] ==round(t_bvp, 2)]
        offset_eda = eda.truncate(before = t0, after = t1_loc.index[0])
        offset_temp = temp.truncate(before = t0, after = t2_loc.index[0])
        offset_bvp = bvp
        offset_acc = acc.truncate(before = t0, after = t3_loc.index[0])
    
    elif t_acc < t_eda and t_acc < t_temp and t_acc < t_bvp :
        print(4)
        t1_loc = eda.loc[eda['Timestamp'] == round(t_acc, 0)]
        t2_loc = temp.loc[temp['Timestamp'] == round(t_acc, 0)]
        t3_loc = bvp.loc[bvp['Timestamp'] == round(t_acc, 2)]
        offset_eda = eda.truncate(before = t0, after = t1_loc.index[0])
        offset_temp = temp.truncate(before = t0, after = t2_loc.index[0])
        offset_bvp = bvp.truncate(before = t0, after = t3_loc.index[0])
        offset_acc = acc
    
    # E4_acc separation
    l1 = offset_acc.iloc[:,0:1]
    l2 = offset_acc.iloc[:,4]

    Acceleration_x = offset_acc.iloc[:,1]
    Acceleration_X1 = pd.concat([l1,Acceleration_x], axis=1, join='outer')
    Acceleration_X = pd.concat([Acceleration_X1,l2], axis=1, join='outer')

    Acceleration_y = offset_acc.iloc[:,2]
    Acceleration_Y1 = pd.concat([l1,Acceleration_y], axis=1, join='outer')
    Acceleration_Y = pd.concat([Acceleration_Y1,l2], axis=1, join='outer')

    Acceleration_z = offset_acc.iloc[:,3]
    Acceleration_Z1 = pd.concat([l1,Acceleration_z], axis=1, join='outer')
    Acceleration_Z = pd.concat([Acceleration_Z1,l2], axis=1, join='outer')



    eda2 = offset_eda.iloc[:,1]
    eda2 = np.expand_dims(eda2.values, axis = 1)
    temp2 = offset_temp.iloc[:,1]
    temp2 = np.expand_dims(temp2.values, axis = 1)
    bvp2 = offset_bvp.iloc[:,1]
    bvp2 = np.expand_dims(bvp2.values, axis = 1)
    Accx = Acceleration_X.iloc[:,1]
    Accx = np.expand_dims(Accx.values, axis = 1)
    Accy = Acceleration_Y.iloc[:,1]
    Accy = np.expand_dims(Accy.values, axis = 1)
    Accz = Acceleration_Z.iloc[:,1]
    Accz = np.expand_dims(Accz.values, axis = 1)

    acc2 = offset_acc.iloc[:,1:4]
    acc2 = acc2.values

    E4_to_dic = {}
    E4_to_dic["EDA"] = offset_eda
    E4_to_dic["TEMP"] = offset_temp
    E4_to_dic["BVP"] = offset_bvp
    #E4_to_dic["Acceleration_X"] = Accx
    #E4_to_dic["Acceleration_Y"] = Accy
    #E4_to_dic["Acceleration_Z"] = Accz 
    E4_to_dic["ACC"] = offset_acc
    
    if False:
        with open(filepath+'_E4.pkl', 'wb') as handle:
            pickle.dump(E4_to_dic, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return E4_to_dic

def Hexsync_offset(ecg,br,accx,accy, accz):
    t_ecg = ecg['Timestamp'].iat[-1]
    t_br = br['Timestamp'].iat[-1]
    t_accx = accx['Timestamp'].iat[-1]
    t_accy = accy['Timestamp'].iat[-1]
    t0 = 0
    

    if t_ecg <= t_br and t_ecg <= t_accx and t_ecg<=t_accy :
        t1_loc = br.loc[round(br['Timestamp'],0) == round(t_ecg, 0)].head(1)
        t2_loc = accx.loc[round(accx['Timestamp'],0) == round(t_ecg, 0)].head(1)
        t3_loc = accy.loc[round(accy['Timestamp'],0) == round(t_ecg, 0)].head(1)
        offset_ecg = ecg
        offset_br = br.truncate(before = t0, after = t1_loc.index[0])
        offset_accx = accx.truncate(before = t0, after = t2_loc.index[0])
        offset_accy = accy.truncate(before = t0, after = t3_loc.index[0])
        offset_accz = accz.truncate(before = t0, after = t3_loc.index[0])

    elif t_br <= t_ecg and t_br <= t_accx and t_br<=t_accy :
        t1_loc = ecg.loc[ecg['Timestamp'] == round(t_br, 7)]
        t2_loc = accx.loc[accx['Timestamp'] == round(t_br, 7)]
        t3_loc = accy.loc[accy['Timestamp'] == round(t_br, 7)]
        offset_ecg = ecg.truncate(before = t0, after = t1_loc.index[0])
        offset_br = br
        offset_accx = accx.truncate(before = t0, after = t2_loc.index[0])
        offset_accy= accy.truncate(before = t0, after = t3_loc.index[0])
        offset_accz= accz.truncate(before = t0, after = t3_loc.index[0])

    elif t_accx <= t_ecg and t_accx <= t_br and t_accx <= t_accy :
        t1_loc = ecg.loc[ecg['Timestamp'] == round(t_accx, 7)]
        t2_loc = br.loc[br['Timestamp'] == round(t_accx, 7)]
        t3_loc = accy.loc[accy['Timestamp'] ==round(t_accx, 7)]
        offset_ecg = ecg.truncate(before = t0, after = t1_loc.index[0])
        offset_br = br.truncate(before = t0, after = t2_loc.index[0])
        offset_accx = accx
        offset_accy = accy.truncate(before = t0, after = t3_loc.index[0])
        offset_accz = accz.truncate(before = t0, after = t3_loc.index[0])

    elif t_accy <= t_ecg and t_accy <= t_br and t_accy <= t_accx :
        t1_loc = ecg.loc[ecg['Timestamp'] == round(t_accy, 7)]
        t2_loc = br.loc[br['Timestamp'] == round(t_accy, 7)]
        t3_loc = accx.loc[accx['Timestamp'] == round(t_accy, 7)]
        offset_ecg = ecg.truncate(before = t0, after = t1_loc.index[0])
        offset_br = br.truncate(before = t0, after = t2_loc.index[0])
        offset_accx = accx.truncate(before = t0, after = t3_loc.index[0])
        offset_accy = accy
        offset_accz = accz

    ecg2 = offset_ecg.iloc[:,1]
    ecg2 = np.expand_dims(ecg2.values, axis = 1)
    accx2 = offset_accx.iloc[:,1]
    accx2 = np.expand_dims(accx2.values, axis = 1)
    accy2 = offset_accy.iloc[:,1]
    accy2 = np.expand_dims(accy2.values, axis = 1)
    accz2 = offset_accz.iloc[:,1]
    accz2 = np.expand_dims(accz2.values, axis = 1)
    br2 = offset_br.iloc[:,1]
    br2 = np.expand_dims(br2.values, axis = 1)

    hx_to_dic = {}
    hx_to_dic["ECG"] = offset_ecg
    hx_to_dic["BR"] = offset_br
    hx_to_dic["ACCX"] = offset_accx
    hx_to_dic["ACCY"] = offset_accy
    hx_to_dic["ACCZ"] = offset_accz
 
    
    
    accx3 = accx.iloc[:,0:2]
    accy3 = accy.iloc[:,1]
    accz3 = accz.iloc[:,1:3]
    hx_to_dic['ACC_n'] = pd.concat([accx3,
                                    accy3,
                                    accz3],
                                    axis=1, join='outer')
    #hx_to_dic['ACC_n'] = np.concatenate([accx2, accy2, accz2], axis=1, join='outer')
    
    
    
    if False:
        string2 = fhex+'_hex.pkl'
        with open(string2, 'wb') as handle:
            pickle.dump(hx_to_dic, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    
    return hx_to_dic
    
def doublesync_offset(ecg, bvp):
    
    t1_ecg = ecg['Timestamp'].iat[-1]
    t1_bvp = bvp['Timestamp'].iat[-1]
    t0_bvp = bvp['Timestamp'].iat[0]
    t0_ecg = ecg['Timestamp'].iat[0]

    if t0_ecg < t0_bvp and t1_ecg < t1_bvp:
        
        t0_loc = ecg.loc[round(ecg['Timestamp'],1) == round(t0_bvp, 1)].head(1)
        t1_loc = bvp.loc[round(bvp['Timestamp'],1) == round(t1_ecg, 1)].head(1)
        offset_bvp = bvp.truncate(after = t1_loc.index[0])
        offset_ecg = ecg.truncate(before = t0_loc.index[0])
        
        
    elif t0_ecg > t0_bvp and t1_ecg > t1_bvp:
        
        t0_loc = bvp.loc[round(bvp['Timestamp'],1) == round(t0_ecg, 1)].head(1)
        t1_loc = ecg.loc[round(ecg['Timestamp'],1) == round(t1_bvp, 1)].head(1)
        offset_bvp = bvp.truncate(before = t0_loc.index[0])
        offset_ecg = ecg.truncate(after = t1_loc.index[0])
        
    elif t0_ecg < t0_bvp and t1_ecg > t1_bvp:
        
        t0_loc = ecg.loc[round(ecg['Timestamp'],1) == round(t0_bvp, 1)].head(1)
        t1_loc = ecg.loc[round(ecg['Timestamp'],1) == round(t1_bvp, 1)].head(1)
        offset_bvp = bvp
        offset_ecg = ecg.truncate(before = t0_loc.index[0], after = t1_loc.index[0])
    
    elif t0_ecg > t0_bvp and t1_ecg < t1_bvp:
        
        t0_loc = bvp.loc[round(bvp['Timestamp'],1) == round(t0_ecg, 1)].head(1)
        t1_loc = bvp.loc[round(bvp['Timestamp'],1) == round(t1_ecg, 1)].head(1)
        offset_bvp = bvp.truncate(before = t0_loc.index[0], after = t1_loc.index[0])
        offset_ecg = ecg
    
        #offset = {'offset_ecg':offset_ecg, 'offset_bvp':offset_bvp}
        #return offset
    
        
    ecg2 = offset_ecg.iloc[:,1]
    ecg2 = np.expand_dims(ecg2.values, axis = 1)
    bvp2 = offset_bvp.iloc[:,1]
    bvp2 = np.expand_dims(bvp2.values, axis = 1)

    E4_to_dic = {}
    E4_to_dic["ECG"] = ecg2
    E4_to_dic["BVP"] = bvp2

    if False:
        string3 = fhex+'_ECG&BVP.pkl'
        with open(string3, 'wb') as handle:
            pickle.dump(E4_to_dic, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return offset_ecg,offset_bvp

def accsync_offset(ecg, bvp):
    
    t1_ecg = ecg['Timestamp'].iat[-1]
    t1_bvp = bvp['Timestamp'].iat[-1]
    t0_bvp = bvp['Timestamp'].iat[0]
    t0_ecg = ecg['Timestamp'].iat[0]

    if t0_ecg < t0_bvp and t1_ecg < t1_bvp:
        
        t0_loc = ecg.loc[round(ecg['Timestamp'],1) == round(t0_bvp, 1)].head(1)
        t1_loc = bvp.loc[round(bvp['Timestamp'],1) == round(t1_ecg, 1)].head(1)

        offset_bvp = bvp.truncate(after = t1_loc.index[0])
        offset_ecg = ecg.truncate(before = t0_loc.index[0])
        
    elif t0_ecg > t0_bvp and t1_ecg > t1_bvp:

        t0_loc = bvp.loc[round(bvp['Timestamp'],1) == round(t0_ecg,1)].head(1)
        t1_loc = ecg.loc[round(ecg['Timestamp'],1) == round(t1_bvp,1)].head(1)

        offset_bvp = bvp.truncate(before = t0_loc.index[0])
        offset_ecg = ecg.truncate(after = t1_loc.index[0])
        
    elif t0_ecg < t0_bvp and t1_ecg > t1_bvp:
        
        t0_loc = ecg.loc[round(ecg['Timestamp'],1) == round(t0_bvp, 1)].head(1)
        t1_loc = ecg.loc[round(ecg['Timestamp'],1) == round(t1_bvp, 1)].head(1)
        
        offset_bvp = bvp
        offset_ecg = ecg.truncate(before = t0_loc.index[0], after = t1_loc.index[0])
    
    elif t0_ecg > t0_bvp and t1_ecg < t1_bvp:
        
        t0_loc = bvp.loc[round(bvp['Timestamp'],1) == round(t0_ecg, 1)].head(1)
        t1_loc = bvp.loc[round(bvp['Timestamp'],1) == round(t1_ecg, 1)].head(1)
        
        offset_bvp = bvp.truncate(before = t0_loc.index[0], after = t1_loc.index[0])
        offset_ecg = ecg
    
        #offset = {'offset_ecg':offset_ecg, 'offset_bvp':offset_bvp}
        #return offset


    offset_acc2 = offset_ecg
    offset_ACC2 = offset_bvp
    offset_ACC2.iloc[:,2]

    l3 = offset_acc2.iloc[:,0:1]
    l4 = offset_acc2.iloc[:,4]

    Acceleration_x2 = offset_acc2.iloc[:,1]
    Acceleration_X12 = pd.concat([l3,Acceleration_x2], axis=1, join='outer')
    Acceleration_X2 = pd.concat([Acceleration_X12,l4], axis=1, join='outer')

    Acceleration_y2 = offset_acc2.iloc[:,2]
    Acceleration_Y12 = pd.concat([l3,Acceleration_y2], axis=1, join='outer')
    Acceleration_Y2 = pd.concat([Acceleration_Y12,l4], axis=1, join='outer')

    Acceleration_z2 = offset_acc2.iloc[:,3]
    Acceleration_Z12 = pd.concat([l3,Acceleration_z2], axis=1, join='outer')
    Acceleration_Z2 = pd.concat([Acceleration_Z12,l4], axis=1, join='outer')

    l5 = offset_ACC2.iloc[:,0:1]
    l6 = offset_ACC2.iloc[:,4]

    Acceleration_x3 = offset_ACC2.iloc[:,1]
    Acceleration_X13 = pd.concat([l5,Acceleration_x3], axis=1, join='outer')
    Acceleration_X3 = pd.concat([Acceleration_X13,l6], axis=1, join='outer')

    Acceleration_y3 = offset_ACC2.iloc[:,2]
    Acceleration_Y13 = pd.concat([l5,Acceleration_y3], axis=1, join='outer')
    Acceleration_Y3 = pd.concat([Acceleration_Y13,l6], axis=1, join='outer')

    Acceleration_z3 = offset_ACC2.iloc[:,3]
    Acceleration_Z13 = pd.concat([l5,Acceleration_z3], axis=1, join='outer')
    Acceleration_Z3 = pd.concat([Acceleration_Z13,l6], axis=1, join='outer')



    acc22 = offset_acc2.iloc[:,1:4]
    acc22 = acc22.values
    ACC22 = offset_ACC2.iloc[:,1:4]
    ACC22 = ACC22.values

    ax = Acceleration_X2.iloc[:,1]
    ax = np.expand_dims(ax.values, axis = 1)
    ay = Acceleration_Y2.iloc[:,1]
    ay = np.expand_dims(ay.values, axis = 1)
    az = Acceleration_Z2.iloc[:,1]
    az = np.expand_dims(az.values, axis = 1)

    Ax = Acceleration_X3.iloc[:,1]
    Ax = np.expand_dims(Ax.values, axis = 1)
    Ay = Acceleration_Y3.iloc[:,1]
    Ay = np.expand_dims(Ay.values, axis = 1)
    Az = Acceleration_Z3.iloc[:,1]
    Az = np.expand_dims(Az.values, axis = 1)

    E4_to_dic = {}
    #E4_to_dic["accx_e4"] = ax
    #E4_to_dic["accy_e4"] = ay
    #E4_to_dic["accz_e4"] = az

    #E4_to_dic["accx_hx"] = Ax
    #E4_to_dic["accy_hx"] = Ay
    #E4_to_dic["accz_hx"] = Az

    E4_to_dic["acc_e4"] = offset_bvp
    E4_to_dic["acc_hx"] = offset_acc2

    acc_dic = E4_to_dic

    if False:
        string4 = fhex+'_accE4&accHex.pkl'
        with open(string4, 'wb') as handle:
            pickle.dump(E4_to_dic, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return acc_dic
