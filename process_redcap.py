import pickle
import csv
from pathlib import Path
import csv
import pandas as pd
import numpy as np
import pickle
import json
import os.path


def process_redcap(path, participants, force_update = False):
    '''
    Written by A. Alkurdi and X. Fan 
    usage: process_redcap(path, participants, force_update = False) wow
    If force_update is set to True, the function will update the pickle file containing the redcap data. use when redcap data is updated. 
    returns:
        label_df: pandas dataframe containing all the relevant information from the redcap csv files
        and saves the dataframe as a pickle file in the same directory as the redcap csv files. Enjoy!
    '''
    participants = participants
    redcap_path = path
    relevant_keys = ['daily_check_in_date', 'daily_feeling', 'daily_distressed_level', 'daily_anxious_level', 
                 'daily_overall_anxiety', 'daily_covid_contact', 'daily_covid_team_contact']
    partcipant_days = {4:[0,0], 5:[0,8], 7:[11,10], 9:[10,10], 12:[8,10], 14:[9,10], 16:[9,0], 17:[10,10], 18:[0,0], 20:[0,0], 21:[9,0]}
    all_d = {}
    

    my_file = Path(redcap_path+'redcap_dict.pkl')
    if my_file.is_file() and not force_update:
        print('redcap dict exists')
        with open(redcap_path+'redcap_dict.pkl', 'rb') as f:
            all_participants_redcap_dict = pickle.load(f)
            #all_participants_dict = pd.read_pickle(f)
    else: 
        if force_update:
            print('redcap dict exists but update forced')
        else:
            print('redcap dict does not exist')

        for participant in participants:
            participant__d = {}
            path = redcap_path+'Participant_' + str(participant) + '_RADWearStudy.csv'
            try:
                participant__d['global_info_dict'] = create_global_dict(path)
                participant__d['local_info_dict'] = get_daily_info_dict(path)

            except FileNotFoundError as e:
                print('MISTAKE! REDCap csv file for participant {} not found'.format(participant)) # Lord Jaraxxus, Eredar Lord of the Burning Legion, reference. 
                participant__d['global_info_dict'] = []
                participant__d['local_info_dict'] = []
            all_d[participant] = participant__d

        with open(redcap_path+'redcap_dict.pkl', 'wb') as f:
            pickle.dump(all_d, f)
            all_participants_redcap_dict = all_d


    


    

    my_file = Path(redcap_path+'redcap_df.pkl')
    if my_file.is_file() and not force_update:
        print('redcap df exists')
        with open(redcap_path+'redcap_df.pkl', 'rb') as f:
            #all_participants_redcap_dict = pickle.load(f)
            label_df = pickle.load(f)
            print('redcap df pickle loaded')
    else: 
        if force_update:
            print('redcap df exists but update forced')
        else:
            print('redcap df does not exist')
        subject_labels = []
        for key in all_participants_redcap_dict.keys():
            subject_data = []
            for instance in all_participants_redcap_dict[key]['local_info_dict'].keys():
                key_data = []
                for rel_key in relevant_keys:
                    # Ensure that the relevant keys exist to avoid runtime error
                    if len(all_participants_redcap_dict[key]['local_info_dict'][instance]) > 0:
                        key_data.append(all_participants_redcap_dict[key]['local_info_dict'][instance][rel_key])
                    else:
                        break
                # Ensure that subject labels contain all the necessary information      
                if len(key_data) == len(relevant_keys):
                    key_data.insert(0, key)
                    subject_labels.append(key_data)
        label_df = pd.DataFrame(subject_labels)
        label_df.rename(columns={0:'participant', 1:'daily_check_in_date', 2:'daily_anxious_level',
                                3:'daily_overall_anxiety', 4:'daily_distressed_level', 5:'daily_feeling',
                                6:'daily_covid_contact', 7:'daily_covid_team_contact'}, inplace=True)
        
        # Include Covid-19 contact
        attributes = ['daily_anxious_level', 'daily_overall_anxiety', 'daily_distressed_level', 'daily_feeling','daily_covid_contact', 'daily_covid_team_contact']
        label_df.loc[label_df['daily_covid_team_contact'] == ''] = '0'
        label_df[attributes] = label_df[attributes].astype(int)
        np_label = np.array(label_df[attributes])
        redcap_labels = attributes 
        
        with open(redcap_path+'redcap_df.pkl', 'wb') as f:
            pickle.dump(label_df, f)
            print('redcap df created, pickle dumped')
    
    # fixing for p7 missing data        
    label_df.at[28, 'daily_check_in_date'] = '2022-09-12' 
    label_df.at[30, 'daily_check_in_date'] = '2022-09-13'
    label_df.loc[29], label_df.loc[30] =  label_df.loc[30].copy(), label_df.loc[29].copy()
    label_df.drop([44], inplace = True)
    return label_df
def create_global_dict(path):
    global_info_dict = {}
    with open(path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['redcap_event_name'][:5] == 'daily':
                break
            for key in row.keys():
                if key != 'subject_id' and key != 'redcap_event_name' and key != 'redcap_repeat_instrument' and row[key] != '':
                    global_info_dict[key] = row[key]
    return global_info_dict
def get_daily_info_dict(path):

    daily_info_dict = {}
    event_type = ['daily_checkin_timestamp', 'daily_check_in_date', 'synce_reminder', 'daily_feeling', 
                  'daily_distressed_level', 'daily_covid_contact', 'daily_covid_team_contact', 'daily_contact_in_rotation', 
                  'daily_anxious_level', 'daily_overall_anxiety', 'tag_event_0', 'daily_anxious_event',
                  'tag_event_1', 'daily_other_event_detail', 'daily_time_anxiety', 'daily_anxiety_recall', 'daily_anxiety_event',
                  'daily_checkin_complete']
    with open(path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['redcap_event_name'][:5] != 'daily':
                continue
            unique_id = row['redcap_event_name'] + '/' + row['redcap_repeat_instrument'] + '/instance' + row['redcap_repeat_instance']
            nested_d = {}
            counter = 0
            for key in row.keys():
                if (key[-9:] != 'timestamp' or row[key] == '') and counter == 0:
                    continue
                if counter == len(event_type):
                    break
                nested_d[event_type[counter]] = row[key]
                counter += 1
                
            daily_info_dict[unique_id] = nested_d
    return daily_info_dict
def process_redcap_calibration(radwear_path = '/mnt/c/Users/alkurdi/Desktop/Vansh/data/RADWear/'):
    
    # check if file exists
    my_file = Path(radwear_path+'redcap_calib_dict.pkl')
    if my_file.is_file():
        print('redcap calibration dict exists')
        with open(radwear_path+'redcap_calib_dict.pkl', 'rb') as f:
            all_calib_df = pickle.load(f)
            print('redcap calibration dict pickle loaded')
    else: 
        print('redcap calibration dict does not exist')
        print('BUT WORRY NOT, WE WILL CREATE IT FOR YOU')

            
                                                            
        with open(radwear_path + 'all_p_metadata.json') as f:
            all_p_metadata = json.load(f)
                                                            
            participant__d = {}
            global_info_dict = {}
            all_calib_df = pd.DataFrame()
            redcap_path = radwear_path+'REDCap responses/'

            #with open(radwear_path+'all_p_metadata.json', 'rb') as f:
            #            all_p_metadata = json.load(f)
            list_of_participants = all_p_metadata['list of participant IDs']
            for i in list_of_participants:
                print(f' participant {i} status is {all_p_metadata[str(i)]["status"]} with e4 file {all_p_metadata[str(i)]["calibration"][1]} and hx file {all_p_metadata[str(i)]["calibration"][0]}') 

            for participant in list_of_participants:
                    
                path = redcap_path+'Participant_' + str(participant) + '_RADWearStudy.csv'
                with open(path, newline='') as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        if row['redcap_event_name'][:5] == 'basel':
                                for key in row.keys():
                                    #print(key[-2:])
                                    #if key != 'subject_id' and key != 'redcap_event_name' and key != 'redcap_repeat_instrument' and row[key] != '':
                                    #if key[-2:] == 'x2' or key[-2:] == 'y6':
                                    if '_cal' in key:
                                        global_info_dict[key] = row[key]
                                        #print('row[key]: ', row[key])
                daily_info_dict = {}
                event_type = ['calm_cal_x2', 'secure_cal_x2', 'tense_cal_x2', 'regretful_cal_x2',
                                'ease_cal_x2', 'upset_cal_x2', 'worrying_cal_x2', 'rested_cal_x2',
                                'anxious_cal_x2', 'comfort_cal_x2', 'self_conf_cal_x2', 'nervous_cal_x2',
                                'jittery_cal_x2', 'strun_cal_x2', 'relaxed_cal_x2', 'content_cal_x2',
                                'worried_cal_x2', 'excited_cal_x2', 'joyful_cal_x2', 'pleasant_cal_x2',
                                'calm_cal_y6', 'tense_cal_y6', 'upset_cal_y6', 'relax_cal_y6',
                                'content_cal_y6', 'worried_cal_y6',
                                'calm_cal_y6_post', 'tense_cal_y6_post', 'upset_cal_y6_post',
                                'relax_cal_y6_post', ' content_cal_y6_post', 'worry_cal_y6_post',
                                'calm_cal_y6_cold', 'tense_cal_y6_cold', 'upset_cal_y6_cold',
                                'relax_cal_y6_cold', 'content_cal_y6_cold', 'worry_cal_y6_cold',
                                'baseline_calibration_complete']
                with open(path, newline='') as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        if row['redcap_event_name'][:5] != 'basel':
                            continue
                        unique_id = row['redcap_event_name'] + '/' + row['redcap_repeat_instrument'] + '/instance' + row['redcap_repeat_instance']
                        nested_d = {}
                        counter = 0
                        for key in row.keys():
                            if (key[-9:] != 'timestamp' or row[key] == '') and counter == 0:
                                continue
                            if counter == len(event_type):
                                break
                            nested_d[event_type[counter]] = row[key]
                            counter += 1
                            
                        daily_info_dict[unique_id] = nested_d
                
                subject_labels = []
                subject_data = []
                for instance in daily_info_dict.keys():
                    key_data = []
                    for rel_key in event_type:
                        # Ensure that the relevant keys exist to avoid runtime error
                        if len(daily_info_dict[instance]) > 0:
                            key_data.append(daily_info_dict[instance][rel_key])
                        else:
                            break
                    # Ensure that subject labels contain all the necessary information      
                    if len(key_data) == len(event_type):
                        key_data.insert(0, 12)
                        subject_labels.append(key_data)
                label_df = pd.DataFrame(subject_labels)
                label_df.columns = list(global_info_dict.keys())
                label_df['participant'] = np.ones(len(label_df))*participant
                all_calib_df = pd.concat([all_calib_df, label_df])

            with open(redcap_path+'redcap_calib_dict.pkl', 'wb') as f:
                pickle.dump(all_calib_df, f)
            # get HADs survey, baseline calibration 
    return all_calib_df