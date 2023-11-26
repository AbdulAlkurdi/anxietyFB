"""
process_redcap.py

This module contains the function `process_redcap` which processes 
participant data from a given path. 

The function takes the following parameters:
- path: The filepath of the participant data.
- participants: A list of participants to process data for.
- force_update: A boolean flag indicating whether to force update the data.

The module uses the pandas library for data manipulation, the numpy
library for numerical operations, the json and csv modules for reading JSON
and CSV files, and the pathlib module for handling file paths.

Author: Abdul Alkurdi
Date: created oct 10 2023, modified 11/25/2023
"""
import json
import pickle
import csv
from pathlib import Path
import pandas as pd

radwear_path = '/mnt/c/Users/alkurdi/Desktop/Vansh/data/RADWear/'
redcap_path = radwear_path + 'REDCap responses/'

with open(radwear_path + 'all_p_metadata.json', 'rb') as f:
    all_p_metadata = json.load(f)


def process_redcap(
    path=redcap_path,
    participants=all_p_metadata['list of participant IDs'],
    force_update=True,
):
    '''
    Written by A. Alkurdi and X. Fan for RADWear and WEAR data
    usage: process_redcap(path, participants, force_update = False) wow
    """
    This script processes participant data for a study. It reads data from CSV files and stores it in two formats: a dictionary and a pandas DataFrame. Both are saved as pickled files.

    The dictionary (`redcap_dict.pkl`) is a nested structure where each key is a participant ID, and the value is another dictionary containing two keys: `global_info_dict` and `local_info_dict`. These are created by the `create_global_dict` and `get_daily_info_dict` functions respectively, which process participant data from a CSV file. If the CSV file for a participant is not found, these dictionaries are empty lists.

    The DataFrame (`redcap_df.pkl`) is created from the dictionary. Each row corresponds to a participant and a day (an instance in `local_info_dict`), and the columns are the keys specified in `relevant_keys`, plus the participant ID.

    The script checks if the pickled files exist and if `force_update` is set. If the files exist and `force_update` is not set, it uses the existing files. Otherwise, it creates or updates the files.

    The script also handles missing data for participant 7 by manually setting the `daily_check_in_date` for certain rows and swapping the data for two rows.

    If force_update is set to True, the function will update the pickle file containing the redcap data. use when redcap data is updated.
    #returns:
        label_df: pandas dataframe containing all the relevant information from the redcap csv files
        and saves the dataframe as a pickle file in the same directory as the redcap csv files. Enjoy!
    '''
    relevant_keys = [
        'daily_check_in_date',
        'daily_feeling',
        'daily_distressed_level',
        'daily_anxious_level',
        'daily_overall_anxiety',
        'daily_covid_contact',
        'daily_covid_team_contact',
    ]
    # partcipant_days = {
    #    4: [0, 0],
    #    5: [0, 8],
    #    7: [11, 10],
    #    9: [10, 10],
    #    12: [8, 10],
    #    14: [9, 10],
    #    16: [9, 0],
    #    17: [10, 10],
    #    18: [0, 0],
    #    20: [0, 0],
    #    21: [9, 0],
    # }
    all_d = {}

    my_file = Path(path + 'redcap_dict.pkl')
    print('my_file', my_file)
    if my_file.is_file() and not force_update:
        print(f'redcap dict exists and named redcap_dict.pkl in {path}')
    else:
        if force_update:
            print('redcap dict exists but update forced')
        else:
            print('redcap dict does not exist')
        for participant in participants:
            participant__d = {}
            p_path = path + 'Participant_' + str(participant) + '_RADWearStudy.csv'
            try:
                participant__d['global_info_dict'] = create_global_dict(p_path)
                participant__d['local_info_dict'] = get_daily_info_dict(p_path)

            except FileNotFoundError:
                print(
                    f'MISTAKE! REDCap csv file for participant {participant} not found'
                )  # Lord Jaraxxus, Eredar Lord of the Burning Legion, reference.
                participant__d['global_info_dict'] = []
                participant__d['local_info_dict'] = []
            all_d[participant] = participant__d

        with open(path + 'redcap_dict.pkl', 'wb') as f:
            pickle.dump(all_d, f)
        all_participants_redcap_dict = all_d

    my_file = Path(path + 'redcap_df.pkl')
    if my_file.is_file() and not force_update:
        print('redcap df exists')
        with open(path + 'redcap_df.pkl', 'rb') as f:
            # all_participants_redcap_dict = pickle.load(f)
            label_df = pickle.load(f)
            print('redcap df pickle loaded')
    else:
        if force_update:
            print('redcap df exists but update forced')
        else:
            print('redcap df does not exist')
        subject_labels = []
        for key in all_participants_redcap_dict.keys():
            # subject_data = []
            for instance in all_participants_redcap_dict[key]['local_info_dict'].keys():
                key_data = []
                for rel_key in relevant_keys:
                    # Ensure that the relevant keys exist to avoid runtime error
                    if (
                        len(
                            all_participants_redcap_dict[key]['local_info_dict'][
                                instance
                            ]
                        )
                        > 0
                    ):
                        key_data.append(
                            all_participants_redcap_dict[key]['local_info_dict'][
                                instance
                            ][rel_key]
                        )
                    else:
                        break
                # Ensure that subject labels contain all the necessary information
                if len(key_data) == len(relevant_keys):
                    key_data.insert(0, key)
                    subject_labels.append(key_data)
        label_df = pd.DataFrame(subject_labels)
        label_df.rename(
            columns={
                0: 'participant',
                1: 'daily_check_in_date',
                2: 'daily_anxious_level',
                3: 'daily_overall_anxiety',
                4: 'daily_distressed_level',
                5: 'daily_feeling',
                6: 'daily_covid_contact',
                7: 'daily_covid_team_contact',
            },
            inplace=True,
        )

        # Include Covid-19 contact
        attributes = [
            'daily_anxious_level',
            'daily_overall_anxiety',
            'daily_distressed_level',
            'daily_feeling',
            'daily_covid_contact',
            'daily_covid_team_contact',
        ]
        label_df.loc[label_df['daily_covid_team_contact'] == ''] = '0'
        label_df[attributes] = label_df[attributes].astype(int)
        # np_label = np.array(label_df[attributes])
        # redcap_labels = attributes

        with open(path + 'redcap_df.pkl', 'wb') as f:
            pickle.dump(label_df, f)
            print('redcap df created, pickle dumped')

    # fixing for p7 missing data
    label_df.at[28, 'daily_check_in_date'] = '2022-09-12'
    label_df.at[30, 'daily_check_in_date'] = '2022-09-13'
    label_df.loc[29], label_df.loc[30] = (
        label_df.loc[30].copy(),
        label_df.loc[29].copy(),
    )
    label_df.drop([44], inplace=True)


def create_global_dict(path):
    """
    Create a global dictionary from a CSV file.

    Args:
        path (str): The path to the CSV file.

    Returns:
        dict: The global dictionary containing non-empty values from the CSV file.
    """
    global_info_dict = {}
    print('path inside create_global_dict', path)
    with open(path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['redcap_event_name'][:5] == 'daily':
                break
            for key in row.keys():
                if (
                    key != 'subject_id'
                    and key != 'redcap_event_name'
                    and key != 'redcap_repeat_instrument'
                    and row[key] != ''
                ):
                    global_info_dict[key] = row[key]
    return global_info_dict


def get_daily_info_dict(path):
    """
    Retrieves daily information from a CSV file and returns it as a dictionary.

    Args:
        path (str): The path to the CSV file.

    Returns:
        dict: A dictionary containing the daily information, with unique IDs as keys and nested dictionaries as values.
              The nested dictionaries contain the daily information with event types as keys and corresponding values.

    """
    daily_info_dict = {}
    event_type = [
        'daily_checkin_timestamp',
        'daily_check_in_date',
        'synce_reminder',
        'daily_feeling',
        'daily_distressed_level',
        'daily_covid_contact',
        'daily_covid_team_contact',
        'daily_contact_in_rotation',
        'daily_anxious_level',
        'daily_overall_anxiety',
        'tag_event_0',
        'daily_anxious_event',
        'tag_event_1',
        'daily_other_event_detail',
        'daily_time_anxiety',
        'daily_anxiety_recall',
        'daily_anxiety_event',
        'daily_checkin_complete',
    ]
    with open(path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['redcap_event_name'][:5] != 'daily':
                continue
            unique_id = (
                row['redcap_event_name']
                + '/'
                + row['redcap_repeat_instrument']
                + '/instance'
                + row['redcap_repeat_instance']
            )
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


def process_redcap_calibration(
    path='/mnt/c/Users/alkurdi/Desktop/Vansh/data/RADWear/',
):
    """
    Process REDCap calibration data.

    Args:
        path (str): The path to the RADWear data directory.
        Defaults to '/mnt/c/Users/alkurdi/Desktop/Vansh/data/RADWear/'.

    Returns:
        pandas.DataFrame: The processed calibration data.

    Raises:
        FileNotFoundError: If the redcap_calib_dict.pkl file does not exist.
    """
    # check if file exists
    my_file = Path(path + 'redcap_calib_dict.pkl')
    if my_file.is_file():
        print('redcap calibration dict exists')
        with open(path + 'redcap_calib_dict.pkl', 'rb') as f:
            all_calib_df = pickle.load(f)
            print('redcap calibration dict pickle loaded')
    else:
        print('redcap calibration dict does not exist')
        print('BUT WORRY NOT, WE WILL CREATE IT FOR YOU')

        global_info_dict = {}
        all_calib_df = pd.DataFrame()
        list_of_participants = all_p_metadata['list of participant IDs']
        for participant in list_of_participants:
            path = redcap_path + 'Participant_' + str(participant) + '_RADWearStudy.csv'
            with open(path, newline='') as csvfile:
                reader = csv.DictReader(csvfile)
            for row in reader:
                if row['redcap_event_name'][:5] == 'basel':
                    for key in row.keys():
                        if '_cal' in key:
                            global_info_dict[key] = row[key]
            participant_df = pd.read_csv(path)
            participant_df = participant_df[
                participant_df['redcap_event_name'].str.contains('baseline_')
            ]
            participant_df = participant_df.dropna(axis=1, how='all')
            participant_df = participant_df.dropna()
            participant_df['subject_id'] = participant
            participant_df['baseline_calibration_timestamp'] = pd.to_datetime(
                participant_df['baseline_calibration_timestamp']
            )
            participant_df['baseline_calibration_unixtime'] = (
                participant_df['baseline_calibration_timestamp'].astype(int) / 10**9
            )
            if participant == 12:
                participant_df = participant_df[
                    participant_df['redcap_repeat_instance'] != 1
                ]
            all_calib_df = pd.concat([all_calib_df, participant_df]).reset_index(
                drop=True
            )
        with open(redcap_path + 'redcap_calib_dict.pkl', 'wb') as f:
            pickle.dump(all_calib_df, f)
    return all_calib_df


process_redcap()
# process_redcap_calibration()
