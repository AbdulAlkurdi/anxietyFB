import os
import utils
import respiration
import fb_code.ecg_fcns as ecg_fcns
import pandas as pd
import numpy as np
import neurokit2 as nk
import pickle

from feature_extraction import *
from datetime import datetime





savePath = 'data'
subject_feature_path = '/WESAD/subject_feats'

subjects = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]



now = datetime.today().strftime('%Y-%m-%d--%H-%M-%p')



# combine files

df_list = []
for s in subjects:
    df = pd_old.read_csv(f'{savePath}{subject_feature_path}/S{s}_feats.csv', index_col=0)
    df['subject'] = s
    df_list.append(df)

df = pd_old.concat(df_list)

df['label'] = (df['0'].astype(str) + df['1'].astype(str) + df['2'].astype(str)).apply(lambda x: x.index('1'))
df.drop(['0', '1', '2'], axis=1, inplace=True)

df.reset_index(drop=True, inplace=True)

df.to_csv(f'{savePath}/{now}_feats4.csv')

counts = df['label'].value_counts()
print('Number of samples per class:')
for label, number in zip(counts.index, counts.values):
    print(f'{int_to_label[label]}: {number}')
