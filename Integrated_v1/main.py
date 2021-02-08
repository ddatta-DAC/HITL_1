import pandas as pd
import yaml
from tqdm import tqdm
import os
from pathlib import Path
from os.path import join as pjoin
import numpy as np
from collections import OrderedDict
import json
import sys
sys.path.append('./../..')
sys.path.append('./..')
from common_utils import utils
import pickle


config = None
AD_result_loc = None
data_dir = None
anomalies_dir = None
ID_COL = 'PanjivaRecordID'
domain_dims = None
serialID_mapping_loc = None

def setup():
    global DIR
    global AD_result_loc
    global anomalies_dir
    global data_dir
    global config
    global domain_dims
    with open('config.yaml', 'r') as fh:
        config = yaml.safe_load(fh)
    AD_result_loc = config['AD_result_loc']
    anomalies_dir = config['anomalies_dir']
    data_dir = config['data_dir']

    with open(pjoin(data_dir,DIR,'domain_dims.pkl'), 'rb') as fh:
        domain_dims = OrderedDict(pickle.load(fh))
    return




def get_serialID_to_entityID():
    global serialID_mapping_loc
    idMapper_file = os.path.join(serialID_mapping_loc)
    mapping_df = pd.read_csv(idMapper_file, index_col=None)
    serialID_to_entityID = {}

    for i, row in mapping_df.iterrows():
        serialID_to_entityID[row['serial_id']] = row['entity_id']
    return serialID_to_entityID




def read_in_data():
    global AD_result_loc
    global DIR
    global data_dir
    global anomalies_dir
    global ID_COL

    df_ranked = pd.read_csv(pjoin(AD_result_loc, DIR, 'AD_output.csv'), index_col=None)
    # This file is in serialized format
    df_pos_anomalies = pd.read_csv(pjoin(anomalies_dir, DIR, 'pos_anomalies.csv'), index_col=None)
    df_pos_anomalies = utils.convert_from_serialized(df_pos_anomalies.copy(), DIR)

    # ------------------------------------
    # Not considering negative anomalies
    # ------------------------------------
    # This file is in un-serialized format
    df_test_data = pd.read_csv(pjoin(data_dir, DIR, 'test_data.csv'), index_col=None)
    df_combined = df_test_data.copy().append(df_pos_anomalies, ignore_index=True)
    df_combined = df_combined.merge(
        df_ranked,
        on=ID_COL,
        how='inner'
    )

    # This file has entity ids
    df_combined = df_combined.sort_values(by='rank', ascending=False)
    # higher score -> anomalous
    df_combined = df_combined.rename(columns={'rank': 'score'})
    df_combined['score'] = df_combined['score'].apply(lambda x : 1.000-x)
    # Ensure columns are in correct order
    columns = [ID_COL] + list(sorted(domain_dims.keys())) + ['label','score']
    df_combined = df_combined[columns]
    print(columns)

    return df_combined

DIR = 'us_import1'
setup()
# tmp = read_in_data()
# print(tmp.head(10))