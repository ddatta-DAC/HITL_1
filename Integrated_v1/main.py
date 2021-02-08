import pandas as pd
import yaml
from tqdm import tqdm
import os
from pathlib import Path
from os.path import join as pjoin
import numpy as np
from common_utils import utils

DIR = None
config = None
AD_result_loc = None
data_dir = None
anomalies_dir = None
ID_COL = 'PanjivaRecordID'


def setup(DIR):
    global AD_result_loc
    global anomalies_dir
    global config
    with open('config.yaml', 'r') as fh:
        config = yaml.parse(fh)
    AD_result_loc = config['AD_result_loc']
    return


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
    df_test_data = pd.read.csv(pjoin(data_dir, DIR, 'test_data.csv'), index_col=None)

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

    return df_combined
