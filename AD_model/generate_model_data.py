# Script to generate data for AD model
# -------------------------------------------
# Anomaly Data is already serialized
# Anomalies location: generated_data_v1/generated_anomalies/__DIR__
# Positive : pos_anomalies.csv
# Negative : neg_anomalies.csv

import argparse
import os
import sys
sys.path.append('../../..')
sys.path.append('./..')
import pandas as pd
try:
    from .common_utils import utils
except:
    from common_utils import utils
import numpy as np
from pathlib import Path
# ============================
ID_COL = 'PanjivaRecordID'

parser = argparse.ArgumentParser()
parser.add_argument(
    '--DIR', choices=['us_import1', 'us_import2', 'us_import3', 'us_import4', 'us_import5', 'us_import6'],
    default=None
)

args = parser.parse_args()
DIR = args.DIR
data_source_dir = './../generated_data_v1/{}'.format(DIR)
anomaly_source_dir = './../generated_data_v1/generated_anomalies/{}'.format(DIR)
save_dir_stage_2 = './../generated_data_v1/{}/stage_2'.format(DIR)
path = Path(save_dir_stage_2)
path.mkdir(exist_ok=True,parents=True)



train_df = pd.read_csv(os.path.join(data_source_dir, 'train_data.csv' ), index_col=None)
attributes = list(train_df.columns)
attributes.remove(ID_COL)
train_df = train_df.drop_duplicates(subset=attributes)
train_df = utils.convert_to_serializedID_format(train_df,DIR)
train_df.to_csv(os.path.join(save_dir_stage_2,'train_serialized.csv'), index=None)
print('Post dropping duplicates, size of train set', len(train_df))
x_pos, x_neg = utils.generate_negative_samples(train_df, DIR, num_neg_samples=10)
# Save training data
np.save(os.path.join(save_dir_stage_2,'train_x_pos.npy'),x_pos)
np.save(os.path.join(save_dir_stage_2,'train_x_neg.npy'),x_neg)

# --------------------------------------
test_normal_df = pd.read_csv(os.path.join(data_source_dir, 'test_data.csv' ), index_col=None)
test_normal_df = utils.convert_to_serializedID_format(test_normal_df,DIR)
test_normal_df = test_normal_df.drop_duplicates(subset=attributes)
print('Post dropping duplicates, size of test set', len(test_normal_df))
test_normal_df.to_csv(os.path.join(save_dir_stage_2,'test_serialized.csv'),index=None)


# --------------------------------------
# The id s in the files are already serialized
# --------------------------------------
test_pos_fpath = os.path.join( './../generated_data_v1/{}/stage_2/test_pos_serialized.csv'.format(DIR)) 
test_neg_fpath = os.path.join( './../generated_data_v1/{}/stage_2/test_neg_serialized.csv'.format(DIR)) 

src_path = './../generated_data_v1/genearted_anomalies/{}'.format(DIR)
anom_pos_df = pd.read_csv(os.path.join(anomaly_source_dir, 'pos_anomalies.csv' ), index_col=None)
anom_neg_df = pd.read_csv(os.path.join(anomaly_source_dir, 'neg_anomalies.csv' ), index_col=None)
# simple sanity check!
print('Positive anomalies :: ', len(anom_pos_df))
print('Negative anomalies :: ', len(anom_neg_df))

