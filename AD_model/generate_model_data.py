import argparse
import os
import sys
sys.path.append('./../..')
sys.path.append('./..')
import pandas as pd
try:
    import utils
except:
    from common_utils import utils
import numpy as np

# ============================

parser = argparse.ArgumentParser()
parser.add_argument(
    '--DIR', choices=['us_import1', 'us_import2', 'us_import3', 'us_import4', 'us_import5'],
    default=None
)

args = parser.parse_args()
DIR = args.DIR

save_dir_stage_1 = './../generated_data_v1/{}'.format(DIR)
save_dir_stage_2 = './../generated_data_v1/{}/stage_2'.format(DIR)

# -------------------------------------

test_normal_df = pd.read_csv(os.path.join(save_dir_stage_2, 'test_normal_data.csv' ), index_col=None)
test_normal_df = utils.convert_to_serializedID_format(test_normal_df,DIR)
test_normal_df.to_csv(os.path.join(save_dir_stage_2,'test_normal_serialized.csv'),index=None)

test_neg_df = pd.read_csv(os.path.join(save_dir_stage_2, 'test_neg_data.csv' ), index_col=None)
test_neg_df = utils.convert_to_serializedID_format(test_neg_df, DIR)
test_neg_df.to_csv(os.path.join(save_dir_stage_2,'test_neg_serialized.csv'),index=None)

test_pos_df = pd.read_csv(os.path.join(save_dir_stage_2, 'test_pos_data.csv' ), index_col=None)
test_pos_df = utils.convert_to_serializedID_format(test_pos_df,DIR)
test_pos_df.to_csv(os.path.join(save_dir_stage_2,'test_pos_serialized.csv'), index=None)

train_df = pd.read_csv(os.path.join(save_dir_stage_1, 'train_data.csv' ), index_col=None)
ID_COL = 'PanjivaRecordID'
attributes = list(train_df.columns)
attributes.remove(ID_COL)
train_df = train_df.drop_duplicates(subset=attributes)
print('Post dropping duplicates, size of train set', len(train_df))

test_df = pd.read_csv(os.path.join(save_dir_stage_1, 'test_data.csv'), index_col=None)
train_df = utils.convert_to_serializedID_format(train_df,DIR)
test_df = utils.convert_to_serializedID_format(test_df, DIR)
# Save the train data
train_df.to_csv(os.path.join(save_dir_stage_2,'train_data_serialized.csv'), index=None)
# ---------------------
# Save test data
# ---------------------
test_df.to_csv(os.path.join(save_dir_stage_2,'test_data_serialID.csv'),index=None)

# =============================
# Generate negative data
# =============================
# Drop duplicates
ID_COL = 'PanjivaRecordID'
attributes = list(train_df.columns)
attributes.remove(ID_COL)
train_df = train_df.drop_duplicates(subset=attributes)
print('Post dropping duplicates, size of train set', len(train_df))

x_pos, x_neg = utils.generate_negative_samples(train_df, DIR, num_neg_samples=10)
# Save training data
np.save(os.path.join(save_dir_stage_2,'train_x_pos.npy'),x_pos)
np.save(os.path.join(save_dir_stage_2,'train_x_neg.npy'),x_neg)



