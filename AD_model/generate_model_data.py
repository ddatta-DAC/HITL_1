import argparse
import os
import sys
sys.path.append('./../..')
sys.path.append('./..')
import pandas as pd
try:
    import utils
except:
    from . import utils
import numpy as np

# ============================

parser = argparse.ArgumentParser()
parser.add_argument(
    '--DIR', choices=['us_import1', 'us_import2', 'us_import3'],
    default='us_import1'
)

args = parser.parse_args()
DIR = args.DIR

save_dir_stage_1 = './../generated_data_v1/us_import1'
save_dir_stage_2 = './../generated_data_v1/us_import1/stage_2'

train_df = pd.read_csv(os.path.join(save_dir_stage_1, 'train_data.csv' ), index_col=None)
test_df = pd.read_csv(os.path.join(save_dir_stage_1, 'test_data.csv'), index_col=None)
train_df = utils.convert_to_serializedID_format(train_df,DIR)
test_df = utils.convert_to_serializedID_format(test_df, DIR)
# Save test data
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