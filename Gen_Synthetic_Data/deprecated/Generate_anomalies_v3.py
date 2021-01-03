import pandas as pd
import numpy as np
import os
import sys
sys.path.append('./../..')
sys.path.append('./..')
import glob
from tqdm import tqdm
from itertools import combinations
import joblib
from joblib import Parallel, delayed
from pandarallel import pandarallel
pandarallel.initialize()
import re
import yaml
from collections import Counter
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel,delayed
import multiprocessing as mp
id_col = 'PanjivaRecordID'
import networkx as nx
import operator
import collections
import argparse
try:
    from common_utils.utils import remove_spurious_coOcc
except:
    from .common_utils.utils import remove_spurious_coOcc

CONFIG = None
DIR_LOC = None
CONFIG = None
CONFIG_FILE = 'config.yaml'
save_dir = None
id_col = 'PanjivaRecordID'
use_cols = None
freq_bound = None
attribute_columns = None
domain_dims = None
TOTAL_ANOM_PERC = -1

def set_up_config(_DIR=None):
    global DIR
    global CONFIG
    global CONFIG_FILE
    global use_cols
    global num_neg_samples
    global DATA_SOURCE
    global DIR_LOC
    global save_dir
    global id_col
    global attribute_columns
    global domain_dims

    DATA_SOURCE = './../generated_data_v1/'
    with open(CONFIG_FILE) as f:
        CONFIG = yaml.safe_load(f)

    if _DIR is not None:
        DIR = _DIR
        CONFIG['DIR'] = _DIR
    else:
        DIR = CONFIG['DIR']

    DIR_LOC = re.sub('[0-9]', '', DIR)
    DATA_SOURCE = os.path.join(DATA_SOURCE, DIR)
    save_dir = 'stage_2'
    save_dir = os.path.join(
        DATA_SOURCE,
        save_dir
    )

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    use_cols = CONFIG[DIR]['use_cols']
    _cols = list(use_cols)
    _cols.remove(id_col)
    attribute_columns = list(sorted(_cols))
    with open(os.path.join(DATA_SOURCE, 'domain_dims.pkl'), 'rb') as fh:
        domain_dims = pickle.load(fh)
    print('Domains and sizes', domain_dims)
    return

def get_positive_edges():
    global save_dir
    edges_df = pd.read_csv(
        os.path.join(save_dir, 'seed_edges.csv'), index_col=None)
    return edges_df

def perturb_row(
        row,
        fixed_columns,
        domain_dims,
        hash_list,
        perturb_count=3,
        id_col='PanjivaRecordID'
):
    all_cols = sorted(domain_dims.keys())
    row_dict = row.to_dict()
    domains_perturb = [_ for _ in domain_dims.keys() if _ not in fixed_columns]

    domains_perturb = np.random.choice(domains_perturb, size=perturb_count, replace=False)
    while True:
        new_row = row.copy()
        for i in range(perturb_count):
            d = domains_perturb[i]
            e = row_dict[d]
            # select a random entity
            while True:
                rnd_e = np.random.randint(0, domain_dims[d])
                if rnd_e == e:
                    continue
                new_row[d] = rnd_e
                break
        _hash = '_'.join([str(new_row[_]) for _ in all_cols])

        # Exit : on not a duplicate
        if _hash not in hash_list:
            break
    # Perturb the ID
    new_row[id_col] = int(str(new_row[id_col]) + str(1001))
    return new_row


def generate_anomalies(
        target_df,
        ref_df,
        actor_columns,
        domain_dims,
        anom_perturb_count
):
    # Create hash of each row to check for duplicates/clashes
    all_cols = sorted(domain_dims.keys())
    ref_df['hash'] = ref_df.parallel_apply(
        lambda row: '_'.join([str(row[_]) for _ in all_cols]), axis=1
    )
    hash_list = ref_df['hash'].values.tolist()
    del ref_df['hash']

    anomalous_records = target_df.parallel_apply(
        perturb_row,
        axis=1,
        args=(actor_columns, domain_dims, hash_list, anom_perturb_count)
    )
    return anomalous_records


def main():
    global DATA_SOURCE
    global domain_dims
    global save_dir
    global anom_perturb_count
    global TOTAL_ANOM_PERC
    global id_col

    target_edges = get_positive_edges()
    actor_columns = ['ConsigneePanjivaID', 'ShipperPanjivaID']
    CONSIGNEE_column = 'ConsigneePanjivaID'
    SHIPPER_column = 'ShipperPanjivaID'

    train_df = pd.read_csv(os.path.join(DATA_SOURCE, 'train_data.csv'), low_memory=False)
    train_df = train_df.drop_duplicates(list(domain_dims.keys()))
    test_df = pd.read_csv(os.path.join(DATA_SOURCE, 'test_data.csv'), low_memory=False)
    test_df = test_df.drop_duplicates(list(domain_dims.keys()))


    # ------------------------------------------------------------------------------------
    # spurious co-occurrence to be removed from data that is supposed to be non-anomalous
    # ------------------------------------------------------------------------------------
    cleaned_test_df: object = remove_spurious_coOcc(
        test_df.copy(),
        train_df,
        domain_dims
    )
    # -----------------------------------------------------------------
    # Select some records that should be used  to generate anomalies
    # -----------------------------------------------------------------

    record_count = 0
    positive_samples = None
    sampling_DF = test_df.copy()
    sampling_DF = sampling_DF.append(train_df,ignore_index=True)
    for pair in zip(target_edges['ConsigneePanjivaID'].values, target_edges['ShipperPanjivaID'].values):
        _C = int(pair[0])
        _S = int(pair[1])
        # -------------------------
        # use test_df - without removing spurious co-occurrences. Since we are going to perturb it anyway
        # -------------------------
        tmp = sampling_DF.loc[(sampling_DF['ConsigneePanjivaID'] == _C) &
                                (sampling_DF['ShipperPanjivaID'] == _S)]
        record_count += len(tmp)
        if positive_samples is None:
            positive_samples = pd.DataFrame(tmp)
        else:
            positive_samples = positive_samples.append(tmp, ignore_index=True)

    positive_sample_ID_list = positive_samples[id_col].values.tolist()

    if TOTAL_ANOM_PERC != -1:
        count = int(TOTAL_ANOM_PERC * len(cleaned_test_df) / 100)
        if len(positive_samples) <  count:
            _replace =  True
        else:
            _replace = False
        positive_samples = positive_samples.sample(n=count, replace=_replace)
        positive_samples = positive_samples.reset_index(drop=True)

        # Ensure that ids are different
        positive_samples['serial_id'] = np.arange(positive_samples.shape[0])
        positive_samples[id_col] = positive_samples.parallel_apply(lambda x: int(str(x[id_col]) + '00'+ str(x['serial_id'])),axis=1)
        del positive_samples['serial_id']

    num_positive_samples = len(positive_samples)
    print(' >> ', num_positive_samples)

    set_consignee = target_edges[CONSIGNEE_column]
    set_shipper = target_edges[SHIPPER_column]

    # ------------------------------
    # For negative samples :
    # have negative samples which have neither of the actors
    # and also one of the marked entities.
    # ------------------------------

    candidates_N_1 = sampling_DF.loc[
        (sampling_DF['ConsigneePanjivaID'].isin(set_consignee)) | (
            sampling_DF['ShipperPanjivaID'].isin(set_shipper))
    ]
    candidates_N_1 = candidates_N_1.loc[~(candidates_N_1[id_col].isin(positive_sample_ID_list))]
    candidates_N_2 = sampling_DF.loc[
        ~(sampling_DF['ConsigneePanjivaID'].isin(set_consignee)) & ~(
            sampling_DF['ShipperPanjivaID'].isin(set_shipper))
        ]
    candidates_N_2 = candidates_N_2.loc[~(candidates_N_2[id_col].isin(positive_sample_ID_list))]
    candidates_N_1 = candidates_N_1.sample(num_positive_samples// 2)
    candidates_N_2 = candidates_N_2.sample(num_positive_samples// 2)
    candidates_N = candidates_N_1.append(candidates_N_2, ignore_index=True)
    candidates_N = candidates_N.drop_duplicates(subset=[id_col])
    print('Count of candidates for negative samples ', len(candidates_N))
    negative_samples = candidates_N.sample(frac=1)

    print('Print # positive, negative samples', num_positive_samples, len(negative_samples))
    pos_neg_IDs = negative_samples[id_col].values.tolist() + positive_sample_ID_list

    # -------------------------------------------
    ref_df = train_df.copy()
    negative_samples = generate_anomalies(
        negative_samples,
        ref_df,
        actor_columns,
        domain_dims,
        anom_perturb_count
    )

    positive_samples = generate_anomalies(
        positive_samples,
        ref_df,
        actor_columns,
        domain_dims,
        anom_perturb_count
    )

    # -------------------------
    # These samples have no anomalies
    # -------------------------
    normal_samples = cleaned_test_df.loc[~(cleaned_test_df[id_col].isin(pos_neg_IDs))]
    normal_samples = cleaned_test_df.loc[
            ~(cleaned_test_df['ConsigneePanjivaID'].isin(set_consignee)) &
            ~(cleaned_test_df['ShipperPanjivaID'].isin(set_shipper))]

    # ========================================
    # Save the data to csv file
    # ========================================

    # Normal samples
    save_path = os.path.join(save_dir, 'test_normal_data.csv')
    normal_samples.to_csv(save_path, index=None)
    # Positive samples
    save_path = os.path.join(save_dir, 'test_pos_data.csv')
    positive_samples.to_csv(save_path, index=None)
    # Negative samples
    save_path = os.path.join(save_dir, 'test_neg_data.csv')
    negative_samples.to_csv(save_path, index=None)
    # # Save all the cleaned records
    # save_path = os.path.join(save_dir, 'cleaned_test_data.csv')


# ===========================================================
parser = argparse.ArgumentParser()
parser.add_argument(
    '--DIR', choices=['us_import1', 'us_import2', 'us_import3', 'us_import4', 'us_import5'],
    default='us_import1'
)

parser.add_argument(
    '--anom_perturb_count',
    type=int,
    default=4
)

parser.add_argument(
    '--total_anom_perc',
    type=float,
    default=1.5
)

# --------------------------------------------------------------------------
args = parser.parse_args()
DIR = args.DIR
anom_perturb_count = args.anom_perturb_count
TOTAL_ANOM_PERC = args.total_anom_perc
set_up_config(DIR)
main()


