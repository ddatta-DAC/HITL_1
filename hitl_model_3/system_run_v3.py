# %load_ext autoreload
# %autoreload 2
import os
import argparse

import seaborn as sns
from matplotlib import pyplot as plt
import time
from pathlib import Path

#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
import os
import sys
import glob
from tqdm import tqdm
from sklearn.preprocessing import normalize
from GD import GD
sys.path.append('./.')
sys.path.append('./..')
from pathlib import Path
import argparse
import pickle
import copy
import json
from GD import GD
from linear_model_v2 import linearClassifier_bEF
import seaborn as sns
from record import record_class
import yaml
import time
from collections import OrderedDict
from common_utils import utils
from sklearn.utils import shuffle

explantions_file_path = None
embedding_data_path = None
serialID_mapping_loc = None
anomalies_pos_fpath = None
anomalies_neg_fpath = None
interaction_type = 'concat'
DIR = None


'''
embedding_data_path  = './../../createGraph_trade/saved_model_data/{}'.format(DIR)
serialID_mapping_loc = './../../generated_data_v1/{}/idMapping.csv'.format(DIR)
anomalies_pos_fpath = './../../generated_data_v1/generated_anomalies/{}/pos_anomalies.csv'.format(DIR)
anomalies_neg_fpath = './../../generated_data_v1/generated_anomalies/{}/neg_anomalies.csv'.format(DIR)
explantions_f_path =  './../../generated_data_v1/generated_anomalies/{}/pos_anomalies_explanations.json'.format(DIR)
'''

def setup_config(DIR):
    global explantions_file_path
    global embedding_data_path
    global serialID_mapping_loc
    global anomalies_pos_fpath
    global anomalies_neg_fpath
    global domain_dims
    global test_data_serialized_loc
    with open('config.yaml', 'r') as fh:
        config = yaml.safe_load(fh)

    serialID_mapping_loc = config['serialID_mapping_loc'].format(DIR)
    embedding_data_path = config['embedding_data_path'].format(DIR)
    explantions_file_path = config['explantions_file_path'].format(DIR)
    anomalies_pos_fpath = config['anomalies_pos_fpath'].format(DIR)
    anomalies_neg_fpath = config['anomalies_neg_fpath'].format(DIR)
    test_data_serialized_loc = config['test_data_serialized_loc'].format(DIR)

    with open(config['domain_dims_file_path'].format(DIR), 'rb') as fh:
        domain_dims = OrderedDict(pickle.load(fh))
    return

# ---------------------------------------------------------------------------------
def get_serialID_to_entityID():
    global serialID_mapping_loc
    global DIR
    
    idMapper_file = os.path.join(serialID_mapping_loc)
    mapping_df = pd.read_csv(idMapper_file, index_col=None)
    serialID_to_entityID = {}

    for i, row in mapping_df.iterrows():
        serialID_to_entityID[row['serial_id']] = row['entity_id']
    return serialID_to_entityID

# ---------------------------
# Get records which are deemed nominal/normal
# ---------------------------
def obtain_normal_samples():
    global test_data_serialized_loc
    normal_data = pd.read_csv(
        test_data_serialized_loc, index_col=None
    )

    _df = normal_data.sample(5000)
    obj_list = []
    data_x = []
    for i in tqdm(range(_df.shape[0])):
        obj = record_class(_df.iloc[i].to_dict(), -1)
        obj.calc_features()
        obj_list.append(obj)
        data_x.append(obj.features)
    
    return data_x

def get_trained_classifier(X, y, num_domains, emb_dim, num_epochs=10000):
    global domain_dims
    global interaction_type
    classifier_obj = linearClassifier_bEF(
        num_domains=num_domains,
        emb_dim=emb_dim,
        num_epochs=num_epochs,
        L2_reg_lambda=0.0025,
        force_reg=False,
        interaction_type=interaction_type
    )

    classifier_obj.setup_binaryFeatures(
        domain_dims,
        binaryF_domains=['ConsigneePanjivaID', 'ShipperPanjivaID']
    )

    # classifier_obj.fit_on_pos(X, np.ones(X.shape[0]),n_epochs=10000)
    classifier_obj.fit(X, y, log_interval=5000)
    classifier_obj.fit_on_pos(X, y, n_epochs=num_epochs // 2, log_interval=1000)
    return classifier_obj


def fetch_entityID_arr_byList(data_df, id_list):
    global domain_dims
    domain_list = list(domain_dims.keys())
    ID_COL = 'PanjivaRecordID'
    data_df = data_df.copy(deep=True)
    data_df = data_df.loc[data_df[ID_COL].isin(id_list)]
    # Order of id_list has to be preserved!!!
    X = []
    for _id in id_list:
        _tmp = data_df.loc[data_df[ID_COL] == _id][domain_list].iloc[0].values.tolist()
        X.append(_tmp)
    return np.array(X).astype(int)



def get_data():
    global anomalies_pos_fpath
    global anomalies_neg_fpath
    global domain_dims
    global explantions_file_path
    global embedding_data_path
    global serialID_mapping_loc

    # ============================================

    anom_pos_df = pd.read_csv(anomalies_pos_fpath, index_col=None)
    anom_neg_df = pd.read_csv(anomalies_neg_fpath, index_col=None)
    serialID_to_entityID = get_serialID_to_entityID()
    print('Setting up record class embedding...',embedding_data_path)
    record_class.__setup_embedding__(embedding_data_path, serialID_to_entityID, _normalize=True)
    main_data_df = pd.concat(
        [anom_pos_df,
         anom_neg_df],
        axis=0
    )
    # main_data_df has the records with entity ids


    obj_list = []
    for i in tqdm(range(anom_neg_df.shape[0])):
        obj = record_class(anom_neg_df.iloc[i].to_dict(), -1)
        obj.calc_features()
        obj_list.append(obj)

    for i in tqdm(range(anom_pos_df.shape[0])):
        obj = record_class(anom_pos_df.iloc[i].to_dict(), 1)
        obj.calc_features()
        obj_list.append(obj)

    # Read in the explantions
    with open(explantions_file_path, 'rb') as fh:
        explanations = json.load(fh)

    explanations = {int(k): [sorted(_) for _ in v] for k, v in explanations.items()}
    data_x = []
    data_x_features = []
    data_id = []
    data_label = []
    data_ID_to_matrix = {}

    for _obj in obj_list:
        data_x.append(_obj.x)
        data_id.append(_obj.id)
        data_label.append(_obj.label)
        data_ID_to_matrix[_obj.id] = _obj.features
        data_x_features.append(_obj.features)
    data_x = np.stack(data_x)
    data_label = np.array(data_label)
    data_id = np.array(data_id)
    return main_data_df, explanations , data_id, data_x, data_label, data_x_features, data_ID_to_matrix



# ------------------------------------------------------
def execute_with_input(
    check_next = 25,
    batch_size= 25
):

    serialID_to_entityID = get_serialID_to_entityID()
    main_data_df, explanations, data_id, data_x, data_label, data_x_features, data_ID_to_matrix = get_data()
    emb_dim = record_class.embedding['HSCode'].shape[1]

    # -------------------------------------------
    domain_idx = {e[0]: e[1] for e in enumerate(domain_dims.keys())}
    domainInteraction_index = {}
    num_domains = len(domain_dims)
    k = 0
    for i in range(num_domains):
        for j in range(i + 1, num_domains):
            domainInteraction_index['_'.join((domain_idx[i], domain_idx[j]))] = k
            k += 1

    idx = np.arange(len(data_id), dtype=int)
    np.random.shuffle(idx)

    data_x = data_x[idx]
    data_label = data_label[idx]
    data_id = data_id[idx]

    X_0 = np.array(data_x_features)[idx.tolist()]  # Relevant anomalies
    X_1 = obtain_normal_samples()  # Nominal


    X_1 =np.array(X_1)

    y_0 = np.ones(X_0.shape[0])
    y_1 = -1 * np.ones(X_1.shape[0])
    y = np.hstack([y_0, y_1])
    X = np.vstack([X_0, X_1])
    num_coeff = len(domainInteraction_index)

    classifier_obj = get_trained_classifier(
        X,
        y,
        num_domains,
        emb_dim
    )

    W = classifier_obj.W.cpu().data.numpy()
    emb_dim = W.shape[-1]

    # classifier_obj.predict_score_op(X_0)
    # Create a reference dataframe  :: data_reference_df
    data_reference_df = pd.DataFrame(
        data=np.vstack([data_id, data_label]).transpose(),
        columns=['PanjivaRecordID', 'label']
    )

    data_reference_df['baseID'] = data_reference_df['PanjivaRecordID'].apply(lambda x: str(x)[:-3])
    data_reference_df['expl_1'] = -1
    data_reference_df['expl_2'] = -1
    data_reference_df['original_score'] = 1

    for i, row in data_reference_df.iterrows():
        _id = int(row['PanjivaRecordID'])
        if _id in explanations.keys():
            entry = explanations[_id]
            domain_1 = entry[0][0]
            domain_2 = entry[0][1]
            data_reference_df.loc[i, 'expl_1'] = domainInteraction_index['_'.join(sorted([domain_1, domain_2]))]
            domain_1 = entry[1][0]
            domain_2 = entry[1][1]
            data_reference_df.loc[i, 'expl_2'] = domainInteraction_index['_'.join(sorted([domain_1, domain_2]))]
        _x = data_ID_to_matrix[_id]
        data_reference_df.loc[i, 'original_score'] = classifier_obj.predict_score_op(np.array([_x]))[0]

    data_reference_df['cur_score'] = data_reference_df['original_score'].values


    # To get random results
    # Randomization
    cur_df = data_reference_df.copy()
    cur_df = cur_df.sample(frac=1).reset_index(drop=True)
    cur_df = shuffle(cur_df).reset_index(drop=True)
    
    clf_obj = copy.deepcopy(classifier_obj)
    
    working_df=cur_df.copy(deep=True)
    ref_data_df=main_data_df.copy(deep=True)

    precision = []
    recall = []
    domain_list = list(domain_dims.keys())
    total_posCount = len(working_df.loc[working_df['label'] == 1])

    # -------------------------------------------------
    #  Main loop
    # -------------------------------------------------
    next_K_precision = []
    prev_discovered_count = 0
    BATCH_SIZE = batch_size
    ID_COL = 'PanjivaRecordID'
    discovered_df = pd.DataFrame( columns = list(working_df.columns))
    
    W = clf_obj.W.cpu().data.numpy()
    GD_obj = GD(
        num_coeff,
        emb_dim,
        interaction_type = interaction_type
    )
    GD_obj.set_original_W(W)
    num_batches  = len(data_reference_df)//batch_size
    zero_count = 0
    for batch_idx in tqdm(range(num_batches)):
        print('Batch : {}'.format(batch_idx+1))
        if batch_idx == 0:
            lr = 0.25
            max_iter = 1000
        else:
            lr = 1
            max_iter = 500

        cur = working_df.head(BATCH_SIZE).reset_index(drop=True)
        if len(cur) < 2:
            break
       
        _tail_count = len(working_df) - BATCH_SIZE
        tmp = working_df.tail(_tail_count).reset_index(drop=True)
        if len(tmp.loc[tmp['label']==1]) == 0 :
            zero_count +=1
            if zero_count > 5:
                next_K_precision.append(0)
                working_df = working_df.tail(_tail_count).reset_index(drop=True)
                continue
        else :
            zero_count = 0    
        # -----
        # Count( of discovered in the current batch ( at the top; defined by batch size )
        # -----
        cum_cur_discovered = prev_discovered_count + len(cur.loc[cur['label'] == 1])
        prev_discovered_count = cum_cur_discovered
        _recall = float(cum_cur_discovered) / total_posCount
        recall.append(_recall)

        x_ij = []
        x_entityIds = []

        flags = []  # Whether a pos anomaly or not
        terms = []  # Explanation terms
        discovered_df = discovered_df.append(cur, ignore_index=True)
        
        for i, row in discovered_df.iterrows():
            _mask = np.zeros(len(domainInteraction_index))
            if row['label'] == 1:
                _mask[row['expl_1']] = 1
                _mask[row['expl_2']] = 1
                flags.append(1)
                terms.append((row['expl_1'], row['expl_2'],))
            else:
                flags.append(0)
                terms.append(())
            id_value = row['PanjivaRecordID']
            x_ij.append(data_ID_to_matrix[id_value])

            row_dict = ref_data_df.loc[(ref_data_df[ID_COL] == id_value)].iloc[0].to_dict()
            x_entityIds.append([row_dict[d] for d in domain_list])

        x_entityIds = np.array(x_entityIds)
        x_ij = np.array(x_ij)

        updated_W = GD_obj.update_weight(
            flags,
            terms,
            x_ij,
            lr = lr,
            max_iter= max_iter
        )
        
        # ----------------------------------------------------
        # Update Model
        # ----------------------------------------------------
        clf_obj.update_W(updated_W)
        clf_obj.update_binary_VarW(x_entityIds, flags)

        _tail_count = len(working_df) - BATCH_SIZE
        working_df = working_df.tail(_tail_count).reset_index(drop=True)


        # Obtain scores
        x_ij_test = []
        x_entityIds = fetch_entityID_arr_byList(
            ref_data_df,
            working_df['PanjivaRecordID'].values.tolist()
        )
        for _id in working_df['PanjivaRecordID'].values:
            x_ij_test.append(data_ID_to_matrix[_id])

        x_ij_test = np.array(x_ij_test)

        new_scores = clf_obj.predict_bEF(x_entityIds, x_ij_test)

        old_scores = working_df['cur_score'].values
        _delta = new_scores - old_scores
        working_df['delta'] = new_scores
        working_df = working_df.sort_values(by='delta', ascending=False)
        working_df = working_df.reset_index(drop=True)
        
        tmp = working_df.head(check_next)
        _labels = tmp['label'].values
        res = len(np.where(_labels == 1)[0])
        _precison = res / check_next
        next_K_precision.append(_precison)
        
    return next_K_precision





