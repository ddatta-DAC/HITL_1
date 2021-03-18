#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
import os
import sys
import glob
from sklearn.preprocessing import normalize

sys.path.append('')
sys.path.append('..')
from pathlib import Path
from tqdm import tqdm
import pickle
import copy
import json
import linear_model
import seaborn as sns
from matplotlib import pyplot as plt
import argparse
from record import record_class
import yaml
import time
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
explantions_file_path = None
embedding_data_path = None
serialID_mapping_loc = None
anomalies_pos_fpath = None
anomalies_neg_fpath = None
feedback_batch_size = None
top_K_count = None
CLASSIFIER_TYPE = 'LR'
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
        domain_dims = pickle.load(fh)
    return


# ---------------------------------------------------------------------------------
def get_serialID_to_entityID():
    global serialID_mapping_loc
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
    for i in tqdm(range(_df.shape[0])):
        obj = record_class(_df.iloc[i].to_dict(), -1)
        obj_list.append(obj)
    data_x = []
    for _obj in obj_list:
        data_x.append(_obj.x)
    data_x = np.stack(data_x)
    return data_x


def get_trained_classifier(
        X,
        y
):
    global CLASSIFIER_TYPE
    if CLASSIFIER_TYPE == 'LR':
        clf_obj = LogisticRegression(
            warm_start=True,
            class_weight='balanced'
        )

    if CLASSIFIER_TYPE =='SGC':
        clf_obj = SGDClassifier(
            loss='log',
            penalty='elasticnet',
            learning_rate='optimal',
            class_weight='balanced',
            warm_start=True
        )
    clf_obj.fit(X,y)
    return clf_obj
# -------------------------------------------
def retrain_classifier(
    clf_obj,
    X,
    y
):
    # Retrain with new samples
    if CLASSIFIER_TYPE == 'LR':
        clf_obj.fit(X,y)
    if CLASSIFIER_TYPE == 'SGC':
        clf_obj.fit(X, y)
    return clf_obj

# -------------------------------------------
def execute_with_input(
        clf_obj,
        working_df,
        data_ID_to_matrix,
        check_next=10,
        batch_size=10
):
    BATCH_SIZE = batch_size
    working_df['delta'] = 0
    max_num_batches = len(working_df) // BATCH_SIZE + 1

    precision = []
    for b in range(max_num_batches):
        cur = working_df.head(BATCH_SIZE)
        feedback_x = []
        feedback_y = []

        for i, row in cur.iterrows():
            feedback_y.append(row['label'])
            feedback_x.append(data_ID_to_matrix[row['PanjivaRecordID']])

        # Update weights
        clf_obj = retrain_classifier(
            clf_obj,
            feedback_x,
            feedback_y
        )

        working_df = working_df.iloc[BATCH_SIZE:]
        # Obtain scores
        x_test = []
        for _id in working_df['PanjivaRecordID'].values:
            x_test.append(data_ID_to_matrix[_id])

        x_test = np.array(x_test)
        target_class_idx = clf_obj.classes_.index(1)

        new_scores = clf_obj.predict_proba(x_test)[:,target_class_idx]
        working_df['delta'] = new_scores

        working_df = working_df.sort_values(by='delta', ascending=False)
        working_df = working_df.reset_index(drop=True)
        tmp = working_df.head(check_next)
        _labels = tmp['label'].values

        res = len(np.where(_labels == 1)[0])
        _precision = res / check_next
        precision.append(_precision)

    return precision

# -----------------------------------------------
def execute_without_input(
        working_df,
        check_next=10,
        batch_size=10
):
    BATCH_SIZE = batch_size
    working_df['delta'] = 0

    num_batches = len(working_df.loc[working_df['label'] == 1]) // BATCH_SIZE + 5
    acc = []
    for b in range(num_batches):
        working_df = working_df.iloc[BATCH_SIZE:]
        working_df = working_df.reset_index(drop=True)
        tmp = working_df.head(check_next)
        _labels = tmp['label'].values
        res = len(np.where(_labels == 1)[0])
        _acc = res / check_next
        acc.append(_acc)
    return acc


def plot_figure(df1, df2):
    global DIR
    global feedback_batch_size
    global top_K_count

    ts = str(time.time()).split('.')[0]
    plt.figure(figsize=[6, 4])
    plt.title('Accuracy in next {} samples| Iteration(batch) : {} samples'.format(top_K_count, feedback_batch_size))
    plt.xlabel('Batch index', fontsize=14)
    plt.ylabel('Accuracy in next {} samples'.format(top_K_count), fontsize=14)
    sns.lineplot(data=df1, x="idx", y="acc", markers=True, label='Input provided')
    sns.lineplot(data=df2, x="idx", y="acc", markers=True, label='No Input')
    plt.legend(fontsize=14)
    plt.grid()
    plt.savefig('{}_results_v1_{}.png'.format(DIR, ts))
    try:
        plt.show()
    except:
        pass
    plt.close()


def plot_figure(list_df1):
    global DIR
    global feedback_batch_size
    global top_K_count

    ts = str(time.time()).split('.')[0]
    plt.figure(figsize=[6, 4])
    plt.title('Accuracy in next {} samples| Iteration(batch) : {} samples'.format(top_K_count, feedback_batch_size))
    plt.xlabel('Batch index', fontsize=14)
    plt.ylabel('Accuracy in next {} samples'.format(top_K_count), fontsize=14)
    sns.lineplot(data=df1, x="idx", y="acc", markers=True, label='Input provided')
    sns.lineplot(data=df2, x="idx", y="acc", markers=True, label='No Input')
    plt.legend(fontsize=14)
    plt.grid()
    plt.savefig('{}_results_v1_{}.png'.format(DIR, ts))
    try:
        plt.show()
    except:
        pass
    plt.close()


def main_executor():
    global explantions_file_path
    global embedding_data_path
    global serialID_mapping_loc
    global anomalies_pos_fpath
    global anomalies_neg_fpath
    global domain_dims
    global test_data_serialized_loc
    global feedback_batch_size
    global top_K_count
    # ============================================
    anom_pos_df = pd.read_csv(anomalies_pos_fpath, index_col=None)
    anom_neg_df = pd.read_csv(anomalies_neg_fpath, index_col=None)
    # ============================================
    # setup objects

    serialID_to_entityID = get_serialID_to_entityID()
    record_class.__setup_embedding__(embedding_data_path, serialID_to_entityID, _normalize=True)
    emb_dim = record_class.embedding['HSCode'].shape[1]

    # -------------------------------------------
    obj_list = []
    for i in tqdm(range(anom_neg_df.shape[0])):
        obj = record_class(anom_neg_df.iloc[i].to_dict(), -1)
        obj_list.append(obj)

    for i in tqdm(range(anom_pos_df.shape[0])):
        obj = record_class(anom_pos_df.iloc[i].to_dict(), 1)
        obj_list.append(obj)


    # Read in the explantions
    # with open(explantions_file_path, 'rb') as fh:
    #     explanations = json.load(fh)
    # explanations = {int(k): [sorted(_) for _ in v] for k, v in explanations.items()}

    # num_domains = len(domain_dims)
    # domain_idx = {e[0]: e[1] for e in enumerate(domain_dims.keys())}
    domainInteraction_index = {}
    # k = 0
    # for i in range(num_domains):
    #     for j in range(i + 1, num_domains):
    #         domainInteraction_index['_'.join((domain_idx[i], domain_idx[j]))] = k
    #         k += 1

    data_x = []
    data_id = []
    data_label = []
    data_ID_to_matrix = {}

    for _obj in obj_list:
        data_x.append(_obj.x)
        data_id.append(_obj.id)
        data_label.append(_obj.label)
        data_ID_to_matrix[_obj.id] = _obj.x

    data_x = np.stack(data_x)
    data_label = np.array(data_label)
    data_id = np.array(data_id)

    idx = np.arange(len(data_id), dtype=int)
    np.random.shuffle(idx)

    data_x = data_x[idx]
    data_label = data_label[idx]
    data_id = data_id[idx]

    X_0 = data_x                    # Relevant anomalies
    X_1 = obtain_normal_samples()   # Nominal

    y_0 = np.ones(X_0.shape[0])
    y_1 = -1 * np.ones(X_1.shape[0])

    X = np.vstack([X_0, X_1])
    y = np.hstack([y_0, y_1])

    classifier_obj = get_trained_classifier(X, y)

    # classifier_obj.predict_score_op(X_0)
    # Create a reference dataframe  :: data_reference_df
    working_df = pd.DataFrame(
        data=np.vstack([data_id, data_label]).transpose(),
        columns=['PanjivaRecordID', 'label']
    )

    working_df['baseID'] = working_df['PanjivaRecordID'].apply(lambda x: str(x)[:-3])
    working_df['expl_1'] = -1
    working_df['expl_2'] = -1
    working_df['original_score'] = 1

    for i, row in working_df.iterrows():
        _id = int(row['PanjivaRecordID'])
        # if _id in explanations.keys():
        #     entry = explanations[_id]
        #     domain_1 = entry[0][0]
        #     domain_2 = entry[0][1]
        #     working_df.loc[i, 'expl_1'] = domainInteraction_index['_'.join(sorted([domain_1, domain_2]))]
        #     domain_1 = entry[1][0]
        #     domain_2 = entry[1][1]
        #     working_df.loc[i, 'expl_2'] = domainInteraction_index['_'.join(sorted([domain_1, domain_2]))]
        _x = data_ID_to_matrix[_id]
        working_df.loc[i, 'original_score'] = classifier_obj.predict_score_op(np.array([_x]))[0]

    working_df['cur_score'] = working_df['original_score'].values
    data_reference_df = working_df.copy()

    # To get random results
    results_with_input = pd.DataFrame(columns=['idx', 'acc'])
    results_no_input = pd.DataFrame(columns=['idx', 'acc'])

    cur_df = data_reference_df.copy()
    # Randomization
    cur_df = cur_df.sample(frac=1).reset_index(drop=True)
    precision = execute_with_input(
        clf_obj=copy.deepcopy(classifier_obj),
        working_df=cur_df,
        emb_dim=emb_dim,
        data_ID_to_matrix=data_ID_to_matrix,
        check_next=top_K_count,
        batch_size=feedback_batch_size
    )

    _tmpdf = pd.DataFrame([(e[0], e[1]) for e in enumerate(precision)], columns=['idx', 'precision'])
    results_with_input = results_with_input.append(
        _tmpdf, ignore_index=True
    )

    acc = execute_without_input(
        working_df=cur_df,
        check_next=top_K_count,
        batch_size=feedback_batch_size
    )
    _tmpdf = pd.DataFrame([(e[0], e[1]) for e in enumerate(acc)], columns=['idx', 'precision'])
    results_no_input = results_no_input.append(
        _tmpdf, ignore_index=True
    )
    return results_with_input


# ------------------------------------------------------ #


# -------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument(
    '--DIR',
    choices=['us_import1', 'us_import2', 'us_import3', 'us_import4', 'us_import5', 'us_import6'],
    default=None
)

parser.add_argument(
    '--feedback_size',
    type=int,
    default=10
)
parser.add_argument(
    '--top_K',
    type=int,
    default=10,
)

args = parser.parse_args()
DIR = args.DIR
feedback_batch_size = args.feedback_size
top_K_count = args.top_K
setup_config(DIR)

# --------------------------------

# def checkPerformance():
#     num_runs = 10
#     df1 = None
#     df2 = None
#     for n in range(num_runs):
#
#         results_with_input, results_no_input = main_executor()
#         if df1 is None:
#             df1 = results_with_input
#         else:
#             df1 = df1.append(results_with_input, ignore_index=True)
#         if df2 is None:
#             df2 = results_no_input
#         else:
#             df2 = df2.append(results_no_input, ignore_index=True)
#
#     plot_figure(df1,df2)
#     return
#
# checkPerformance()
