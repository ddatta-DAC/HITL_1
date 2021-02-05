import os
import pandas as pd
import sys

sys.path.append('./..')
sys.path.append('./../..')

import numpy as np
import pickle
from tqdm import tqdm
from ApE import model as ape_model
from MEAD import model_AD_1 as mead_model
import yaml
from common_utils import borda_count

ID_COL = 'PanjivaRecordID'
DATA_LOC = './../generated_data_v1'
config = None
DIR = 'us_import1'

def get_domain_dims(DIR):
    global DATA_LOC
    with open(os.apth.join(DATA_LOC, '{}/domain_dims.pkl'.format(DIR)), 'rb')  as fh:
        domain_dims = pickle.load(fh)
    return domain_dims

def setup_up(DIR):
    global config
    config_file = 'ensemble_config.yaml'
    with open(config_file, 'r') as fh:
        config = yaml.safe_load(fh)
    config = config[DIR]
    return


def main():
    global DIR
    global config
    global ID_COL
    test_df = pd.read_csv(
        os.path.join(DATA_LOC, '/{}/stage_2/test_serialized.csv'.format(DIR)),
        index_col=None
    )
    domain_dims = get_domain_dims(DIR)
    entity_count = sum(domain_dims.values())
    id_list_normal = test_df[ID_COL].values.tolist()
    del test_df[ID_COL]
    test_x = test_df.values

    # -------------------------------------------------
    # Positive anomalies

    anomalies_src_path = os.path.join(DATA_LOC, 'generated_anomalies/{}'.format(DIR))
    test_df_p = pd.read_csv(os.path.join(anomalies_src_path, 'pos_anomalies.csv'), index_col=None)
    id_list_Pos = test_df_p[ID_COL].values.tolist()
    del test_df_p[ID_COL]
    test_x_Pos = test_df_p.values
    # scores_2 = model.score_samples(test_xp)

    # -------------------------------------------------
    # Negative anomalies

    test_df_n = pd.read_csv(os.path.join(anomalies_src_path, 'neg_anomalies.csv'), index_col=None)
    id_list_Neg = test_df_n[ID_COL].values.tolist()
    del test_df_n[ID_COL]

    test_x_Neg = test_df_n.values
    # scores_3 = model.score_samples(test_xn)

    for model_type, _list in config['models']:

        if model_type == 'ape':
            for _dict in _list:
                emb_dim = _list['emb_list']
                model_container_obj = ape_model.APE_container(
                    emb_dim=emb_dim,
                    domain_dims=domain_dims
                )
                model_container_obj.load_model(_dict['file'])

        if model_type == 'mead':
            for _dict in _list:
                emb_dim = _list['emb_list']
                model_container_obj = mead_model.AD_model_container(
                    emb_dim=emb_dim,
                    entity_count = entity_count
                )
                model_container_obj.load_model(_dict['file'])




    # label_list_normal = [0 for _ in range(len(scores_1))]
    # label_list_p = [1 for _ in range(len(scores_2))]
    # label_list_n = [-1 for _ in range(len(scores_3))]
    # scores = scores_1 + scores_2 + scores_3
    # id_list = id_list_normal + id_list_p + id_list_n
    # labels = label_list_normal + label_list_p + label_list_n
    # data = {'label': labels, 'score': scores , 'PanjivaRecordID': id_list}
    # df = pd.DataFrame(data)
setup_up(DIR)
print(config)