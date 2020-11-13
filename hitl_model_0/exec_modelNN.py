#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import torch
import os
import glob
import yaml
import pickle
import numpy as np
import matplotlib.pyplot as plt
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import sys
from tqdm import tqdm
sys.path.append('./..')
sys.path.append('./../..')
from pandarallel import pandarallel
pandarallel.initialize()

from scipy.special import softmax as SOFTMAX
import matplotlib.pyplot as plt
import argparse
from common_utils import utils

try:
    import model_NN_1 as model_NN_1
    import model_baselineNoInput as model_baselineNoInput
    import model_baselineWBListing as model_baselineWBListing
except:
    from . import model_NN_1
    from . import model_baselineNoInput
    from . import model_baselineWBListing

try:
    from common_utils import AD_result_fetcher
except:
    from .common_utils import AD_result_fetcher
from sklearn.metrics.pairwise import pairwise_distances_argmin
from time import time
from matplotlib import pyplot as plt
from sklearn.metrics import silhouette_score
from torch import nn
from torch.nn import functional as F
from torch import LongTensor as LT
from torch import FloatTensor as FT
from scipy.spatial.distance import cosine
from collections import Counter
import faiss
from sklearn.manifold import TSNE


def obtain_bipartite_embeddings():
    global DIR
    LOC = './saved_model_data'
    _files = sorted(glob.glob(os.path.join(LOC, DIR, '**.npy')))
    emb_dict = {}
    for file in _files:
        _domain = file.split('/')[-1].split('_')[0]
        emb_dict[_domain] = np.load(file)
    return emb_dict


# ---------------------------------------------
# Convert the df from serial_ID to entity_ID
# ---------------------------------------------
def convert_toEntityID(target_df):
    global bipartite_domains
    serialID_map_df = utils.fetch_idMappingFile(DIR)
    serialID_entityID_dict = {}

    for domain in set(bipartite_domains):
        if domain not in target_df.columns: continue
        tmp = serialID_map_df.loc[(serialID_map_df['domain'] == domain)]
        serial_id = tmp['serial_id'].values.tolist()
        entity_id = tmp['entity_id'].values.tolist()
        serialID_entityID_dict[domain] = {k: v for k, v in zip(serial_id, entity_id)}

    def convert_aux(val, domain):
        return serialID_entityID_dict[domain][val]

    domain_list = bipartite_domains
    for domain in tqdm(domain_list):
        target_df[domain] = target_df[domain].parallel_apply(convert_aux, args=(domain,))
    return target_df


# -----------------------------
# Create synthetic mapping
# So that ids are continuous.
# Also embeddings can be accessed using an numpy array
# -----------------------------
def convert_to_SynID(data_df, emb_dict):
    global bipartite_domains
    global DATA_LOC
    global DIR

    with open(os.path.join(DATA_LOC, DIR, 'domain_dims.pkl'), 'rb') as fh:
        domain_dims = pickle.load(fh)

    synID = 0
    cur = 0
    col_syn_id = []
    col_entity_id = []
    col_domain_names = []

    # ------------------
    for d in sorted(bipartite_domains):
        s = domain_dims[d]
        col_entity_id.extend(list(range(s)))
        col_domain_names.extend([d for _ in range(s)])
        tmp = np.arange(s) + cur
        tmp = tmp.tolist()
        col_syn_id.extend(tmp)
        cur += s

    data = {'domain': col_domain_names, 'entity_id': col_entity_id, 'syn_id': col_syn_id}
    synID_mapping_df = pd.DataFrame(data)

    # -------------------
    # Replace entity_id with synthetic id
    # -------------------
    mapping_dict = {}
    for domain in sorted(set(synID_mapping_df['domain'])):
        tmp = synID_mapping_df.loc[(synID_mapping_df['domain'] == domain)]
        syn_id = tmp['syn_id'].values.tolist()
        entity_id = tmp['entity_id'].values.tolist()
        mapping_dict[domain] = {k: v for k, v in zip(entity_id, syn_id)}

        def convert_aux(val, domain):
            return mapping_dict[domain][val]

    for domain in tqdm(bipartite_domains):
        data_df[domain] = data_df[domain].parallel_apply(convert_aux, args=(domain,))

    num_entities = len(synID_mapping_df)
    emb_array = np.zeros([num_entities, emb_dict[bipartite_domains[0]].shape[-1]])

    for dom in bipartite_domains:
        tmp = synID_mapping_df.loc[synID_mapping_df['domain'] == dom]
        synID = tmp['syn_id'].values
        entityID = tmp['entity_id'].values
        emb_array[synID] = emb_dict[dom][entityID]

    return data_df, emb_array, synID_mapping_df


def get_whiteListedEntities(working_df, assumed_normal_score_threshold, assumed_normal_entity_minCount):
    neg_label_cutoff_score = np.percentile(working_df['score'], assumed_normal_score_threshold)
    df2 = working_df.loc[working_df['score'] <= neg_label_cutoff_score]
    elements = df2[attr_consignee].values.tolist() + df2[attr_shipper].values.tolist()
    white_list_entities = []
    for k, count in Counter(elements).items():
        if count > assumed_normal_entity_minCount:
            white_list_entities.append(k)

    print(' Number of  white_list_entities', len(white_list_entities))
    return white_list_entities


# ============================================================================
parser = argparse.ArgumentParser()
parser.add_argument(
    '--DIR', choices=['us_import1', 'us_import2', 'us_import3', 'us_import4', 'us_import5'],
    default='us_import1'
)

args = parser.parse_args()
DIR = args.DIR

config_file = 'config.yaml'
with open(config_file, 'r') as fh:
    CONFIG = yaml.safe_load(fh)

MAX_ANOM_LABELS = CONFIG['MAX_ANOM_LABELS']  # How many  positive/relevant cases are marked by the annotator at max
KNN_K = CONFIG['KNN-K']
assumed_normal_entity_minCount = CONFIG['assumed_normal_entity_minCount']
assumed_normal_score_threshold = CONFIG['assumed_normal_score_threshold']
precision_check_points = CONFIG['precision_check_points']
attr_consignee = CONFIG['attribute_CONSIGNEE']
attr_shipper = CONFIG['attribute_SHIPPER']
ID_COL = CONFIG['ID_COL']
DATA_LOC = CONFIG['DATA_LOC']
precision_check_points = CONFIG['precision_check_points']

NEG_PERCENTILE_THRESHOLD = CONFIG['NEG_PERCENTILE_THRESHOLD']
labelled_results = AD_result_fetcher.read_in_AD_result(DIR)
ANOMALY_PERCENTILE_THRESHOLD = CONFIG['ANOMALY_PERCENTILE_THRESHOLD']
score_threshold = np.percentile(labelled_results['score'], CONFIG['ANOMALY_PERCENTILE_THRESHOLD'])

bipartite_domains = sorted([attr_consignee, attr_shipper])
bipartite_embeddings = obtain_bipartite_embeddings()
main_df = (labelled_results[[ID_COL, 'label', 'score'] + bipartite_domains]).copy()
df1 = convert_toEntityID(main_df.copy())
data_df, emb_array, synID_mapping_df = convert_to_SynID(df1.copy(), bipartite_embeddings)

# ------------------
# Type conversion : to ensure no bugs
# Drop index to avoid issues
# ------------------
working_df = data_df.copy()
working_df['dynamic_score'] = working_df['score'].values
working_df['PanjivaRecordID'] = working_df['PanjivaRecordID'].astype(int)
working_df['ConsigneePanjivaID'] = working_df['ConsigneePanjivaID'].astype(int)
working_df['ShipperPanjivaID'] = working_df['ShipperPanjivaID'].astype(int)
working_df = working_df.reset_index(drop=True)
# -----------------------------------
# Build Nearest Neighbor Query Index
# -----------------------------------

index = model_NN_1.get_index_object(
    emb_array=emb_array
)

white_list_entities = get_whiteListedEntities(
    working_df,
    assumed_normal_score_threshold,
    assumed_normal_entity_minCount
)

# ============================================================================
# Run the models
# ============================================================================
df_cur = working_df.copy()
res = []
for k in range(1, MAX_ANOM_LABELS + 1, 1):
    print(' ----------------- ')
    print('labelling {} positive entities and checking ...'.format(k))
    p0 = model_baselineNoInput.main_process(
        df_cur,
        label_top_k=k,
        precision_check_range=precision_check_points,
        anomaly_perc_threshold=ANOMALY_PERCENTILE_THRESHOLD,
    )
    df_cur = working_df.copy()
    p1 = model_baselineWBListing.main_process(
        df_cur, label_top_k=k,
        precision_check_range=precision_check_points,
        anomaly_perc_threshold=ANOMALY_PERCENTILE_THRESHOLD,
        white_list_entities=white_list_entities,
    )

    p2 = model_NN_1.main_process(
        df_cur=working_df.copy(),
        label_top_k=k,
        KNN_K=KNN_K,
        index=index,
        white_list_entities=white_list_entities,
        precision_check_points=precision_check_points,
        emb_array=emb_array,
        anomaly_perc_threshold=ANOMALY_PERCENTILE_THRESHOLD,
        ID_COL=ID_COL,
        attr_consignee=attr_consignee,
        attr_shipper=attr_shipper
    )
    res.append((p1, p2, p0))

'''
 Cant plot in all cases !
'''
x = precision_check_points

num_sub_plots = MAX_ANOM_LABELS
num_cols = 5
num_rows = int(math.ceil(MAX_ANOM_LABELS / num_cols))
c = 0
r = -1
plot_idx = []
for k in range(MAX_ANOM_LABELS):
    c = c % num_cols
    if c == 0: r += 1
    plot_idx.append((r, c))
    c = c + 1

fig, axs = plt.subplots(r + 1, c, figsize=(5 * num_cols, 5 * (r + 1)))
for k in range(1, MAX_ANOM_LABELS + 1, 1):
    p1, p2, p0 = res[k]
    _r = plot_idx[k][0]
    _c = plot_idx[k][1]
    axs[_r, _c].plot(x, p2, 'r', marker='*', label='Proposed')
    axs[_r, _c].plot(x, p1, 'b', marker='o', label='white(black)-list')
    axs[_r, _c].plot(x, p0, 'k', marker='p', label='no-input')
    axs[_r, _c].legend(fontsize=14)
    axs[_r, _c].set_title('K={} records marked positive'.format(k), fontsize=14)

for ax in axs.flat:
    ax.set(xlabel='Next records', ylabel='Precision')
for ax in axs.flat:
    ax.label_outer()
    #     ax.set_title('Number of positive anomalies marked vs Precision in next records\n\n')

# ---------------------------------------------- #
fname = 'results_{}.png'.format(DIR)
fig.savefig(fname)
print('Saving figure : ', fname)
try:
    plt.show()
except:
    pass
plt.close()
