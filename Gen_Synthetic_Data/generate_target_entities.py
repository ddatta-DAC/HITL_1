import pandas as pd
import numpy as np
import os
import sys

sys.path.append('./../..')
sys.path.append('./..')
import glob
from tqdm import tqdm
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

id_col = 'PanjivaRecordID'
import networkx as nx
import operator
import collections
import argparse

CONFIG = None
DIR_LOC = None
CONFIG = None
CONFIG_FILE = 'config.yaml'
save_dir = None
id_col = 'PanjivaRecordID'
use_cols = None
freq_bound = None
attribute_columns = None


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
    return


# ====================================
#
# ====================================
def main(
        DIR,
        init_seed_count=10,
        min_oh_nbr_count=2,
        seed_degree_upper_bound=None
):
    global DATA_SOURCE
    global save_dir

    company_cols = ['ConsigneePanjivaID', 'ShipperPanjivaID']
    company_col_abbr = {'C': 'ConsigneePanjivaID', 'S': 'ShipperPanjivaID'}
    df = pd.read_csv(os.path.join(DATA_SOURCE, 'train_data.csv'), low_memory=False, index_col=None)
    df_subset = df[company_cols].groupby(company_cols).size().reset_index(name='count').sort_values(by='count',
                                                                                                    ascending=False)

    df_subset['ConsigneePanjivaID'] = df_subset['ConsigneePanjivaID'].apply(
        lambda x: 'C' + str(x)
    )

    df_subset['ShipperPanjivaID'] = df_subset['ShipperPanjivaID'].apply(
        lambda x: 'S' + str(x)
    )

    B = nx.Graph()

    B.add_nodes_from(set(df_subset['ConsigneePanjivaID'].values), bipartite=0)
    B.add_nodes_from(set(df_subset['ShipperPanjivaID'].values), bipartite=1)

    # Add edges
    edges = []
    for i, j, k in zip(df_subset['ConsigneePanjivaID'].values, df_subset['ShipperPanjivaID'].values,
                       df_subset['count'].values):
        edges.append((i, j, {'weight': k}))

    B.add_edges_from(edges)
    print(' IS the bipartite graph of Comapnies connected ? ', nx.is_connected(B))
    component_id = 0
    components = {}
    component_size_dict = {}

    for c in nx.connected_components(B):
        components[component_id] = c
        component_size_dict[component_id] = len(c)
        component_id += 1

    component_size_dict = sorted(component_size_dict.items(), key=operator.itemgetter(1), reverse=True)

    max_component = components[component_size_dict[0][0]]
    subgraph = B.subgraph(max_component)
    subgraph.number_of_edges()
    subgraph.number_of_nodes()
    degree_sequence = sorted([d for n, d in subgraph.degree()], reverse=True)  # degree sequence
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())
    try:
        fig, ax = plt.subplots()
        plt.bar(deg, cnt, width=0.80, color="b")
        plt.title("Degree Histogram")
        plt.ylabel("Count")
        plt.xlabel("Degree")

        plt.show()
    except:
        pass

    degrees = [d for n, d in subgraph.degree()]
    if seed_degree_upper_bound is None:
        seed_degree_upper_bound = np.percentile(degrees, 75)

    # -----------------------------
    # Select the set of seeds
    # -----------------------------
    # Per type seed count

    min_seed_degree = min_oh_nbr_count + 1
    max_seed_degree = min_seed_degree + 1
    print(' Min, max of seed degree', min_seed_degree, max_seed_degree)
    node_dict = {}
    for n in subgraph.nodes():
        _type = n[0]
        if _type in node_dict.keys():
            node_dict[_type].append(n)
        else:
            node_dict[_type] = [n]

    seed_dict = {}
    for _type, _list in node_dict.items():
        _list = [n for n in _list if subgraph.degree(n) >= min_seed_degree and subgraph.degree(n) <= max_seed_degree]
        seed_dict[_type] = np.random.choice(_list, replace=False, size=init_seed_count)

    # ==================================================
    # Find one hop neighbors of these seeds
    # Sample from them - to be labelled as "positive"
    # ==================================================

    def find_one_hop_neigbbor(G, target_node, degree_upper_bound, exclsion_list=None, count=-1):
        onehop_nbrs = []
        for n in G.neighbors(target_node):
            for n1 in G.neighbors(n):
                if target_node not in G.neighbors(n1) and n1 != target_node:
                    if subgraph.degree(n1) > degree_upper_bound:
                        continue
                    if exclsion_list is not None and n1 not in exclsion_list:
                        onehop_nbrs.append(n1)
                    elif exclsion_list is None:
                        onehop_nbrs.append(onehop_nbrs)
        onehop_nbrs = set(onehop_nbrs)
        onehop_nbrs = list(onehop_nbrs)
        if count == -1: count = len(onehop_nbrs)
        try:
            return np.random.choice(onehop_nbrs, count, replace=False)
        except:
            return onehop_nbrs

    C_onehop_nbrs = []
    for nc in seed_dict['C']:
        ohn = find_one_hop_neigbbor(subgraph, nc, seed_degree_upper_bound, exclsion_list=seed_dict['S'],
                                    count=min_oh_nbr_count)
        C_onehop_nbrs.extend(ohn)

    S_onehop_nbrs = []
    for nc in seed_dict['S']:
        ohn = find_one_hop_neigbbor(subgraph, nc, seed_degree_upper_bound, exclsion_list=seed_dict['C'],
                                    count=min_oh_nbr_count)
        S_onehop_nbrs.extend(ohn)

    seed_onehop_nbrs = {
        'S': S_onehop_nbrs,
        'C': C_onehop_nbrs
    }

    seed_bipartite_nbrs = {'C': [], 'S': []}

    for _type in ('C', 'S'):
        nbrs = []
        for n in seed_dict[_type]:
            nbrs.extend([n1 for n1 in subgraph.neighbors(n) if subgraph.degree(n1) <= seed_degree_upper_bound])
        nbrs = list(set(nbrs))
        # get C or S
        n_type = nbrs[0][0]
        seed_bipartite_nbrs[n_type] = nbrs

    # ==================================
    # Accumulate the nodes of each type
    # These are the ones that are "Bad"
    # ==================================
    target_nodes_dict = {_: [] for _ in company_col_abbr.values()}

    for _type, _list in seed_dict.items():
        _type = company_col_abbr[_type]
        target_nodes_dict[_type].extend(_list)

    for _type, _list in seed_bipartite_nbrs.items():
        _type = company_col_abbr[_type]
        target_nodes_dict[_type].extend(list())

    for _type, _list in seed_onehop_nbrs.items():
        _type = company_col_abbr[_type]
        target_nodes_dict[_type].extend(_list)

    for _type in target_nodes_dict.keys():
        target_nodes_dict[_type] = set([int(_[1:]) for _ in target_nodes_dict[_type]])
    save_file = os.path.join(save_dir, 'seed_nodes.pkl')
    # Save in a pickle file
    with open(save_file, "wb") as fh:
        pickle.dump(target_nodes_dict, fh, pickle.HIGHEST_PROTOCOL)

    # ============================
    return target_nodes_dict


parser = argparse.ArgumentParser()
parser.add_argument(
    '--DIR', choices=['us_import1', 'us_import2', 'us_import3'],
    default='us_import1'
)

args = parser.parse_args()
DIR = args.DIR
set_up_config(
    DIR
)
target_nodes_dict = main(
    DIR,
    init_seed_count=25,
    min_oh_nbr_count=3
)