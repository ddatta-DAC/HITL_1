import pandas as pd
import numpy as np
import os
import sys
from networkx import bipartite

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
from networkx.algorithms import community
from common_utils.utils import remove_spurious_coOcc

CONFIG = None
DIR_LOC = None
CONFIG = None
CONFIG_FILE = 'config.yaml'
save_dir = None
id_col = 'PanjivaRecordID'
use_cols = None
freq_bound = None
attribute_columns = None
DIR = None
CUT_OFF = None


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

# =============================
# Main utility function
# =============================
def get_clusters(G, comm, max_pairs=1, max_indirect_nbr_count=3):
    import operator
    global CUT_OFF
    DEGREE_LB = 3
    DEGREE_UB = CUT_OFF

    sg_obj = G.subgraph(comm)
    edgeWt_dict = {}
    for e in sg_obj.edges():
        edgeWt_dict[e] = sg_obj.get_edge_data(e[0], e[1])['weight']
        # 50% of weights

    wt_lb = 3
    wt_ub = 10
    candidate_edges = {}

    for edge, wt in edgeWt_dict.items():
        if wt <= wt_lb or wt > wt_ub: continue
        if sg_obj.degree(e[0]) < DEGREE_LB and sg_obj.degree(e[1]) < DEGREE_LB: continue
        if sg_obj.degree(e[0]) > DEGREE_UB or sg_obj.degree(e[1]) > DEGREE_UB: continue

        #     print(edge, wt ,(sg_obj.degree(e[0]),sg_obj.degree(e[1])))
        candidate_edges[edge] = wt * ((sg_obj.degree(e[0]) + sg_obj.degree(e[1])) / 2) / DEGREE_UB
    candidate_edges = sorted(candidate_edges.items(), key=operator.itemgetter(1), reverse=True)
    # --------------------
    np.random.shuffle(candidate_edges)
    count = 0

    marked_edges = []
    for edge_item in candidate_edges:
        node1 = edge_item[0][0]
        node2 = edge_item[0][1]

        n1_nbr = None
        n2_nbr = None
        try:
            n1_nbr = np.random.choice([_ for _ in sg_obj.neighbors(node1) if
                                       sg_obj.degree(_) > DEGREE_LB and sg_obj.degree(_) < DEGREE_UB and _ != node2],
                                      1)[0]
            n2_nbr = np.random.choice([_ for _ in sg_obj.neighbors(node2) if
                                       sg_obj.degree(_) > DEGREE_LB and sg_obj.degree(_) < DEGREE_UB and _ != node1],
                                      1)[0]
            # print('>>', n1_nbr, n2_nbr)
        except:
            continue
        if n1_nbr is None or n2_nbr is None:
            continue
        # find neighbors of n1_nbr and n2_nbr
        try:
            valid_nbrs = [N for N in sg_obj.neighbors(n1_nbr) if N != node1 and N not in list(sg_obj.neighbors(node2))]
            node1_nbr_2 = set(np.random.choice(valid_nbrs, max_indirect_nbr_count, replace=True))
            valid_nbrs = [N for N in sg_obj.neighbors(n2_nbr) if N != node2 and N not in list(sg_obj.neighbors(node1))]
            node2_nbr_2 = set(np.random.choice(valid_nbrs, max_indirect_nbr_count, replace=True))

        #             print(node1_nbr_2, node2_nbr_2)
        except:
            continue

        target_nodes = [node1, node2, n1_nbr, n2_nbr]
        target_nodes.extend(node1_nbr_2)
        target_nodes.extend(node2_nbr_2)

        new_subgraph = sg_obj.subgraph(target_nodes)
        # print(new_subgraph.edges(), new_subgraph.number_of_edges(), new_subgraph.number_of_nodes())
        marked_edges.extend(new_subgraph.edges())
        count += 1
        if count >= max_pairs: break
    return marked_edges


def check_degree_distribution(graph_obj):
    degree_sequence = sorted([d for n, d in graph_obj.degree()], reverse=True)
    try:
        plt.title('Boxplot of degree')
        plt.boxplot(degree_sequence)
        plt.show()
    except:
        pass
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


def get_largest_component_subgraph(graph_obj):
    component_id = 0
    components = {}
    component_size_dict = {}

    for c in nx.connected_components(graph_obj):
        components[component_id] = c
        component_size_dict[component_id] = len(c)
        component_id += 1
    component_size_dict = sorted(component_size_dict.items(), key=operator.itemgetter(1), reverse=True)
    # Get the largest connected component
    max_component = components[component_size_dict[0][0]]
    subgraph = graph_obj.subgraph(max_component)
    return subgraph


def main_process():
    global DIR
    global DATA_SOURCE
    global CUT_OFF
    global save_dir

    with open(os.path.join(DATA_SOURCE, 'domain_dims.pkl'), 'rb') as fh:
        domain_dims = pickle.load(fh)

    # =======================================
    company_cols = ['ConsigneePanjivaID', 'ShipperPanjivaID']
    company_col_abbr = {'C': 'ConsigneePanjivaID', 'S': 'ShipperPanjivaID'}
    # -----------------------------------------------------------
    # Before creating community out of nodes - this flag modulates - whether to prune high degree edges premptively
    # setting to false  so that "cluster/community" is better captured
    # ------------------------------------------------------------
    PRE_PRUNE_BY_DEGREE = False
    # -----------------------------------------------------------
    # Total anomalies should be approximately between 5 to 10%
    # -----------------------------------------------------------
    ANOM_PERC_THRESHOLD_LB = 2.00
    ANOM_PERC_THRESHOLD_UB = 10.50
    # -----------------------------------------------------------
    # generate anomalies from top k communities by size
    TOP_K_COMMUNITIES = 20
    # ------------------------------------------------------------

    train_df = pd.read_csv(
        os.path.join(DATA_SOURCE, 'train_data.csv'),
        low_memory=False,
        index_col=None
    )
    attributes = [_ for _ in list(train_df.columns) if _ not in id_col]
    train_df = train_df.drop_duplicates(subset=attributes)
    df_subset = train_df[company_cols].groupby(
        company_cols).size().reset_index(
        name='count').sort_values(by='count', ascending=False)

    df_subset['ConsigneePanjivaID'] = df_subset['ConsigneePanjivaID'].apply(
        lambda x: 'C' + str(x)
    )

    df_subset['ShipperPanjivaID'] = df_subset['ShipperPanjivaID'].apply(
        lambda x: 'S' + str(x)
    )

    # --------------------
    # Create a bipartite graph
    # --------------------
    B = nx.Graph()
    B.add_nodes_from(set(df_subset['ConsigneePanjivaID'].values), bipartite=0)
    B.add_nodes_from(set(df_subset['ShipperPanjivaID'].values), bipartite=1)
    edges = []
    for i, j, k in zip(df_subset['ConsigneePanjivaID'].values,
                       df_subset['ShipperPanjivaID'].values,
                       df_subset['count'].values):
        edges.append((i, j, {'weight': k}))

    B.add_edges_from(edges)
    print(' IS the bipartite graph of Comapnies connected ? ', nx.is_connected(B))

    #  ---------------------------
    # Remove nodes that are in 90th percentile of degree or more
    # That is remove companies that are higly connected - since those are visible and deemed not suspicous/malacious
    # ----------------------------
    degree_sequence = sorted([d for n, d in B.degree()], reverse=True)
    CUT_OFF = int(np.percentile(degree_sequence, 90))
    print(' cut off degree ', CUT_OFF)

    # ====================================================================================
    # if PRE_PRUNE_BY_DEGREE == True
    # Remove nodes of high degree beforehand
    # ====================================================================================
    if PRE_PRUNE_BY_DEGREE == True:
        nodes_to_remove = [n for n, d in B.degree() if d > CUT_OFF]
        B1 = nx.Graph(B)
        B1.remove_nodes_from(nodes_to_remove)
        print('Number of nodes and edges', B1.number_of_edges(), B1.number_of_nodes())
    else:
        B1 = nx.Graph(B)

    # =====================================================================================
    # Largest connected component in the bipartite graph , after pruning high degree nodes
    # ======================================================================================
    B2 = get_largest_component_subgraph(B1)
    check_degree_distribution(B2)
    print('Number of nodes and edges', B2.number_of_edges(), B2.number_of_nodes())
    # ==========================================
    # Parititon the graph into communities
    # ==========================================

    communities = community.greedy_modularity_communities(B2)
    df_test = pd.read_csv(
        os.path.join(DATA_SOURCE, 'test_data.csv'),
        low_memory=False,
        index_col=None
    )
    df_test = df_test.drop_duplicates(subset=attribute_columns)
    df_test = remove_spurious_coOcc(
        df_test,
        train_df,
        domain_dims
    )
    while True:
        all_marked = []
        for i in range(0, TOP_K_COMMUNITIES):
            edges = get_clusters(B2, communities[i], max_pairs=1, max_indirect_nbr_count=5)
            all_marked.extend(edges)
        all_marked = [sorted(_) for _ in all_marked]
        record_count = 0
        for pair in all_marked:
            _C = int(pair[0][1:])
            _S = int(pair[1][1:])
            record_count += len(df_test.loc[
                                    (df_test['ConsigneePanjivaID'] == _C) &
                                    (df_test['ShipperPanjivaID'] == _S)]
                                )
        percentage = (record_count / len(df_test)) * 100
        print(' [  ======>  ', percentage, record_count, len(df_test),']')
        edge_list = all_marked
        print(' Length of edge list >> ', len(all_marked))
        # --- Found:  break
        if percentage >= ANOM_PERC_THRESHOLD_LB and percentage <= ANOM_PERC_THRESHOLD_UB:
            break
    print('Number of edges ', len(edge_list))
    result_edge_pairs = {}
    for d in company_col_abbr.values():
        result_edge_pairs[d] = []
    for e in edge_list:
        e1 = int(e[0][1:])
        d1 = company_col_abbr[e[0][0]]
        result_edge_pairs[d1].append(e1)
        e2 = int(e[1][1:])
        d2 = company_col_abbr[e[1][0]]
        result_edge_pairs[d2].append(e2)

    result_df = pd.DataFrame.from_dict(result_edge_pairs)
    print('Result dataframe, with edge list', len(result_df))
    # Save the results
    f_path = os.path.join(save_dir, 'seed_edges.csv')
    result_df.to_csv(f_path, index=None)

    return result_df


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
main_process()
