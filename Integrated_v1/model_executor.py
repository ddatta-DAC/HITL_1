import pandas as pd
import os
import sklearn
import numpy as np
from sklearn.utils import shuffle
import sys
import json
from collections import Counter
import copy

sys.path.append('./../..')
sys.path.append('./..')
from hitl_model_2.record import record_class
from common_utils import utils
from hitl_model_2.onlineGD import onlineGD
from hitl_model_2.linear_model_v2 import linearClassifier_bEF
from tqdm import tqdm

explantions_file_path = None
embedding_data_path = None
serialID_mapping_loc = None
domain_dims = None
DIR = None
ID_COL = 'PanjivaRecordID'

def initialize(
        _DIR,
        _explanations_file_path,
        _serialID_mapping_loc,
        _embedding_data_path,
        _domain_dims,
):
    global explanations_file_path
    global embedding_data_path
    global serialID_mapping_loc
    global domain_dims
    global DIR

    explanations_file_path = _explanations_file_path
    embedding_data_path = _embedding_data_path
    serialID_mapping_loc = _serialID_mapping_loc
    domain_dims = _domain_dims
    DIR = _DIR
    return


def get_serialID_to_entityID():
    global serialID_mapping_loc
    idMapper_file = os.path.join(serialID_mapping_loc)
    mapping_df = pd.read_csv(idMapper_file, index_col=None)
    serialID_to_entityID = {}

    for i, row in mapping_df.iterrows():
        serialID_to_entityID[row['serial_id']] = row['entity_id']
    return serialID_to_entityID


def fetch_entityID_arr_byList(data_df, id_list):
    global domain_dims
    domain_list =  list(domain_dims.keys())
    ID_COL = 'PanjivaRecordID'
    data_df = data_df.copy(deep=True)
    data_df = data_df.loc[data_df[ID_COL].isin(id_list)]
    # Order of id_list has to be preserved!!!
    X = []
    for _id in id_list:
        _tmp = data_df.loc[data_df[ID_COL]==_id][domain_list].iloc[0].values.tolist()
        X.append(_tmp)
    return np.array(X).astype(int)




def execute_with_input(
        clf_obj,
        working_df,
        ref_data_df,
        domainInteraction_index,
        num_coeff,
        emb_dim,
        data_ID_to_matrix,
        check_next=10,
        batch_size=10
):
    global domain_dims
    global ID_COL

    BATCH_SIZE = batch_size
    working_df['delta'] = 0
    GD_obj = onlineGD(num_coeff, emb_dim, learning_rate=0.05)
    W = clf_obj.W.cpu().data.numpy()
    GD_obj.set_original_W(W)

    max_num_batches = len(working_df) // BATCH_SIZE + 1
    acc = []

    domain_list = list(domain_dims.keys())
    for batch_idx in tqdm(range(max_num_batches)):

        cur = working_df.head(BATCH_SIZE)
        flags = []  # Whether a pos anaomaly or not
        terms = []  # Explanation terms

        x_ij = []
        x_entityIds = []

        for i, row in cur.iterrows():
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
        final_gradient, _W = GD_obj.update_weight(
            flags,
            terms,
            x_ij
        )
        # ----------------------------------------------------
        # Update Model
        # ----------------------------------------------------
        clf_obj.update_W(_W)
        clf_obj.update_binary_VarW(x_entityIds, flags)

        _tail_count = len(working_df) - BATCH_SIZE
        working_df = working_df.tail(_tail_count).reset_index(drop=True)
        #         working_df = working_df.iloc[BATCH_SIZE:]
        if len(working_df) == 0:
            break

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
        old_scores = working_df[ 'cur_score'].values
        _delta = new_scores - old_scores
        working_df['delta'] = _delta
        working_df = working_df.sort_values(by='delta', ascending=False)
        working_df = working_df.reset_index(drop=True)
        tmp = working_df.head(check_next)
        _labels = tmp['label'].values

        res = len(np.where(_labels == 1)[0])
        _acc = res / check_next
        acc.append(_acc)
    return acc


def get_trained_classifier(
        X,
        y,
        num_domains,
        emb_dim,
        num_epochs=10000
):
    global domain_dims
    classifier_obj = linearClassifier_bEF(
        num_domains = num_domains ,
        emb_dim = emb_dim,
        num_epochs=num_epochs
    )

    classifier_obj.setup_binaryFeatures(
        domain_dims,
        binaryF_domains = ['ConsigneePanjivaID','ShipperPanjivaID']
    )

    # classifier_obj.fit_on_pos(X, np.ones(X.shape[0]),n_epochs=10000)
    classifier_obj.fit(X, y, log_interval=5000)
    classifier_obj.fit_on_pos( X, y, n_epochs=num_epochs//2, log_interval=1000)
    return classifier_obj




# ---------------------------
# Get records which are deemed nominal/normal
# ---------------------------
def obtain_normal_samples(
        df,
        min_count=2,
        company_columns=['PanjivaConsigneeID', 'PanjivaShipperID']
):
    global domain_dims
    _df = df.copy()
    # Fetch companies records that are scored high(normal)
    threshold1 = np.percentile(df['score'], 70)
    df_high = df.loc[df['score'] >= threshold1]
    threshold2 = np.percentile(df['score'], 10)
    df_low = df.loc[df['score'] <= threshold2]

    # Discard companies occurring in the lowest 10 percentile
    for comp_attr in company_columns:
        filter_list = list(set(df_low[comp_attr]))
        df_high = df_high.loc[~df_high[comp_attr].isin(filter_list)]

    # filter records such that companies occur more than min_count times in the df_low
    filter_list = {}
    for comp_attr in company_columns:
        filter_list[comp_attr] = [k for k, v in Counter(df_high[comp_attr]).items() if v > min_count]

    for comp_attr in company_columns:
        df_high = df_high.loc[~df_high[comp_attr].isin(filter_list[comp_attr])]

    # Remove 'label' and 'score' from dataframe
    del df_high['label']
    del df_high['score']

    obj_list = []
    for i in tqdm(range(df_high.shape[0])):
        record_obj = record_class(
            _df.iloc[i].to_dict(),
            label=-1,
            is_unserialized=True
        )
        obj_list.append(record_obj)

    data_x = []
    for record_obj in obj_list:
        data_x.append(record_obj.x)
    data_x = np.stack(data_x)
    return data_x


def main_executor(
        combined_df,
        anomaly_threshold=7.5,
        feedback_batch_size=10,
        check_next = 10
):
    global explanations_file_path
    global embedding_data_path
    global serialID_mapping_loc
    global domain_dims
    global DIR
    global ID_COL

    # ============================================
    # setup objects

    serialID_to_entityID = get_serialID_to_entityID()
    record_class.__setup_embedding__(
        embedding_data_path,
        serialID_to_entityID,
        _normalize=True
    )

    emb_dim = record_class.embedding['HSCode'].shape[1]
    # main_data_df has the records with entity ids
    main_data_df = combined_df.copy()
    del main_data_df['label']
    del main_data_df['score']

    # -------------------------------------------
    #  Select lowest % of data as potential anomalies
    # -------------------------------------------
    threshold = np.percentile(combined_df['score'].values, anomaly_threshold)
    df_low = (combined_df.loc[combined_df['score'] <= threshold]).copy()
    df_allSelectedAnomalies = df_low.loc[df_low['label']==1][ID_COL].values.tolist()
    list_allTrueAnomalies = combined_df.loc[combined_df['label'] == 1][ID_COL].values.tolist()

    del df_low['label']
    del df_low['score']

    print('Explanations fetched from :: ', explantions_file_path)
    # Read in the explanations
    with open(explanations_file_path, 'rb') as fh:
        explanations = json.load(fh)

    explanations = {int(k): [sorted(_) for _ in v] for k, v in explanations.items()}
    num_domains = len(domain_dims)
    domain_idx = {
        e[0]: e[1] for e in enumerate(domain_dims.keys())
    }

    domainInteraction_index = {}
    k = 0
    for i in range(num_domains):
        for j in range(i + 1, num_domains):
            domainInteraction_index['_'.join((domain_idx[i], domain_idx[j]))] = k
            k += 1

    data_x = []
    data_id = []
    data_label = []
    data_ID_to_matrix = {}

    obj_list = []
    for i in tqdm(range(df_low.shape[0])):
        _id = df_low.iloc[i][ID_COL]
        if _id in list_allTrueAnomalies:
            label = 1
        else:
            label = 0
        obj = record_class(
            df_low.iloc[i].to_dict(),
            _label=label,
            is_unserialized=True
        )
        obj_list.append(obj)

    for _obj in obj_list:
        data_x.append(_obj.x)
        data_id.append(_obj.id)
        data_label.append(_obj.label)
        data_ID_to_matrix[_obj.id] = _obj.x

    data_x = np.stack(data_x)
    data_label = np.array(data_label)
    data_id = np.array(data_id)

    # Relevant(Possible) anomalies
    X_0 = data_x
    # Nominal data
    X_1 = obtain_normal_samples(combined_df)
    y_0 = np.ones(X_0.shape[0])
    y_1 = -1 * np.ones(X_1.shape[0])

    X = np.vstack([X_0, X_1])
    y = np.hstack([y_0, y_1])

    # This should be n*(n-1)/2 where n is the number of domains
    num_coeff = len(domainInteraction_index)

    # =================
    # Initial Classifier
    # ===================
    classifier_obj = get_trained_classifier(X, y, num_domains, emb_dim)
    W = classifier_obj.W.cpu().data.numpy()
    emb_dim = W.shape[-1]

    # ----------------------------------------
    # classifier_obj.predict_score_op(X_0)
    # Create a reference dataframe  :: data_reference_df
    # ----------------------------------------

    data_reference_df = pd.DataFrame(
        data = np.vstack([data_id, data_label]).transpose(),
        columns = ['PanjivaRecordID', 'label']
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
    results_with_input = pd.DataFrame(columns=['idx', 'precision'])

    cur_df = data_reference_df.copy()
    cur_df = shuffle(cur_df).reset_index(drop=True)

    precision = execute_with_input(
        clf_obj=copy.deepcopy(classifier_obj),
        working_df=cur_df,
        ref_data_df=main_data_df,
        domainInteraction_index=domainInteraction_index,
        num_coeff=num_coeff,
        emb_dim=emb_dim,
        data_ID_to_matrix=data_ID_to_matrix,
        check_next=check_next,
        batch_size=feedback_batch_size
    )

    _tmpdf = pd.DataFrame([(e[0], e[1]) for e in enumerate(precision)], columns=['idx', 'precision'])
    results_with_input = results_with_input.append(
        _tmpdf, ignore_index=True
    )

    return results_with_input
