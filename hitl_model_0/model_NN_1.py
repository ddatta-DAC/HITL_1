import faiss
import numpy as np
import pandas as pd
import os


# -----------------------------------
# Build Nearest Neighbor Query Index
# -----------------------------------
def get_index_object(emb_array):
    index = faiss.IndexFlatL2(emb_array.shape[1])  # build the index
    print('Building FAISS index. Index trained ? ', index.is_trained)
    index.add(emb_array.astype(np.float32))
    print(' Size of index ::', index.ntotal)
    return index

# -------------------------------------
def main_process(
        df_cur,
        label_top_k,
        KNN_K,
        index,
        white_list_entities,
        precision_check_points,
        emb_array,
        anomaly_perc_threshold,
        ID_COL = 'PanjivaRecordID',
        attr_consignee= 'ConsigneePanjivaID',
        attr_shipper='ShipperPanjivaID'
):
    precision_check_range = precision_check_points
    white_list = white_list_entities
    print('label_top_k ', label_top_k)
    df_cur = df_cur.sort_values(by='score', ascending=False)
    min_score = np.percentile(df_cur['score'], anomaly_perc_threshold)
    seen_ids = []
    count = 0
    labelled_df = None
    idx = 0

    for i, row in df_cur.iterrows():
        _id = int(row[ID_COL])
        idx = i
        seen_ids.append(_id)

        if row['label'] == 1:
            count += 1
            if count > label_top_k:
                break

    seen_ids = set(seen_ids)
    labelled_df = df_cur.iloc[:idx, :]
    unlabelled_df = df_cur.iloc[idx:, :]
    orig_target_df = unlabelled_df.copy().reset_index(drop=True)

    tmp_N = labelled_df.loc[labelled_df['label'] != 1]
    tmp_P = labelled_df.loc[labelled_df['label'] == 1]
    neg_entities = tmp_N[attr_consignee].values.tolist() + tmp_N[attr_shipper].values.tolist()
    pos_entities = tmp_P[attr_consignee].values.tolist() + tmp_P[attr_shipper].values.tolist()
    pos_entities = list(sorted(set(pos_entities)))
    neg_entities = list(sorted(set(neg_entities)))

    print(' - / +', len(set(neg_entities)), len(set(pos_entities)))
    query_P = (emb_array[pos_entities]).astype(np.float32)
    D, nn_index_P = index.search(query_P, KNN_K)
    nn_index_P = nn_index_P[:, :]

    query_N = (emb_array[neg_entities]).astype(np.float32)
    D, nn_index_N = index.search(query_N, KNN_K)
    nn_index_N = nn_index_N[:, :]

    candidates = unlabelled_df.loc[unlabelled_df['score'] >= min_score]
    # first see records where both match +
    df1 = candidates.loc[candidates[attr_consignee].isin(pos_entities) & (candidates[attr_shipper].isin(pos_entities))]
    df1.loc[:, 'dynamic_score'] = 1

    # first see records where both match -
    df2 = candidates.loc[candidates[attr_consignee].isin(neg_entities) & (candidates[attr_shipper].isin(neg_entities))]
    df2.loc[:, 'dynamic_score'] = 0

    # first see records where one matches -
    df3 = candidates.loc[((candidates[attr_consignee].isin(white_list)) | (candidates[attr_shipper].isin(white_list)))]
    df3.loc[:, 'dynamic_score'] = 0.1

    # ------------------------------------------
    # df1 at top
    # df2 + df_3 at end
    # We use nearest neighbor based filters in the sizeable number of records in between
    # ------------------------------------------
    id_list = df1[ID_COL].values.tolist() + df2[ID_COL].values.tolist() + df3[ID_COL].values.tolist()
    id_list = list(set(id_list))
    df4 = candidates.loc[~candidates[ID_COL].isin(id_list)]
    _P = nn_index_P[:, 0].reshape(-1)
    nn_index_P = nn_index_P[:, 1:].reshape(-1)
    nn_index_N = nn_index_N.reshape(-1)

    df4_0 = df4.loc[
        ((df4[attr_consignee].isin(pos_entities)) & (df4[attr_shipper].isin(nn_index_P))) |
        ((df4[attr_consignee].isin(nn_index_P)) & (df4[attr_shipper].isin(pos_entities)))
        ]

    df4 = df4_0.loc[
        ~(df4_0[attr_consignee].isin(nn_index_N)) & (df4[attr_shipper].isin(nn_index_N))
        ]

    df4.loc[:, 'dynamic_score'] = 0.75
    id_list = df1[ID_COL].values.tolist() + df2[ID_COL].values.tolist() + df3[ID_COL].values.tolist() + df4[
        ID_COL].values.tolist()
    df5 = candidates.loc[~candidates[ID_COL].isin(id_list)]
    df0 = pd.concat([df1, df4, df5, df3, df2])
    new_df = df0.sort_values(by='dynamic_score', ascending=False)

    precision = []
    for k in precision_check_range:
        tmp = new_df.head(k)
        correct = tmp.loc[tmp['label'] == 1]
        prec = len(correct) / len(tmp)
        print(' precision at next top {:3d} : {:.3f}'.format(k, prec))
        precision.append(prec)

    return precision