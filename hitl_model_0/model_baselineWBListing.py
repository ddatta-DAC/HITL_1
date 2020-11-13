import numpy as np
import pandas as pd
import os

def main_process(
        df_cur,
        label_top_k,
        precision_check_range,
        anomaly_perc_threshold,
        white_list_entities,
        ID_COL='PanjivaRecordID',
        attr_consignee='ConsigneePanjivaID',
        attr_shipper='ShipperPanjivaID'
):

    white_list = white_list_entities
    print('Labelling till {} records have been marked positive(relevant) ', label_top_k)
    df_cur = df_cur.sort_values(by='score', ascending=False)
    min_score = np.percentile(df_cur['score'], anomaly_perc_threshold)
    seen_ids = []
    count = 0
    labelled_df = None
    idx = 0
    df_cur = df_cur.reset_index(drop=True)

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

    # print(' - / +', len(set(neg_entities)), len(set(pos_entities)))
    candidates = unlabelled_df.loc[unlabelled_df['score'] >= min_score]

    # first see records where both match +
    df1 = candidates.loc[candidates[attr_consignee].isin(pos_entities) & (candidates[attr_shipper].isin(pos_entities))]
    df1.loc[:, 'dynamic_score'] = 1

    # first see records where both match -
    df2 = candidates.loc[candidates[attr_consignee].isin(neg_entities) & (candidates[attr_shipper].isin(neg_entities))]
    df2.loc[:, 'dynamic_score'] = 0

    # first see records where one matches -
    df3 = candidates.loc[((candidates[attr_consignee].isin(white_list)) | (candidates[attr_shipper].isin(white_list)))]
    df3.loc[:, 'dynamic_score'] = 0

    # At least one matches +, and the other is not part of  -
    df5_1 = candidates.loc[
        (candidates[attr_consignee].isin(pos_entities)) & ~(candidates[attr_shipper].isin(neg_entities))]
    df5_2 = candidates.loc[
        (candidates[attr_shipper].isin(pos_entities)) & ~(candidates[attr_consignee].isin(neg_entities))]
    df_5 = pd.concat([df5_1, df5_2])
    df5 = df_5.loc[~df_5[ID_COL].isin(df1)]
    df5 = df5.loc[~df5[ID_COL].isin(df3)]
    df5.loc[:, 'dynamic_score'] = 0.75

    # One of the entities has been marked negative
    df6 = candidates.loc[
        (candidates[attr_shipper].isin(neg_entities)) | (candidates[attr_consignee].isin(neg_entities))]
    df6 = df6.loc[~df6[ID_COL].isin(df2)]
    df6.loc[:, 'dynamic_score'] = 0.25

    id_list = df1[ID_COL].values.tolist() + df2[ID_COL].values.tolist() + df3[ID_COL].values.tolist() + df5[
        ID_COL].values.tolist() + df6[ID_COL].values.tolist()
    id_list = list(set(id_list))

    df4 = candidates.loc[~candidates[ID_COL].isin(id_list)]
    df4.loc[:, 'dynamic_score'] = 0.5
    df0 = pd.concat([df1, df5, df4, df6, df3, df2])
    new_df = df0

    precision = []
    for k in precision_check_range:
        tmp = new_df.head(k)
        correct = tmp.loc[tmp['label'] == 1]
        prec = len(correct) / len(tmp)
        print('precision at next top {:3d} : {:.3f}'.format(k, prec))
        precision.append(prec)
    return precision