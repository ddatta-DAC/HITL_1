import numpy as np
import pandas as pd
import os
# --------------------------------
def main_process(
        df_cur,
        label_top_k,
        precision_check_range,
        anomaly_perc_threshold,
        ID_COL='PanjivaRecordID',
):

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


    precision = []
    for k in precision_check_range:
        top_k_target = orig_target_df.head(k)
        p = len(top_k_target.loc[top_k_target['label'] == 1]) / k
        precision.append(p)
    return precision,