import pandas as pd
import os
import sys
sys.path.append('../..')
sys.path.append('../../..')
import pickle
import numpy as np
import pandas as pd
import os
import sys
sys.path.append('../..')
sys.path.append('../../..')
import pickle
import numpy as np
# ================================

def read_in_AD_result(DIR, normalize_scores=True, anomalies_score_high=True):
    AD_output_location = './../AD_model/AD_output/{}'.format(DIR)
    fname = 'AD_output.csv'
    results_file = os.path.join(AD_output_location, fname)
    df = pd.read_csv(results_file, index_col=None, low_memory=False)
    results_df = df.sort_values(by='score')

    # ----------------------------------------------
    # Join it with teh actual data with the columns
    # ----------------------------------------------
    data_loc = './../generated_data_v1/{}/stage_2'.format(DIR)
    test_pos_file = 'test_pos_serialized.csv'
    test_neg_file = 'test_neg_serialized.csv'
    test_normal_file = 'test_normal_serialized.csv'

    df1 = pd.read_csv(os.path.join(data_loc, test_neg_file), low_memory=False, index_col=None)
    df2 = pd.read_csv(os.path.join(data_loc, test_pos_file), low_memory=False, index_col=None)
    df3 = pd.read_csv(os.path.join(data_loc, test_normal_file), low_memory=False, index_col=None)
    combined_df = df1.append(df2, ignore_index=True)
    combined_df = combined_df.append(df3, ignore_index=True)
    print(len(combined_df), len(results_df))
    # Combine them on the ID col
    ID_COL = 'PanjivaRecordID'
    df = combined_df.merge(results_df, how='inner', on=ID_COL)
    df = df.sort_values(by='score')

    if anomalies_score_high: normalize_scores = True
    if normalize_scores:
        _min = np.min(df['score'])
        _max = np.max(df['score'])

        def normalize(val, _min, _max):
            return (val - _min) / (_max - _min)

        df['score'] = df['score'].apply(normalize, args=(_min, _max))
    if anomalies_score_high:
        df['score'] = 1 - df['score']
        df = df.sort_values(by='score', ascending=False)
    return df

#
# DIR = 'us_import1'
# data_df = read_in_AD_result(DIR)
# np.percentile(data_df['score'], 2.5)
#
# data_df.loc[data_df['score'] <= np.percentile(data_df['score'], 5)]

