import pandas as pd
import numpy as np
import os
import sys
sys.path.append('./..')
from pathlib import Path
import yaml
import pickle
import glob
from itertools import combinations
from joblib import Parallel, delayed
from tqdm import tqdm
import multiprocessing as mp
import re
from collections import  Counter
import argparse
import json
DIR = None
DATA_LOC = None
train_data_loc = None
test_data_loc = None
anomaly_data_loc = None
domain_dims = None
dist_df_dict = None
reference_df = None

def set_up(_DIR):
    
    global DIR, DATA_LOC, train_data_loc, test_data_loc, anomaly_data_loc, domain_dims, dist_df_dict
 
    with open('config.yaml', 'r') as fh:
        config = yaml.safe_load(fh)
        
    DATA_LOC = config['DATA_LOC']
    pairWiseDist_dir = os.path.join(config['pairWiseDist_dir'],DIR)
    anomaly_data_loc = os.path.join(config['anomaly_data_loc'],DIR)
    train_data_loc = os.path.join(DATA_LOC, DIR, 'train_data.csv')
    test_data_loc = os.path.join(DATA_LOC, DIR, 'test_data.csv')
    with open(os.path.join(DATA_LOC, DIR, 'domain_dims.pkl'.format(DIR)), 'rb')  as fh:
        domain_dims = pickle.load(fh)
    dist_df_dict = {}
    for _file in glob.glob(os.path.join(config['pairWiseDist_dir'],DIR,'**.csv')):
         
        _filename = os.path.split(_file)[-1].split('.')[0]
        _parts = _filename.split('_')
        key = (_parts[1],_parts[2])
     
        dist_df_dict[key] = pd.read_csv(_file,index_col=None)    
    return

def analyze_record(
    record,
    top = 5
):
    global reference_df
    global domain_dims
    global dist_df_dict 
    tqdm._instances.clear()
    try:
        ref_row = reference_df.loc[reference_df['PanjivaRecordID']==record['PanjivaRecordID']].iloc[0]
    except:
        _id =  int( str(record['PanjivaRecordID']) + '0' )
        ref_row = reference_df.loc[reference_df['PanjivaRecordID']==_id].iloc[0]
        
    wrong_domains = {}
    for d in domain_dims.keys():
        if ref_row[d]!= record[d]:
            wrong_domains[d] = record[d]
    
    # format domain1,domain2, dist
    record_entDist = []
    for pair in combinations(list(domain_dims.keys()),2):
        pair = sorted(pair)
        d1,d2 = pair[0],pair[1]
        key = (d1,d2)
        tmp_df = dist_df_dict[key]
        e1 = int(record[d1])
        e2 = int(record[d2])
        
        _dist = tmp_df.loc[(tmp_df[d1]==e1)&(tmp_df[d2]==e2)]['dist'].values[0]
        record_entDist.append([d1,d2,_dist]) 
        
    result = list(sorted(record_entDist, key = lambda x: x[2], reverse=True))
    wrong_domain_list = set(list(wrong_domains.keys()))
    for item in result[:top]:
        if len(wrong_domain_list.intersection(set(item[:2]))) > 0 : 
            return True
    return False



def main():
    
    global DIR, DATA_LOC, train_data_loc, test_data_loc, anomaly_data_loc, domain_dims, dist_df_dict,reference_df
    pos_anomalies = pd.read_csv(os.path.join(anomaly_data_loc, 'pos_anomalies.csv') ,index_col=None).reset_index(drop=True)
    neg_anomalies = pd.read_csv(os.path.join(anomaly_data_loc, 'neg_anomalies.csv') ,index_col=None)
    reference_df = pd.read_csv(train_data_loc, index_col=None).reset_index(drop=True)
    neg_anomalies['PanjivaRecordID'] = neg_anomalies['PanjivaRecordID'].apply(lambda x : int(str(x)[:-3]) ).reset_index(drop=True)
    pos_anomalies['PanjivaRecordID'] = pos_anomalies['PanjivaRecordID'].apply(lambda x : int(re.subn('00\d{1,2}$', '', str(x))[0]) )

    anomalies = neg_anomalies.append(pos_anomalies, ignore_index=True)
    anomalies = anomalies.reset_index(drop=True)
    top_Values= [ 1,2,3,4,5 ]
    results = {}
    for top in top_Values:
        res_neg = Parallel(n_jobs=mp.cpu_count())(
            delayed(analyze_record)(neg_anomalies.iloc[i],top)  for i in tqdm(range(neg_anomalies.shape[0])))

        res_pos = Parallel(n_jobs = mp.cpu_count())(
            delayed(analyze_record)(pos_anomalies.iloc[i])  for i in tqdm(range(pos_anomalies.shape[0])))

        res = res_neg + res_pos
        _dict = Counter(res)
        acc = _dict[True]/(_dict[True]+_dict[False])
        results[top] = acc
    result_path = os.path.join('results',DIR)
    Path(result_path).mkdir(exist_ok=True, parents=True)  
    result_path =os.path.join(result_path,'result.json')
    with open(result_path, "w") as fh:  
        json.dump(results, fh) 
    return results


parser = argparse.ArgumentParser()
parser.add_argument(
    '--DIR',
    choices=['us_import1', 'us_import2', 'us_import3', 'us_import4'],
    default=None
)


# ----------------------------------------
args = parser.parse_args()
DIR = args.DIR
set_up(DIR)
main()