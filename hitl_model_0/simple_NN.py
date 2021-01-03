!/usr/bin/env python
# coding: utf-8

# In[121]:


import pandas as pd
import torch
import os
import sklearn
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import sys
from tqdm import tqdm
sys.path.append('./..')
sys.path.append('./../..')
from pandarallel import pandarallel
pandarallel.initialize()
import glob 
import yaml
import pickle
from scipy.special import softmax as SOFTMAX
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from common_utils import utils
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
from  sklearn.manifold import TSNE


# In[ ]:





# In[2]:


def obtain_bipartite_embeddings():
    global DIR
    LOC = './saved_model_data'
    _files = sorted(glob.glob(os.path.join(LOC,DIR, '**.npy')))
    emb_dict = {}
    for file in _files:
        _domain = file.split('/')[-1].split('_')[0]
        emb_dict[_domain] = np.load(file)
    return emb_dict


# In[3]:


# ---------------------------------------------
# Convert the df from serial_ID to entity_ID
# ---------------------------------------------
def convert_toEntityID(target_df):
    global bipartite_domains
    serialID_map_df = utils.fetch_idMappingFile(DIR)
    serialID_entityID_dict = {}
    
    for domain in set(bipartite_domains):
        if domain not in target_df.columns : continue
        tmp =  serialID_map_df.loc[(serialID_map_df['domain'] == domain)]
        serial_id = tmp['serial_id'].values.tolist()
        entity_id = tmp['entity_id'].values.tolist()
        serialID_entityID_dict[domain] = {k:v for k,v in zip(serial_id,entity_id)}
   
    def convert_aux(val, domain):
        return serialID_entityID_dict[domain][val]
    domain_list = bipartite_domains
    for domain in tqdm(domain_list):
        target_df[domain] = target_df[domain].parallel_apply(convert_aux, args=(domain,))
    return target_df


# In[ ]:





# In[4]:


# -----------------------------
# Create synthetic mapping 
# So that ids are continuous. 
# Also embeddings can be accessed using an numpy array
# -----------------------------  
def convert_to_SynID (data_df, emb_dict):
    global bipartite_domains
    global DATA_LOC
    global DIR
    
    with open(os.path.join(DATA_LOC, DIR, 'domain_dims.pkl'),'rb') as fh:
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
        tmp =  synID_mapping_df.loc[(synID_mapping_df['domain'] == domain)]
        syn_id = tmp['syn_id'].values.tolist()
        entity_id = tmp['entity_id'].values.tolist()
        mapping_dict[domain] = { k:v for k,v in zip(entity_id,syn_id) }
        def convert_aux(val, domain):
            return mapping_dict[domain][val]

    for domain in tqdm(bipartite_domains):
        data_df[domain] = data_df[domain].parallel_apply(convert_aux, args=(domain,))
    
    num_entities = len(synID_mapping_df)
    emb_array = np.zeros( [num_entities, emb_dict[bipartite_domains[0]].shape[-1]])
    
    for dom in bipartite_domains:
        tmp = synID_mapping_df.loc[synID_mapping_df['domain']==dom]
        synID = tmp['syn_id'].values
        entityID = tmp['entity_id'].values
        emb_array[synID] = emb_dict[dom][entityID]
        
    return data_df,  emb_array , synID_mapping_df


# In[5]:


config_file = 'config.yaml'
with open(config_file,'r') as fh:
    CONFIG = yaml.safe_load(fh)


# In[149]:


KNN = 6
WHITELIST_COUNT = 5
attr_consignee = CONFIG['attribute_CONSIGNEE'] 
attr_shipper = CONFIG['attribute_SHIPPER']
ID_COL = CONFIG['ID_COL']
DATA_LOC = CONFIG['DATA_LOC']
bipartite_domains = sorted([attr_consignee, attr_shipper])
NEG_PERCENTILE_THRESHOLD = CONFIG['NEG_PERCENTILE_THRESHOLD']


# In[11]:


DIR = 'us_import1'

labelled_results = AD_result_fetcher.read_in_AD_result(DIR)
ANOMALY_PERCENTILE_THRESHOLD  = CONFIG['ANOMALY_PERCENTILE_THRESHOLD']
score_threshold = np.percentile(labelled_results['score'],CONFIG['ANOMALY_PERCENTILE_THRESHOLD'])
print(CONFIG['ANOMALY_PERCENTILE_THRESHOLD'], score_threshold)
bipartite_embeddings = obtain_bipartite_embeddings()

main_df = (labelled_results[[ID_COL , 'label', 'score'] + bipartite_domains]).copy()
df1 = convert_toEntityID(main_df.copy())
data_df, emb_array , synID_mapping_df = convert_to_SynID (df1.copy(), bipartite_embeddings)


# In[120]:


# -----------------------------------
# Build Nearest Neighbor Query Index
# -----------------------------------
index = faiss.IndexFlatL2(32)   # build the index
print(index.is_trained)
index.add(emb_array.astype(np.float32))                 
print(index.ntotal)


# In[12]:


working_df = data_df.copy()
working_df['dynamic_score'] = working_df['score'].values
# ------------------
# Type conversion : to ensure no bugs
# ------------------
working_df['PanjivaRecordID'] = working_df['PanjivaRecordID'].astype(int)
working_df['ConsigneePanjivaID'] = working_df['ConsigneePanjivaID'].astype(int)
working_df['ShipperPanjivaID'] = working_df['ShipperPanjivaID'].astype(int)
working_df = working_df.reset_index(drop=True)


# In[124]:


neg_label_cutoff_score = np.percentile(df_cur['score'], 60)
df2 = working_df.loc[working_df['score'] <= neg_label_cutoff_score]
elements = df2[attr_consignee].values.tolist() + df2[attr_shipper].values.tolist()
white_list_entities = []
for k,v in Counter(elements).items(): 
    if v > WHITELIST_COUNT : 
        white_list_entities.append(k)


# In[125]:


print(' # white_list_entities', len(white_list_entities))


# In[208]:


label_top_k = 30


# In[232]:


def process_v1(df_cur, label_top_k):
    global attr_consignee
    global ID_COL
    global attr_shipper
    print('label_top_k ',label_top_k)
    df_cur = df_cur.sort_values(by='score',ascending=False)
    min_score = np.percentile(df_cur['score'], ANOMALY_PERCENTILE_THRESHOLD)
    seen_ids = []
    count = 0
    labelled_df = None
    idx = 0

    for i,row in df_cur.iterrows():
        _id = int(row[ID_COL])
        idx = i
        seen_ids.append(_id)

        if row['label']==1: 
            count +=1
            if count > label_top_k : 
                break

    seen_ids = set(seen_ids)
    labelled_df = df_cur.iloc[:idx,:]
    unlabelled_df = df_cur.iloc[idx:,:]
    orig_target_df = unlabelled_df.copy().reset_index(drop=True)


    tmp_N = labelled_df.loc[labelled_df['label']!=1]
    tmp_P = labelled_df.loc[labelled_df['label']==1]
    neg_entities = tmp_N[attr_consignee].values.tolist() + tmp_N[attr_shipper].values.tolist()
    pos_entities = tmp_P[attr_consignee].values.tolist() + tmp_P[attr_shipper].values.tolist()
    pos_entities = list(sorted(set(pos_entities)))
    neg_entities = list(sorted(set(neg_entities)))

    print(' - / +', len(set(neg_entities)),len(set(pos_entities)) )
    query_P = (emb_array[pos_entities]).astype(np.float32)
    D, nn_index_P = index.search(query_P, KNN)  
    nn_index_P = nn_index_P[:,1:]

    query_N = (emb_array[neg_entities]).astype(np.float32)
    D, nn_index_N = index.search(query_N, KNN)  
    nn_index_N = nn_index_N[:,1:]


    candidates = unlabelled_df.loc[unlabelled_df['score']>=min_score ]
    candidates.loc[((candidates[attr_consignee].isin(white_list)) | (candidates[attr_shipper].isin(white_list)))]           
    # first see records where both match +
    df1 = candidates.loc[candidates[attr_consignee].isin(pos_label_entities) & (candidates[attr_shipper].isin(pos_label_entities))]
    df1.loc[:,'dynamic_score'] = 2

    # first see records where both match -
    df2 = candidates.loc[candidates[attr_consignee].isin(neg_entities) & (candidates[attr_shipper].isin(neg_entities))]
    df2.loc[:,'dynamic_score'] = 0

    # first see records where one matches -
    df3 = candidates.loc[((candidates[attr_consignee].isin(white_list)) | (candidates[attr_shipper].isin(white_list)))]
    df3.loc[:,'dynamic_score'] = 0

    # At least one matches +, and the other is not part of  -
    df5_1 = candidates.loc[(candidates[attr_consignee].isin(pos_label_entities)) & ~(candidates[attr_shipper].isin(neg_label_entities))]
    df5_2 = candidates.loc[(candidates[attr_shipper].isin(pos_label_entities)) & ~(candidates[attr_consignee].isin(neg_label_entities))]
    df_5 = pd.concat([df5_1,df5_2])
    df5 = df_5.loc[~df_5[ID_COL].isin(df1)]
    df5 = df5.loc[~df5[ID_COL].isin(df3)]
    df5.loc[:,'dynamic_score'] = 0.75

    # One of the entities has been marked negative 
    df6 = candidates.loc[(candidates[attr_shipper].isin(neg_label_entities)) | (candidates[attr_consignee].isin(neg_label_entities))]
    df6 = df6.loc[~df6[ID_COL].isin(df2)]
    df6.loc[:,'dynamic_score'] = 0.25

    id_list = df1[ID_COL].values.tolist() + df2[ID_COL].values.tolist() + df3[ID_COL].values.tolist() + df5[ID_COL].values.tolist() +  df6[ID_COL].values.tolist()
    id_list = list(set(id_list))

    df4 = candidates.loc[~candidates[ID_COL].isin(id_list)]
    df4.loc[:,'dynamic_score'] = 0.5
    df0 = pd.concat([df1,df5,df4,df6,df3,df2])

    new_df = df0.sort_values( by = 'dynamic_score', ascending=False)

    for k in [50,100,150,200,250]:
        # without input?
        top_k_target = orig_target_df.head(k)
        wo_input_precison = len(top_k_target.loc[top_k_target['label']==1])/k
        tmp =  new_df.head(k)
        correct = tmp.loc[tmp['label']==1]
        prec = len(correct)/len(tmp)
        print(' precision at next top {:3d} : {:.3f}'.format(k, prec), ' || Without input {:.3f}'.format(wo_input_precison) )
    return 


# In[357]:


process_v1(working_df.copy(), label_top_k=10)
process_v1(working_df.copy(), label_top_k=20)
process_v1(working_df.copy(), label_top_k=30)
process_v1(working_df.copy(), label_top_k=40)
process_v1(working_df.copy(), label_top_k=50)


# In[ ]:





# In[368]:


def process_v2(df_cur, label_top_k):
    global attr_consignee
    global ID_COL
    global attr_shipper
    print('label_top_k ',label_top_k)
    df_cur = df_cur.sort_values(by='score',ascending=False)
    min_score = np.percentile(df_cur['score'], ANOMALY_PERCENTILE_THRESHOLD)
    seen_ids = []
    count = 0
    labelled_df = None
    idx = 0

    for i,row in df_cur.iterrows():
        _id = int(row[ID_COL])
        idx = i
        seen_ids.append(_id)

        if row['label']==1: 
            count +=1
            if count > label_top_k : 
                break

    seen_ids = set(seen_ids)
    labelled_df = df_cur.iloc[:idx,:]
    unlabelled_df = df_cur.iloc[idx:,:]
    orig_target_df = unlabelled_df.copy().reset_index(drop=True)

    tmp_N = labelled_df.loc[labelled_df['label']!=1]
    tmp_P = labelled_df.loc[labelled_df['label']==1]
    neg_entities = tmp_N[attr_consignee].values.tolist() + tmp_N[attr_shipper].values.tolist()
    pos_entities = tmp_P[attr_consignee].values.tolist() + tmp_P[attr_shipper].values.tolist()
    pos_entities = list(sorted(set(pos_entities)))
    neg_entities = list(sorted(set(neg_entities)))

    print(' - / +', len(set(neg_entities)),len(set(pos_entities)))
    query_P = (emb_array[pos_entities]).astype(np.float32)
    D, nn_index_P = index.search(query_P, KNN)  
    nn_index_P = nn_index_P[:,:]

    query_N = (emb_array[neg_entities]).astype(np.float32)
    D, nn_index_N = index.search(query_N, KNN)  
    nn_index_N = nn_index_N[:,:]

    candidates = unlabelled_df.loc[unlabelled_df['score']>=min_score ]
    candidates.loc[((candidates[attr_consignee].isin(white_list)) | (candidates[attr_shipper].isin(white_list)))]           
    # first see records where both match +
    df1 = candidates.loc[candidates[attr_consignee].isin(pos_label_entities) & (candidates[attr_shipper].isin(pos_label_entities))]
    df1.loc[:,'dynamic_score'] = 2

    # first see records where both match -
    df2 = candidates.loc[candidates[attr_consignee].isin(neg_entities) & (candidates[attr_shipper].isin(neg_entities))]
    df2.loc[:,'dynamic_score'] = 0

    # first see records where one matches -
    df3 = candidates.loc[((candidates[attr_consignee].isin(white_list)) | (candidates[attr_shipper].isin(white_list)))]
    df3.loc[:,'dynamic_score'] = 0
    
    # ------------------------------------------
    # df1 at top 
    # df2 + df_3 at end 
    # We use nearest neighbor based filters in the sizeable number of records in between
    # ------------------------------------------
    id_list = df1[ID_COL].values.tolist() + df2[ID_COL].values.tolist() + df3[ID_COL].values.tolist()
    id_list = list(set(id_list))
    df4 = candidates.loc[~candidates[ID_COL].isin(id_list)]  
    _P = nn_index_P[:,0].reshape(-1)
    nn_index_P = nn_index_P[:,1:].reshape(-1)
    nn_index_N = nn_index_N.reshape(-1)
    
    df4_0 = df4.loc[
        ((df4[attr_consignee].isin(pos_label_entities) )&(df4[attr_shipper].isin(nn_index_P)))|
        ((df4[attr_consignee].isin(nn_index_P) )&(df4[attr_shipper].isin(pos_label_entities)))
    ]
    
    df4 =  df4_0.loc[
        ~(df4_0[attr_consignee].isin(nn_index_N) )&(df4[attr_shipper].isin(nn_index_N))
    ]
    
    df4.loc[:,'dynamic_score'] = 0.75
    
    id_list = df1[ID_COL].values.tolist() + df2[ID_COL].values.tolist() + df3[ID_COL].values.tolist() + df4[ID_COL].values.tolist()
    df5 = candidates.loc[~candidates[ID_COL].isin(id_list)] 
    df0 = pd.concat([df1,df4,df5,df3,df2])
    new_df = df0.sort_values( by = 'dynamic_score', ascending=False)

    for k in [50,100,150,200,250]:
        # without input?
        top_k_target = orig_target_df.head(k)
        wo_input_precison = len(top_k_target.loc[top_k_target['label']==1])/k
        tmp =  new_df.head(k)
        correct = tmp.loc[tmp['label']==1]
        prec = len(correct)/len(tmp)
        print(' precision at next top {:3d} : {:.3f}'.format(k, prec), ' || Without input {:.3f}'.format(wo_input_precison) )
    
    return 


# ## process_v1 : Whitelist Blacklist  (multiple filters )
# ## process_v2 : Whitelist Blascklist  + embedding based nearest neighbor  ( LSH used)

# In[369]:


process_v1(df_cur, label_top_k = 10)
process_v2(df_cur, label_top_k = 10)


# In[370]:


process_v1(df_cur, label_top_k = 20)
process_v2(df_cur, label_top_k = 20)


# In[371]:


process_v1(df_cur, label_top_k = 30)
process_v2(df_cur, label_top_k = 30)


# In[372]:


process_v1(df_cur, label_top_k = 40)
process_v2(df_cur, label_top_k = 40)


# In[373]:


process_v1(df_cur, label_top_k = 50)
process_v2(df_cur, label_top_k = 50)


# In[ ]:



















