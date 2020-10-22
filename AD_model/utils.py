import numpy as np
import pandas as pd
import os
import sys
from pandarallel import pandarallel
from joblib import Parallel,delayed
pandarallel.initialize()
from tqdm import tqdm
import pickle
import multiprocessing as mp

# ======================================== #

def generate_negative_samples(
        df,
        DIR,
        data_source_loc=None,
        id_col = 'PanjivaRecordID',
        num_neg_samples = 10
):
    if data_source_loc is None:
        data_source_loc = './../generated_data_v1/'

    loc = os.path.join(data_source_loc, DIR)
    with open(os.path.join(loc, 'domain_dims.pkl'), 'rb') as fh:
        domain_dims = pickle.load(fh)

    idMapper_file = os.path.join(loc, 'idMapping.csv')
    idMapping_df = pd.read_csv(idMapper_file, index_col=None)
    domain_values = {}
    for d in domain_dims.keys():
        domain_values[d] = idMapping_df.loc[idMapping_df['domain']==d]['serial_id'].vlaues.tolist()
    def gen_sample_aux(row, domain_values, num_neg_samples):
        domains = list(domain_values.keys())
        num_domains = len(domains)
        res = []
        for _ in range(num_neg_samples):
            new_row = row.copy()
            pert_count = np.random.randint(0,num_domains//2)
            sel_domains = np.random.choice( domains, pert_count, replace=False )
            for sd in sel_domains:
                new_row[sd] = np.random.choice(domain_values[sd],1)

            del new_row[id_col]
            vals = new_row.values.tolist()
            res.append(vals)

        x_n = np.array(res)
        row = row.copy()
        del row[id_col]
        x = row.values
        return (x, x_n)

    results = Parallel(
        n_jobs=mp.cpu_count()
    )(delayed(gen_sample_aux)(
        row, domain_dims, num_neg_samples
    ) for i,row in tqdm(df.iterrows(), total=df.shape[0])
      )
    x_p = []
    x_n = []
    for r in results:
        x_p.append(r[0])
        x_n.append(r[1])

    x_n = np.vstack(x_n)
    print(x_n.shape)
    x_p = np.array(x_p)
    print(x_p.shape)
    return x_p,x_n

def convert_to_serializedID_format(
        target_df,
        DIR,
        data_source_loc=None,
        REFRESH=False
):
    if data_source_loc is None:
        data_source_loc = './../generated_data_v1/'
    loc = os.path.join(data_source_loc, DIR)
    with open(os.path.join(loc, 'domain_dims.pkl'), 'rb') as fh:
        domain_dims = pickle.load(fh)

    idMapper_file = os.path.join(loc, 'idMapping.csv')
    if os.path.exists(idMapper_file) and REFRESH is False:
        idMapping_df = pd.read_csv(idMapper_file, index_col=None)
    else:
        cur = 0
        col_serial_id = []
        col_entity_id = []
        col_domain_names = []
        # ------------------
        for d in sorted(domain_dims.keys()):
            s = domain_dims[d]
            col_entity_id.extend(list(range(s)))
            col_domain_names.extend([d for _ in range(s)])
            tmp = np.arange(s) + cur
            tmp = tmp.tolist()
            col_serial_id.extend(tmp)
            cur += s

        data = {'domain': col_domain_names, 'entity_id': col_entity_id, 'serial_id': col_serial_id}
        idMapping_df = pd.DataFrame(data)
        # Save the idMapper
        idMapping_df.to_csv(idMapper_file, index=False)

    # Convert
    def convert_aux(val, domain):
        return \
        idMapping_df.loc[(idMapping_df['domain'] == domain) & (idMapping_df['entity_id'] == val)]['serial_id'].values[0]

    for domain in tqdm(list(domain_dims.keys())):
        target_df[domain] = target_df[domain].parallel_apply(convert_aux, args=(domain,))
    return target_df






