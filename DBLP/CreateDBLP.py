#!/usr/bin/env python
# coding: utf-8

# In[37]:


import numpy as np
import pandas as pd
from collections import Counter
import networkx as nx
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
print(stopwords.words('english'))
from joblib import Parallel, delayed

import pickle
from scipy import sparse

with open('DBLP_Triplets.pickle','rb') as fh :
    data_obj = pickle.load(fh)

author_label = pd.read_csv('author_label.txt',low_memory=False,header=None,sep='\t',names=['ID','label','name'])
author_label['ID'] =  author_label['ID'].apply(lambda x :'A' +  str(x))



Counter(author_label['label'])


valid_authors = list(author_label['ID'])
print('# of authors with labels ', len(valid_authors))

author_file = 'author.txt' 
author_df = pd.read_csv(
    author_file,
    low_memory=False,
    header=None,
    sep='\t',
    names=['ID','name']
)
author_df['ID'] =  author_df['ID'].apply(lambda x :'A' +  str(x))
author_df =  author_df.loc[author_df['ID'].isin(author_label['ID'].values.tolist())]


paper_file = 'paper.txt' 
paper_df = pd.read_csv(
    paper_file,
    low_memory=False,
    header=None,
    sep='\t',
    names=['ID','name']
)
paper_df['ID'] =  paper_df['ID'].apply(lambda x :'P' +  str(x))

term_file = 'term.txt' 
term_df = pd.read_csv(
    term_file,
    low_memory=False,
    header=None,
    sep='\t',
    names=['ID','name']
)

term_df = term_df.loc[~term_df['name'].isin(set(stopwords.words('english')))]
term_df['ID'] =  term_df['ID'].apply(lambda x :'T' +  str(x))

conf_file = 'conf.txt' 
conf_df = pd.read_csv(
    conf_file,
    low_memory=False,
    header=None,
    sep='\t',
    names=['ID','name']
)
conf_df['ID'] =  conf_df['ID'].apply(lambda x :'C' +  str(x))



paper_ID_list = paper_df['ID'].values
author_ID_list = author_df['ID'].values
conf_ID_list = conf_df['ID'].values
term_ID_list = term_df['ID'].values

print(len(paper_ID_list), len(author_ID_list), len(conf_ID_list), len(term_ID_list))

PA = pd.read_csv('paper_author.txt', sep='\t', names = ['P','A'])
PC = pd.read_csv('paper_conf.txt', sep='\t', names = ['P','C'])
PT = pd.read_csv('paper_term.txt', sep='\t', names = ['P','T'])
PA['P']  =  PA['P'].apply(lambda x :'P' +  str(x))
PT['P']  =  PA['P'].apply(lambda x :'P' +  str(x))
PC['P']  =  PA['P'].apply(lambda x :'P' +  str(x))
PA['A']  =  PA['A'].apply(lambda x :'A' +  str(x))
PT['T']  =  PT['T'].apply(lambda x :'T' +  str(x))
PC['C']  =  PC['C'].apply(lambda x :'C' +  str(x))




# --------------------------
# Create graph 
# --------------------------
G = nx.Graph()

for i,row in PA.iterrows():
    p = row['P']
    a = row['A']
    if a in valid_authors:
        G.add_edge(p,a)

print(G.number_of_nodes(), G.number_of_edges())

for i,row in PC.iterrows():
    p = row['P']
    c = row['C']
    G.add_edge(p,c)  

print(G.number_of_nodes(), G.number_of_edges())



PT = PT.loc[PT['T'].isin(term_df['ID'].values.tolist())]
_p = PT['P'].values.tolist()
_t = PT['T'].values.tolist()
arr = [ (i,j) for i,j in zip(_p,_t)]
G.add_edges_from(arr)

print(G.number_of_nodes(), G.number_of_edges())

print('Author / label count', Counter(author_label['label']))

label_0_ids =author_label.loc[author_label['label']==0]['ID'].values.tolist()
label_1_ids =author_label.loc[author_label['label']==1]['ID'].values.tolist()
label_2_ids =author_label.loc[author_label['label']==2]['ID'].values.tolist()
label_3_ids =author_label.loc[author_label['label']==3]['ID'].values.tolist()

G1 = G.copy()
print(G1.number_of_nodes(),G1.number_of_edges())

filter_terms = []
for node in G1.nodes():
    if node in list(term_df['ID']):
        if nx.degree(G1,node) < 2 : 
            filter_terms.append(node)
print(' Rare Terms to remove ',len(filter_terms))

for n in filter_terms:
    G1.remove_node(n)
    
isolates = [n for n in nx.isolates(G1) ]
for n in isolates:
    G1.remove_node(n)

print(G1.number_of_nodes(), G1.number_of_edges())

# Figure out the valid term ids
list_terms = set(list(G1.nodes())).intersection(set(term_ID_list))
list_authors = set(list(G1.nodes())).intersection(set(valid_authors))
list_confs = set(list(G1.nodes())).intersection(set(conf_ID_list))
list_papers = set(list(G1.nodes())).intersection(set(paper_ID_list))

print( [len(list_terms),len(list_authors),len(list_confs),len(list_papers)])

# =================================
# Create synthetic ids for continuous ids in the riginal dataset
# ==================================
def get_df(id_list, label_df=None):
    data = [( i,j) for i,j in enumerate(id_list)]
    df = pd.DataFrame(np.array(data),columns=['synID','ID'])
    print(len(df))
    if label_df is not None:
         df = df.merge(label_df, on=['ID'],how='inner')
    print(len(df))
    return df


valid_ids = list(G1.nodes())
C_df = get_df(list_confs,None)
T_df = get_df(list_terms)
P_df = get_df(list_papers)
author_label_df = author_label[['ID','label']]
A_df = get_df(list_authors, author_label_df)

C_df['type'] = 'C'
P_df['type'] = 'P'
T_df['type'] = 'T'
A_df['type'] = 'A'

C_df['label'] = None
P_df['label'] = None
T_df['label'] = None
# Export the edge lists
G1.number_of_edges()

all_nodes_df = A_df.copy()
all_nodes_df = all_nodes_df.append(C_df,ignore_index=True)
all_nodes_df = all_nodes_df.append(P_df,ignore_index=True)
all_nodes_df = all_nodes_df.append(T_df,ignore_index=True)

edges_df = pd.DataFrame(data_obj[2],columns=['n1','n2','etype'])
all_ids = list(all_nodes_df['ID'])
edges_df = edges_df.loc[(edges_df['n1'].isin(all_ids)) & (edges_df['n2'].isin(all_ids))]

edges_df.to_csv('dblp_edges.csv',index=False)
all_nodes_df.to_csv('dblp_nodes.csv',index=False)

