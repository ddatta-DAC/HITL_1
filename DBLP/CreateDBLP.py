#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
from collections import Counter
import networkx as nx
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
print(stopwords.words('english'))


import pickle
from scipy import sparse




with open('DBLP_Triplets.pickle','rb') as fh :
    data_obj = pickle.load(fh)

print(len(data_obj[0]))
author_label = pd.read_csv('author_label.txt',low_memory=False,header=None,sep='\t',names=['ID','label','name'])

author_file = 'author.txt' 
author_df = pd.read_csv(
    author_file,
    low_memory=False,
    header=None,
    sep='\t',
    names=['ID','name']
)

paper_file = 'paper.txt' 
paper_df = pd.read_csv(
    paper_file,
    low_memory=False,
    header=None,
    sep='\t',
    names=['ID','name']
)

term_file = 'term.txt' 
term_df = pd.read_csv(
    term_file,
    low_memory=False,
    header=None,
    sep='\t',
    names=['ID','name']
)


# In[15]:


conf_file = 'conf.txt' 
conf_df = pd.read_csv(
    conf_file,
    low_memory=False,
    header=None,
    sep='\t',
    names=['ID','name']
)


# In[16]:


paper_ID_list = paper_df['ID'].values
author_ID_list = author_df['ID'].values
conf_ID_list = conf_df['ID'].values
term_ID_list = term_df['ID'].values


# In[ ]:





# In[ ]:





# In[17]:


# --------------
# Make sure all type of nodes have IDs starting from 0
# --------------
class node:
    def __init__(_id,category):
        self.id = _id
        self.category = category
        


# In[18]:


# Read in edges

PA = pd.read_csv('paper_author.txt', sep='\t', names = ['P','A'])
PC = pd.read_csv('paper_conf.txt', sep='\t', names = ['P','C'])
PT = pd.read_csv('paper_term.txt', sep='\t', names = ['P','T'])


# In[19]:


nodes_dict = {
    'P': {},
    'A': {},
    'C': {},
    'T': {}
}


# In[20]:


def search(df, val):
    return df.loc[df['ID']==val].values[0]


# In[ ]:





# In[21]:


t_df = pd.read_csv(
    term_file,
    low_memory=False,
    header=None,
    sep='\t',
    names=['ID','name']
)

rmv_terms = t_df.loc[t_df['name'].isin(set(stopwords.words('english')))]['ID'].values.tolist()


# In[22]:


# --------------------------
# Create graph 
# --------------------------
G = nx.Graph()
for i,row in PA.iterrows():
    p = row['P']
    a = row['A']
    if a in list(author_label['ID']):
        G.add_edge(p,a)
    
for i,row in PC.iterrows():
    p = row['P']
    c = row['C']
    G.add_edge(p,c)  
    
for i,row in PT.iterrows():
    p = row['P']
    t = row['T']
    if t not in rmv_terms:
        G.add_edge(p,t)


# In[23]:


print(Counter(author_label['label']))


# In[24]:


label_0_ids =author_label.loc[author_label['label']==0]['ID'].values.tolist()
label_1_ids =author_label.loc[author_label['label']==1]['ID'].values.tolist()
label_2_ids =author_label.loc[author_label['label']==2]['ID'].values.tolist()
label_3_ids =author_label.loc[author_label['label']==3]['ID'].values.tolist()


# In[26]:


G1 = G.copy()
print(G1.number_of_nodes(),G1.number_of_edges())

for conn_comp in nx.connected_components(G1):
    print(len(conn_comp))
G1.number_of_edges()
filter_terms = []
for node in G1.nodes():
    if node in list(term_df['ID']):
        if nx.degree(G1,node) < 3 : 
            filter_terms.append(node)
print(' Rare Terms to remove ',len(filter_terms))

for n in filter_terms:
    G1.remove_node(n)
isolates = [n for n in nx.isolates(G1) ]
for n in isolates:
    G1.remove_node(n)

print(G1.number_of_nodes(), G1.number_of_edges())


# In[27]:


# =================================
# Create synthetic ids for continuous ids in the riginal dataset
# ==================================
def get_df(id_list, valid_ids, label_df=None):
    id_list = [i for i in id_list if i in valid_ids]
    print(' >> ', len(id_list))
    data = [( i,j) for i,j in enumerate(id_list)]
    df = pd.DataFrame(np.array(data),columns=['synID','ID'])
    print(df.head(10))
    if label_df is not None:
         df = df.merge(label_df, on=['ID'],how='inner')
    return df

valid_ids = list(G1.nodes())


# In[28]:


C_df = get_df(conf_ID_list,valid_ids)
T_df = get_df(term_ID_list,valid_ids)
P_df = get_df(paper_ID_list,valid_ids)


# In[29]:


author_label_df = author_label[['ID','label']]
A_df = get_df(author_ID_list,valid_ids,author_label_df)


# In[30]:


C_df['type'] = 'C'
P_df['type'] = 'P'
T_df['type'] = 'T'

C_df['label'] = None
P_df['label'] = None
T_df['label'] = None


# In[31]:


A_df['type'] = 'A'


# In[32]:


len(A_df)


# In[33]:


# Export the edge lists


# In[34]:


G1.number_of_edges()


# In[43]:


all_nodes_df = A_df.copy()
all_nodes_df = all_nodes_df.append(C_df,ignore_index=True)
all_nodes_df = all_nodes_df.append(P_df,ignore_index=True)
all_nodes_df = all_nodes_df.append(T_df,ignore_index=True)


# In[44]:


all_nodes_df['ID']


# In[63]:


edges_df = pd.DataFrame(columns=['n1','n2','edge_type'])


# In[ ]:





# In[64]:


from joblib import Parallel, delayed

def setup_edges (e):
    row1 = all_nodes_df.loc[all_nodes_df['ID']==e[0]]
    row2 = all_nodes_df.loc[all_nodes_df['ID']==e[1]]
    
    id1 = row1['synID'].values.tolist()[0]
    id2 = row2['synID'].values.tolist()[0]
    type1 = row1['type'].values.tolist()[0]
    type2 = row2['type'].values.tolist()[0]
    
    # Lexicographic ordering
    if type1 > type2:
        type1, type2 = type2, type1
        row1, row2 = row2,row1
        id1, id2 = id2, id1

    
    edge_type = type1 + '_' + type2
    _dict = {
            'n1':id1,
            'n2':id2,
            'edge_type' : edge_type
        }
    return _dict
     
 


# In[65]:


results = Parallel(n_jobs=16)(delayed(setup_edges)(e) for  e in G1.edges())


# In[67]:


# for r in results:
#     edges_df = edges_df.append(r,ignore_index=True)

edges_df = pd.DataFrame(results)


# In[ ]:





# In[68]:


edges_df.to_csv('dblp_edges.csv',index=False)


# In[69]:


all_nodes_df.to_csv('dblp_nodes.csv',index=False)


# In[ ]:




