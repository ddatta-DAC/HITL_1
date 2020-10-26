import os
import torch
import pandas as pd
import numpy as np
try:
    from . import model_AD_1 as AD
except:
    import model_AD_1 as AD
import pickle
import matplotlib.pyplot as plt
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_domain_dims():
    with open('./../generated_data_v1/us_import1/domain_dims.pkl', 'rb')  as fh:
        domain_dims = pickle.load(fh)
    return domain_dims

def get_training_data():
    x_pos = np.load('./../generated_data_v1/us_import1/stage_2/train_x_pos.npy')
    x_neg = np.load('./../generated_data_v1/us_import1/stage_2/train_x_neg.npy')
    return x_pos, x_neg

x_pos, x_neg = get_training_data()
x_neg = x_neg.reshape([x_pos.shape[0], -1 , x_pos.shape[1]])
domain_dims = get_domain_dims()
total_entity_count = sum(domain_dims.values())
model = AD.AD_model_container(total_entity_count, emb_dim=16, device=device)
model.train_model(x_pos,x_neg, batch_size=128, epochs=25)
model.save_model('saved_model/us_import1')
model.model.mode='test'
test_df = pd.read_csv( './../generated_data_v1/us_import1/stage_2/test_normal_serialized.csv', index_col=None )
ID_COL = 'PanjivaRecordID'
id_list_normal = test_df[ID_COL].values.tolist()
del test_df[ID_COL]
test_x = test_df.values
scores_1 = model.score_samples(test_x)

test_df_p = pd.read_csv( './../generated_data_v1/us_import1/stage_2/test_pos_serialized.csv', index_col=None )
id_list_p = test_df_p[ID_COL].values.tolist()
del test_df_p[ID_COL]
test_xp = test_df_p.values
scores_2 =  model.score_samples(test_xp)

test_df_n = pd.read_csv( './../generated_data_v1/us_import1/stage_2/test_neg_serialized.csv', index_col=None )
id_list_n = test_df_n[ID_COL].values.tolist()
del test_df_n[ID_COL]
test_xn = test_df_n.values
scores_3 = model.score_samples(test_xn)

try:
    data = [np.random.normal(0, std, 1000) for std in range(1, 6)]
    box = plt.boxplot([scores_1,scores_2,scores_3], notch=True, patch_artist=True)
    colors = ['cyan', 'pink', 'lightgreen']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    plt.show()
except:
    pass




