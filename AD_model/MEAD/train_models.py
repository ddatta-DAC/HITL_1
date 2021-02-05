import os
import torch
import argparse
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
from pathlib import Path
import sys

sys.path.append('./..')
sys.path.append('./../..')
sys.path.append('./../../..')
DATA_LOC = './../../generated_data_v1'
DIR = None
# ===================================== #

def get_domain_dims():
    global DATA_LOC
    global DIR
    with open( os.path.join(DATA_LOC,'{}/domain_dims.pkl'.format(DIR)), 'rb')  as fh:
        domain_dims = pickle.load(fh)
    return domain_dims


def get_training_data():
    global DATA_LOC
    global DIR
    x_pos = np.load(os.path.join(DATA_LOC,'{}/stage_2/train_x_pos.npy'.format(DIR)))
    x_neg = np.load(os.path.join(DATA_LOC,'{}/stage_2/train_x_neg.npy'.format(DIR)))
    return x_pos, x_neg


# ===================================== #

def main(
        DIR,
        lr=None,
        batch_size=None,
        emb_dim=None,
        epochs=None
):

    saved_model_dir = 'saved_model'
    path_obj = Path(saved_model_dir)
    path_obj.mkdir( parents=True, exist_ok=True)

    x_pos, x_neg = get_training_data()
    x_neg = x_neg.reshape([x_pos.shape[0], -1, x_pos.shape[1]])
    domain_dims = get_domain_dims()
    total_entity_count = sum(domain_dims.values())
    model = AD.AD_model_container(total_entity_count, emb_dim=emb_dim, device=device, lr= lr)
    model.train_model(x_pos, x_neg, batch_size=batch_size, epochs=epochs)
    model.save_model(os.path.join(saved_model_dir,'/{}'.format(DIR)))
    return


# ===================================== #

parser = argparse.ArgumentParser()
parser.add_argument(
    '--DIR', choices=['us_import1', 'us_import2', 'us_import3', 'us_import4', 'us_import5'],
    default='us_import1'
)

parser.add_argument(
    '--saved_model',
    default=None
)

parser.add_argument(
    '--lr',
    default=0.001,
    type=float
)

parser.add_argument(
    '--batch_size',
    default=256,
    type=int
)

parser.add_argument(
    '--emb_dim',
    default=16,
    type=int
)

parser.add_argument(
    '--epochs',
    default=100,
    type=int
)

args = parser.parse_args()
DIR = args.DIR
saved_model = args.saved_model

lr = args.lr
saved_model = args.saved_model
batch_size = args.batch_size
emb_dim = args.emb_dim
epochs = args.epochs

main(
    DIR=DIR,
    lr=lr,
    batch_size=batch_size,
    emb_dim=emb_dim,
    epochs=epochs
)
