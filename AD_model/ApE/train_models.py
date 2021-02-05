import torch
import model
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from pathlib import Path
import os
import argparse
from matplotlib import pyplot as plt
import pandas as pd
import yaml
import sys
import pickle
import model as APE

sys.path.append('./../..')
sys.path.append('./..')
sys.path.append('./../../..')

ID_COL = 'PanjivaRecordID'
RESULTS_OP_PATH = 'APE_output'
DATALOC = './../../generated_data_v1'


def get_domain_dims(DIR):
    global DATALOC
    with open(os.path.join(DATALOC, '{}/domain_dims.pkl'.format(DIR)), 'rb')  as fh:
        domain_dims = pickle.load(fh)
    return domain_dims


def get_training_data(DIR):
    x_pos = np.load(os.path.join(DATALOC, '{}/stage_2/train_x_pos.npy'.format(DIR)))
    x_neg = np.load(os.path.join(DATALOC, '{}/stage_2/train_x_neg.npy'.format(DIR)))
    return x_pos, x_neg


def main(
        DIR=None,
        lr=None,
        batch_size=None,
        emb_dim=None,
        epochs=None
):
    global ID_COL
    global RESULTS_OP_PATH

    saved_model_dir = 'saved_model'
    path_obj = Path(saved_model_dir)
    path_obj.mkdir( parents=True, exist_ok=True)

    RESULTS_OP_PATH = os.path.join(RESULTS_OP_PATH, DIR)
    results_path = Path(RESULTS_OP_PATH)
    results_path.mkdir(parents=True, exist_ok=True)

    x_pos, x_neg = get_training_data(DIR)
    x_neg = x_neg.reshape([x_pos.shape[0], -1, x_pos.shape[1]])
    domain_dims = get_domain_dims(DIR)

    container = APE.model.APE_container(emb_dim, domain_dims, device, batch_size=batch_size, LR=lr)

    loss = container.train_model(
        x_pos,
        x_neg,
        num_epochs=epochs,
        tol=0.1
    )
    container.save_model('saved_model/{}'.format(DIR))
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
    default=0.001
)

parser.add_argument(
    '--batch_size',
    default=256
)

parser.add_argument(
    '--emb_dim',
    default=16
)


parser.add_argument(
    '--epochs',
    default=100
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
