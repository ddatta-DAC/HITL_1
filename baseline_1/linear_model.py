import pandas as pd
import os
import sys
import numpy as np
from torch import nn
import torch
from sklearn.base import TransformerMixin
from tqdm import tqdm
from torch import FloatTensor as FT
from torch.nn import functional as F


class linearClassifier(
    nn.Module,
    TransformerMixin
):
    def __init__(
            self, num_domains, emb_dim, LR=0.001, num_epochs=250, batch_size=32
    ):
        super(linearClassifier, self).__init__()
        self.emb_dim = emb_dim
        self.K = int(num_domains * (num_domains - 1) / 2)
        self.num_domains = num_domains
        self.W = nn.parameter.Parameter(
            data=torch.normal(mean=0, std=1, size=[self.K, emb_dim])
        )
        self.opt = torch.optim.Adam([self.W], lr=LR)
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        return

    # -------------
    # Main function to be called when training the model
    # -------------
    def fit(self, X, y, log_interval=100):
        self._train(X, y, log_interval = log_interval)
        return

    def predict(self, X):
        self.eval()
        return self.score_sample(X)

    def score_sample(self, X):
        return self.forward(FT(X))

    # ---------------------------
    # Externally set the weights
    # ----------------------------
    def update_W(self, new_W):
        self.W.data = torch.from_numpy(new_W).float()
        return

    # -------------
    # X has shape [ N, nd, emb_dm ]
    # y has shape [N]
    # -------------
    def forward(self, x):
        _x_ = torch.chunk(x, self.num_domains, dim=1)
        _x_ = [_.squeeze(1) for _ in _x_]
        terms = []
        k = 0
        for i in range(self.num_domains):
            for j in range(i + 1, self.num_domains):
                x1x2 = _x_[i] * _x_[j]
                _ij = torch.matmul(x1x2, self.W[k])
                terms.append(_ij)
                k += 1
        wx = torch.stack(terms, dim=-1)

        sum_wx = torch.sum(wx, dim=-1)
        return sum_wx

    def train_iter(self, x, y, reg=False):
        self.opt.zero_grad()
        sum_wx = self.forward(FT(x))
        # Regression style MSE loss function
        y = FT(y)
        loss = F.smooth_l1_loss(
            sum_wx,
            y,
            reduction='none'
        )

        if reg:
            l2_reg = torch.norm(self.W.data)
            loss += 0.000001 * l2_reg
        loss = torch.mean(loss)
        loss.backward()
        self.opt.step()
        return loss

    # ============================
    # Assume that the labels are +1, -1
    # ============================
    def _train(self, X, y, log_interval=100):
        self.train()
        # there are 2 labels
        # +1 and -1
        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == -1)[0]

        bs = self.batch_size
        for e in tqdm(range(self.num_epochs)):
            p_idx = np.random.choice(pos_idx, size=bs // 2)
            n_idx = np.random.choice(neg_idx, size=bs // 2)
            _x_p = X[p_idx]
            _x_n = X[n_idx]
            _x = np.vstack([_x_p, _x_n])
            _y = np.hstack([
                np.ones([bs // 2]),
                np.ones([bs // 2]) * -1
            ])
            _loss = self.train_iter(_x, _y, reg=True)
            _loss = _loss.cpu().data.numpy().mean()
            if e % log_interval == 0:
                print('Step {} Loss {:.4f}'.format(e + 1, _loss))
        return

    # ==============================
    # Train the model on positive samples only
    # ==============================
    def fit_on_pos(self, X, y, n_epochs=None, log_interval=250):
        pos_idx = np.where(y == 1)[0]
        bs = self.batch_size
        if n_epochs is None:
            n_epochs = self.num_epochs // 10
        for e in tqdm(range(n_epochs)):
            p_idx = np.random.choice(pos_idx, size=bs)
            _x = X[p_idx]
            _y = np.ones([bs])
            _loss = self.train_iter(_x, _y)
            _loss = _loss.cpu().data.numpy().mean()
            if e % log_interval == 0:
                print('Step {} Loss {:.4f}'.format(e + 1, _loss))
        return

    def predict_score_op(self, X):
        self.eval()
        res_y = self.forward(FT(X))
        return res_y.cpu().data.numpy()

# ----------------------------------------------------------
# X1 = np.random.normal(
#     loc=-2, scale=1.5, size=[1000, 5, 16]
# )
# X2 = np.random.normal(
#     loc=2, scale=1, size=[1000, 5, 16]
# )
# X = np.vstack([X1, X2])
# y = np.hstack([np.ones(1000), np.ones(1000) * -1])

# print(y.shape)
# obj = linearClassifier(num_domains=5, emb_dim=16, num_epochs=10000, batch_size=256)
# obj.fit(X, y)
# obj.fit_on_pos(X, y, 100)

# print(obj.score_sample(X[:10]))
# print(y[:10])

# print(obj.score_sample(X[-10:]))
# print(y[-10:])
