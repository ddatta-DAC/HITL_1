import torch
import numpy as np
from numpy.core._multiarray_umath import ndarray
import os
import sys
import warnings
warnings.filterwarnings("ignore")
sys.path.append('./..')

class GD:

    def __init__(
            self,
            num_coeff,
            emb_dim,
            interaction_type = 'concat'
    ):
        self.num_coeff = num_coeff
        self.coeff_mask: ndarray = np.zeros(num_coeff)

        self.W_cur = None
        self.emb_dim = emb_dim
        self.W_orig = None
        self.interaction_type = interaction_type
        return

    def set_original_W(self, W):
        self.W_orig = W
        self.W_cur = W
        return

    def update_weight(
            self,
            labels = [],
            list_feature_mod_idx = [],
            X=None,
            tol = 0.02,
            lr = 0.1,
            max_iter =1000
    ):
        X = np.array(X)
        print(X.shape)
        W = np.copy(self.W_cur)
        num_coeff = self.num_coeff
        target_y = [ ]


        for _label in labels:
            if _label == 1:
                target_y.append(2.0)
            else:
                target_y.append(0)

        target_y = np.array(target_y)
        # W has shape [num_coeff, emb_dim * 2]

        feature_corr_idx = []
        for idx in range(X.shape[0]):
            _exp_features_ = list_feature_mod_idx[idx]
            if len(list_feature_mod_idx) == 0:
                _feature_idx = np.ones(num_coeff)/num_coeff
            else:
                _feature_idx = np.zeros(num_coeff)
                _feature_idx[_exp_features_] = 1
                _feature_idx = _feature_idx/len(list_feature_mod_idx)
            feature_corr_idx.append(_feature_idx)
        feature_corr_idx = np.array(feature_corr_idx)
        print('>>> _feature_idx', feature_corr_idx.shape)
        iter = 0
        while iter <= max_iter:
            iter +=1
            pred_y = []
            for idx in range(X.shape[0]):
                _pred_y = np.sum([ np.matmul(X[idx][j], W[j]) for j in range(num_coeff)])
                pred_y.append(_pred_y)

            pred_y = np.array(pred_y)
            err = target_y - pred_y
            err = err.reshape([err.shape[0],1])
            if np.mean(err) <= tol :
                break

            # apply correction
            adjusted_error = err * feature_corr_idx
            for coef_idx in range(num_coeff):
                # _grad has shape [batch, emb*x ]
                assert adjusted_error[:, coef_idx].shape[0] == X.shape[0]
                _grad = adjusted_error[:, coef_idx].reshape([-1,1]) * X[:, coef_idx]
                assert _grad.shape[1] == X[:, coef_idx].shape[1]
                # sum gradient over batch
                batch_grad = np.mean(_grad,axis=1)
                W[coef_idx] = W[coef_idx] + lr * batch_grad

        return W

