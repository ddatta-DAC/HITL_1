import torch
import numpy as np
from numpy.core._multiarray_umath import ndarray
import os
import sys
import warnings
warnings.filterwarnings("ignore")
sys.path.append('./..')
from tqdm import tqdm 
from joblib import Parallel,delayed
from tqdm import trange
from joblib import parallel_backend

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
            X = None,
            tol = 0.025,
            lr = 0.25,
            max_iter = 1000,
            min_lr = 0.05 
    ):
        X = np.array(X)
        W = np.copy(self.W_cur)
        
        num_coeff = self.num_coeff
        target_y = [ ]
        

        for _label in labels:
            if _label == 1.5:
                target_y.append(1.0)
            else:
                target_y.append(0.0)

        target_y = np.array(target_y)
        # W has shape [num_coeff, emb_dim * 2]

        feature_corr_idx = []
        for idx in range(X.shape[0]):
            _exp_features_ = list_feature_mod_idx[idx]
            
            if len(list_feature_mod_idx) == 0:
                _feature_idx = np.ones(num_coeff)/num_coeff * 2
            else:
                _feature_idx = np.zeros(num_coeff)
                for ef in _exp_features_:
                    _feature_idx[ef] = 5
                _feature_idx = _feature_idx/len(list_feature_mod_idx)
            feature_corr_idx.append(_feature_idx)
        feature_corr_idx = np.array(feature_corr_idx)
        
#         t = trange(max_iter, desc='Training error: ', leave=True)
        orig_error = 0
        
        for i in range(max_iter):
            pred_y = []
#             tqdm._instances.clear()
            
#             for idx in range(X.shape[0]):
#                 _pred_y = np.sum([ np.matmul(X[idx][j], W[j]) for j in range(num_coeff)])
#                 pred_y.append(_pred_y)
            
            a = (X * W)
            b = np.sum(a,axis=-1)
            c = np.sum(b,axis=1)
            
            pred_y = np.array(c)
            err = (target_y - pred_y)
            err = err.reshape([err.shape[0],1])
            abs_error = np.abs(np.mean(np.power(err,2)))
            
            if i == 0:
                orig_error = abs_error
#             t.refresh()
#             t.set_description("Error : {:.4f}".format(np.abs(np.mean(err))))
#             t.refresh()
            
            if abs_error <= tol :
                break

            # apply correction
            adjusted_error = err * feature_corr_idx
            
            # Parallelize 
            def aux(coef_idx):
                nonlocal adjusted_error
                nonlocal X
                
                _grad = adjusted_error[:, coef_idx].reshape([-1,1]) * X[:, coef_idx]
                
                batch_grad = np.mean(_grad, axis=0)
                batch_grad = np.clip(batch_grad, -0.1, 0.1)
                return (coef_idx, batch_grad)
            
#             with parallel_backend('threading', n_jobs = num_coeff):
            res = Parallel(n_jobs = num_coeff)( 
                delayed(aux)(coef_idx,) for coef_idx in range(num_coeff)
            )
            for r in res :
                W[r[0]] = W[r[0]] + lr * r[1]
                
            lr = max(min_lr, lr * 0.99)

        return W

