import torch
import numpy as np
from numpy.core._multiarray_umath import ndarray
from torch import LongTensor as LT
import os
import sys
import tangent
import warnings
from scipy.linalg import qr

warnings.filterwarnings("ignore")
sys.path.append('./..')


# ========================================
# the columns are the older basis vectors for the qr function ; thus transpose
# ========================================
def gramSchmidt(V):
    _basis, _ = qr(V.transpose(), mode='economic')
    return _basis.transpose()


# -----------------------------------------------------
# Reduce the avg cosine loss between W and (x1*x2)
# Such that W. (x1x2) is maximized
# -----------------------------------------------------
def cosine_loss(X, Y):
    xnorm = np.sqrt(np.sum(X * X))
    ynorm = np.sqrt(np.sum(Y * Y))
    similarity = np.sum(X * Y) / (xnorm * ynorm)
    return 1 - similarity


# =====================================================================
# Combine Projected GD and Confidence weighted GD 
# =====================================================================
class onlineGD:
    def __init__(self, num_coeff, emb_dim):
        self.num_coeff = num_coeff
        self.coeff_mask: ndarray = np.zeros(num_coeff)
        self.prior_grad_vectors = {
            k: [] for k in range(num_coeff)
        }
        self.W_cur = None
        self.emb_dim = emb_dim
        self.gradient_fn = tangent.grad(cosine_loss, verbose=False)
        self.W_orig = None
        return

    def set_original_W(self, W):
        self.W_orig = W
        self.W_cur = W

    # ------------------------------------
    # list_feature_mod_idx: A list of list of indices for each sample.
    # empty list for a sample means no explanation
    # signs :
    # ------------------------------------
    def update_weight(
            self,
            label=[],
            list_feature_mod_idx=[],
            X=None
    ):
        update_mask = []
        W = self.W_cur
        num_coeff = self.num_coeff
        emb_dim = self.emb_dim
        for _label, _feat_idx in zip(label, list_feature_mod_idx):

            _mask = self.coeff_mask.copy()
            # Update on the positive labels only
            if _label == 1:
                for _f in _feat_idx:
                    _mask[_f] = 1
            update_mask.append(_mask)
        update_mask = np.array(update_mask)
        num_samples = update_mask.shape[0]

        # Output mask shape : num_samples, num_coeff, coeff_dim
        output_mask = np.broadcast_to(update_mask.reshape([update_mask.shape[0], update_mask.shape[1], 1]),
                                      [update_mask.shape[0], update_mask.shape[1], emb_dim])
        # tiled_W shape: [ Num_samples, num_coeff, coeff_dim ]
        tiled_W = np.tile(W.reshape([1, W.shape[0], W.shape[1]]), (num_samples, 1, 1))

        gradient_values = np.zeros(tiled_W.shape)
        for i in range(num_samples):
            for j in range(num_coeff):
                g = self.gradient_fn(tiled_W[i][j], X[i][j])
                g = update_mask[i][j] * g
                gradient_values[i][j] = g
        divisor = np.sum(update_mask, axis=0)
        divisor = np.reciprocal(divisor)
        divisor = np.where(divisor == np.inf, 0, divisor)
        divisor = divisor.reshape([-1, 1])
        # --------------------------------
        # Average gradients over the batch

        avg_gradients = np.multiply(np.sum(gradient_values, axis=0), divisor)

        # =================================
        # Calculate the projection of current gradient on each of the prior gradients for the same term 
        # =================================
        coeff_update_flag = np.sum(update_mask, axis=0)
        coeff_update_flag = np.where(coeff_update_flag > 0, True, False)
        cur_gradient = avg_gradients
        sum_grad_projections = []

        # ==================================
        # Create orthonormal basis if and only more than 2 prior vectors available
        # ==================================
        for i in range(num_coeff):
            _x = cur_gradient[i]
            # IF no update needed, store 0
            if not coeff_update_flag[i]:
                g_proj_i = np.zeros(_x.shape)
                sum_grad_projections.append(g_proj_i)
                continue

            # Gram Scmidt process : get the bases
            bases = np.array(self.prior_grad_vectors[i])

            if bases.shape[0] > 1:
                bases = gramSchmidt(bases)
                g_proj_i = np.zeros(_x.shape)
                # Add up sum of all projections
                for orth_base in bases:
                    _g_proj = np.dot(_x, orth_base) / np.linalg.norm(orth_base) * orth_base
                    g_proj_i += _g_proj
            else:
                g_proj_i = _x
            sum_grad_projections.append(g_proj_i)
        # --------
        # Add up the multiple projections
        sum_grad_projections = np.array(sum_grad_projections)
        final_gradient = sum_grad_projections

        # Save avg_gradients
        for i in range(num_coeff):
            if coeff_update_flag[i]:
                self.prior_grad_vectors[i].append(avg_gradients[i])

                # Update the weights
        W = W - final_gradient
        self.W_cur = W
        return final_gradient, W
