import numpy as np


def calculate_cosineDist_gradient(W, X):
    norm_X = np.linalg.norm(X)
    norm_W = np.linalg.norm(W)

    a = np.dot(W, X) / (norm_X * norm_W)
    b = W / norm_W

    c = X / (norm_W * norm_X)
    return a * b - c


def calculate_dotProd_gradient(W, X):
    return -X


# ===============================
# We want W and X to align so, the "gradient" in this case is the cosine distance
# ===============================
def cosine_loss(X, Y):
    return Y - X

    xnorm = np.sqrt(np.sum(X * X))
    ynorm = np.sqrt(np.sum(Y * Y))
    similarity = np.sum(X * Y) / (xnorm * ynorm)
    return 1 - similarity


# -----------------
# W : weight
# X is the input
# gradient to be applied to W : W` = W-grad
# ------------------
def maxDotProd_gradient(X, W):
    d = np.dot(X, W) / np.linalg.norm(X) * X
    grad = W - d
    return -grad
