import numpy as np


def calculate_cosineDist_gradient(W,X):
    norm_X = np.linalg.norm(X)
    norm_W = np.linalg.norm(W)

    a = np.dot(W,X)/(norm_X*norm_W)
    b = W / norm_W
   
    c = X/(norm_W*norm_X)
    return a*b -c 

def calculate_dotProd_gradient(W,X):
    return -X 

# ===============================
# We want W and X to align so, the "gradient" in this case is the cosine distance
# ===============================
def cosine_loss(X, Y):
    return Y-X
    xnorm = np.sqrt(np.sum(X*X))
    ynorm = np.sqrt(np.sum(Y*Y))
    similarity = np.sum(X*Y) / (xnorm * ynorm)
    return 1 - similarity

# --------------------------------------------------

def cosine_loss_grad(X, Y, b_return=1.0):
    X_times_X = X * X
    _xnorm = np.sum(X_times_X)
    xnorm = np.sqrt(_xnorm)
    Y_times_Y = Y * Y
    _ynorm = np.sum(Y_times_Y)
    ynorm = np.sqrt(_ynorm)
    _similarity2 = xnorm * ynorm
    X_times_Y = X * Y
    _similarity = np.sum(X_times_Y)
    similarity = _similarity / _similarity2
    _return = 1 - similarity
    assert tangent.shapes_match(_return, b_return
        ), 'Shape mismatch between return value (%s) and seed derivative (%s)' % (
        numpy.shape(_return), numpy.shape(b_return))

    # Grad of: _similarity = np.sum(X_times_Y)
    _bsimilarity = -tangent.unbroadcast(b_return, similarity)
    bsimilarity = _bsimilarity

    # Grad of: similarity = np.sum(X * Y) / (xnorm * ynorm)
    _b_similarity = bsimilarity / _similarity2
    _b_similarity2 = -bsimilarity * _similarity / (_similarity2 * _similarity2)
    b_similarity = _b_similarity
    b_similarity2 = _b_similarity2
    _bX_times_Y = tangent.astype(tangent.unreduce(b_similarity, numpy.shape
        (X_times_Y), None, False), X_times_Y)
    bX_times_Y = _bX_times_Y
    _bX3 = tangent.unbroadcast(bX_times_Y * Y, X)
    bX = _bX3
    _bxnorm = tangent.unbroadcast(b_similarity2 * ynorm, xnorm)
    bxnorm = _bxnorm

    # Grad of: xnorm = np.sqrt(np.sum(X * X))
    _xnorm2 = xnorm
    _b_xnorm = bxnorm / (2.0 * _xnorm2)
    b_xnorm = _b_xnorm
    _bX_times_X = tangent.astype(tangent.unreduce(b_xnorm, numpy.shape(
        X_times_X), None, False), X_times_X)
    bX_times_X = _bX_times_X
    _bX = tangent.unbroadcast(bX_times_X * X, X)
    _bX2 = tangent.unbroadcast(bX_times_X * X, X)
    bX = tangent.add_grad(bX, _bX)
    bX = tangent.add_grad(bX, _bX2)
    return bX
