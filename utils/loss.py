import numpy as np

def loss(X, Z, W):
    L = np.linalg(X - np.matmul(W, Z))
    P = size(L)

    return L
