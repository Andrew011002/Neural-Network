import numpy as np
from typing import Iterable


def is_onehot(y):
    if y.ndim != 2:
        return False

    if len(np.unique(y[0])) != 2:
        return False

    return True

def is_sparse(y):

    if is_onehot(y):
        return False

    return True


def onehot(y: Iterable[int]):
    # encode to one-hot
    y = np.array(y, dtype=np.uint8)
    # force 1d array
    if y.ndim > 1:
        y = np.squeeze(y, axis=-1)
    # sparse -> one-hot
    encoded = np.zeros((y.size, y.max() + 1))
    encoded[np.arange(y.size), y] = 1
    return encoded

def unhot(y: Iterable[int]):
    # return max index along the columns as 1D ndarray
    return np.argmax(y, axis=-1)
    

def accuracy(p: Iterable[Iterable[float]], y: Iterable[int]):
    # convert calculate categorical or binary class accuracy
    y = np.array(y, dtype=np.int64)
    # unecode
    if is_onehot(y):
        y = unhot(y)
    # force 1d array
    if y.ndim > 1:        
        y = np.squeeze(y, axis=-1)
    # categorical accuracy
    if p.shape[1] > 1:
        return np.sum(np.argmax(p, axis=1) == y) / len(p)
    # binary accuracy
    return np.sum(np.squeeze(np.rint(p), axis=-1) == y) / len(p)


def normalize(x: Iterable[Iterable[float]], axis=-1):
    # normalize across desired axis
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / norm

if __name__ == "__main__":
    pass