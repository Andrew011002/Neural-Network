from typing import Iterable
import numpy as np
from numpy import ndarray


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
    return np.argmax(y, axis=1)
    

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

    # TESTING ONE HOT (2 DIMS)
    y = np.random.randint(0, 5, (5, 1))
    y = onehot(y)
    print(y)

    # TESTING ONE HOT (1 DIM)
    y = np.random.randint(0, 5, (5,))
    y = onehot(y)
    print(y)

    # TESTING UNENCODE
    y = unhot(y)
    print(y)

    # TESTING ACC (2 DIMS)
    p = np.random.rand(5, 3)
    y = np.random.randint(0, 3, (5, 1))
    acc = accuracy(p, y)
    print(acc)
    
    # TESTING ACC (1 DIM)
    y = np.squeeze(y, axis=-1)
    acc = accuracy(p, y)
    print(acc)

    # TESTING ACC (BINARY 2 DIM)
    p = np.random.rand(16, 1)
    y = np.random.choice(2, (16, 1))
    acc = accuracy(p, y)
    print(acc)

    # TESTING ACC (BINARY 1 DIM)
    y = y.squeeze(axis=-1)
    acc = accuracy(p, y)
    print(acc)

    # TESTING NORMALIZATION
    x = np.random.randint(-10000, 10000, (64, 3))
    x = normalize(x)
    print(x.shape)

    # TESTING IS_ONEHOT
    y = np.random.choice(5, (100,))
    y = onehot(y)
    print(is_onehot(y))
    
    y = np.random.choice(5, (100,))
    print(is_onehot(y))

    y = y.reshape(-1, 1)
    print(is_onehot(y))

    # TESTING IS_SPARSE
    y = onehot(y)
    print(is_sparse(y))

    y = np.random.choice(5, (100,))
    print(is_sparse(y))

    y = y.reshape(-1, 1)
    print(is_sparse(y))
