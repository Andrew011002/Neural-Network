import numpy as np
from utils import unhot
from activations import softmax
from abc import ABC, abstractclassmethod


class Loss(ABC):

    """
    Base class for all Loss Functions
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

    @abstractclassmethod
    def forward(self, *args):
        pass

    @abstractclassmethod
    def backward(self, *args):
        pass

    def __call__(self, *args):
        return self.forward(*args)


class MSE(Loss):

    """
    Mean Squared Error Loss Function
    """

    def __init__(self):
        super().__init__()

    def forward(self, p, y):
        # store inputs
        self.p, self.y = p, y
        return mse(p, y)

    def backward(self):
        y, p = self.y, self.p
        # force 2d array
        if y.ndim != 2:
            y = y.reshape(-1, 1)
        # calc avg grad of mse wrt pred & labels
        grad = 2 * (p - y)
        return grad / len(p)

def mse(p, y):
    # force 2d array
    if y.ndim != 2:
        y = y.reshape(-1, 1)
    # calc mse w/ pred & labels
    loss = np.power(p - y, 2)
    return np.mean(loss, axis=0).item()


class BCE(Loss):

    """
    Binary Cross-Entropy Loss Function
    """

    def __init__(self):
        super().__init__()

    def forward(self, p, y):
        self.p, self.y = p, y
        return binary_crossentropy(p, y)

    def backward(self):
        p, y = self.p, self.y
        # force 2d array
        if y.ndim != 2:
            y = y.reshape(-1, 1)
        # calc avg grad of bce wrt pred & labels 
        grad = -y / p + (1 - y) / (1 - p)
        return grad / len(p)
        
def binary_crossentropy(p, y):
    # force 2d array
    if y.ndim != 2:
        y = y.reshape(-1, 1)
    p = np.clip(p, 1e-7, 1 - 1e-7) # clip too small/large vals
    # calc mean bce w/ pred & labels
    loss = -y * np.log(p) - (1 - y) * np.log(1 - p)
    return np.mean(loss, axis=0).item()

def binary_crossentropy_2(p, y):
    # force 2d array
    if y.ndim != 2:
        y = y.reshape(-1, 1)
    p = np.clip(p, 1e-7, 1 - 1e-7) # clip too small/large vals
    # calc mean bce w/ pred & labels
    loss = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    return np.mean(loss, axis=0).item()


class CCE(Loss):

    """
    Categorical Cross Entropy Loss Function
    """

    def __init__(self):
        super().__init__()

    def forward(self, y, p):
        self.p, self.y = p, y
        return cross_entropy(p, y)

    def backward(self):
        p, y = self.p, self.y
        # force 2d array
        if y.ndim != 2:
            y = y.reshape(-1, 1)
        p = softmax(p, axis=-1) # apply sm
        p = np.clip(p, 1e-7, 1 - 1e-7) # clip too small/large vals
        # calc avg grad of cce wrt to inputs & labels
        grad = p - y
        return grad / len(p)

def cross_entropy(p, y):
    # force 2d array
    if y.ndim != 2:
        y = y.reshape(-1, 1)
    p = softmax(p, axis=-1) # apply sm
    p = np.clip(p, 1e-7, 1 - 1e-7) # clip too small/large vals
    # calc mean cce w/ pred & labels
    loss = np.sum(-(y * np.log(p)), axis=1, keepdims=True)
    return np.mean(loss, axis=0).item()


class SCCE(Loss):

    """
    Sparse Categorical Cross-Entropy Loss Function
    """

    def __init__(self):
        super().__init__()

    def forward(self, p, y):
        self.p, self.y = p, y
        return sparse_cross_entropy(p, y)

    def backward(self):
        p, y = self.p, self.y
        # force 1d array
        if y.ndim != 1:
            y = np.squeeze(y, axis=-1)
        p = softmax(p, axis=-1) # apply sm
        p = np.clip(p, 1e-7, 1 - 1e-7) # clip too small/large values
        # calc avg grad of scce wrt pred & labels 
        p[np.arange(len(p)), y] -= 1
        grad = p
        return grad / len(p)       

def sparse_cross_entropy(p, y):
    # force 1d array
    if y.ndim != 1:
        y = np.squeeze(y, axis=-1)
    p = softmax(p, axis=-1) # apply softmax
    p = np.clip(p, 1e-7, 1 - 1e-7) # clip too small/large values
    # calc mean scce of pred & labels
    loss = -np.log(p[np.arange(len(p)), y])
    return np.mean(loss, axis=0).item()


class NLL(Loss):

    """
    Negative Log-Likelihood Loss Function
    """

    def __init__(self):
        super().__init__()

    def forward(self, p, y):
        self.p, self.y = p, y
        return nll(p, y)
    
    def backward(self):
        p, y = self.p, self.y
        p = np.clip(p, 1e-7, 1 - 1e-7) # clip too small/large values
        # calc avg grad of nll wrt to pred & labels
        grad = p - y
        return grad / len(p)

def nll(p, y):
    p = np.clip(p, 1e-7, 1 - 1e-7) # clip too small/large values
    # calc mean nll for pred & labels
    likelihood = -np.log(p[np.arange(len(p)), unhot(y)])
    loss = np.mean(likelihood, axis=0).item()
    return loss

    
if __name__ == '__main__':
    pass