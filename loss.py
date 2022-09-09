import numpy as np
from utils import onehot, unhot
from activations import softmax

class Loss:

    """
    Base class for all Loss Functions
    """

    def __init__(self, loss, derivative):
        # init loss function & it
        self.loss = loss
        self.derivative = derivative

    def __call__(self, p, y):
        # set pred & label (for backward) -> apply loss fn
        self.p = p
        self.y = y
        return self.loss(p, y)

    def backward(self):
        # apply derivative to pred & labels
        return self.derivative(self.p, self.y)



class MSE(Loss):

    """
    Mean Squared Error Loss Function
    """

    def __init__(self):
        super().__init__(mse, mse_d)

def mse(p, y):
    # force 2d array
    if y.ndim != 2:
        y = y.reshape(-1, 1)
    # calc mse w/ pred & labels
    loss = np.power(p - y, 2)
    return np.mean(loss, axis=0).item()

def mse_d(p, y):
    # force 2d array
    if y.ndim != 2:
        y = y.reshape(-1, 1)
    # calc avg grad of mse wrt pred & labels
    grad = 2 * (p - y)
    return grad / len(p)



class BCE(Loss):

    """
    Binary Cross-Entropy Loss Function
    """

    def __init__(self):
        super().__init__(binary_crossentropy, binary_crossentropy_d)

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

def binary_crossentropy_d(p, y):
    # force 2d array
    if y.ndim != 2:
        y = y.reshape(-1, 1)
    # calc avg grad of bce wrt pred & labels 
    grad = -y / p + (1 - y) / (1 - p)
    return grad / len(p)



class CCE(Loss):

    """
    Categorical Cross Entropy Loss Function
    """

    def __init__(self):
        super().__init__(cross_entropy, cross_entropy_d)

def cross_entropy(p, y):
    # force 2d array
    if y.ndim != 2:
        y = y.reshape(-1, 1)
    p = softmax(p) # apply sm
    p = np.clip(p, 1e-7, 1 - 1e-7) # clip too small/large vals
    # calc mean cce w/ pred & labels
    loss = np.sum(-(y * np.log(p)), axis=1, keepdims=True)
    return np.mean(loss, axis=0).item()

def cross_entropy_d(p, y):
    # force 2d array
    if y.ndim != 2:
        y = y.reshape(-1, 1)
    p = softmax(p) # apply sm
    p = np.clip(p, 1e-7, 1 - 1e-7) # clip too small/large vals
    # calc avg grad of cce wrt to inputs & labels
    grad = p - y
    return grad / len(p)


class SCCE(Loss):

    """
    Sparse Categorical Cross-Entropy Loss Function
    """

    def __init__(self):
        super().__init__(sparse_cross_entropy, sparse_cross_entropy_d)

def sparse_cross_entropy(p, y):
    # force 1d array
    if y.ndim != 1:
        y = np.squeeze(y, axis=-1)
    p = softmax(p) # apply softmax
    p = np.clip(p, 1e-7, 1 - 1e-7) # clip too small/large values
    # calc mean scce of pred & labels
    loss = -np.log(p[np.arange(len(p)), y])
    return np.mean(loss, axis=0).item()

def sparse_cross_entropy_d(p, y):
    # force 1d array
    if y.ndim != 1:
        y = np.squeeze(y, axis=-1)
    p = softmax(p) # apply sm
    p = np.clip(p, 1e-7, 1 - 1e-7) # clip too small/large values
    # calc avg grad of scce wrt pred & labels 
    p[np.arange(len(p)), y] -= 1
    grad = p
    return grad / len(p)



class NLL(Loss):

    """
    Negative Log-Likelihood Loss Function
    """

    def __init__(self):
        super().__init__(nll, nll_d)

def nll(p, y):
    # force 1d array
    if y.ndim != 1:
        y = np.squeeze(y, axis=-1)
    p = np.clip(p, 1e-7, 1 - 1e-7) # clip too small/large values
    # calc mean nll for pred & labels
    likelihood = -np.log(p[np.arange(len(p)), y])
    loss = np.mean(likelihood, axis=0).item()
    return loss


def nll_d(p, y):
    y = onehot(y) # make one-hot
    p = np.clip(p, 1e-7, 1 - 1e-7) # clip too small/large values
    # calc avg grad of nll wrt to pred & labels
    grad = -y / p
    return grad / len(p)

    
        

if __name__ == '__main__':
    # DATA
    p = np.random.rand(16, 1)
    y = np.random.choice(2, (16, ))

    # MSE EXAMPLE (1 DIM)
    error = mse(p, y)
    print(f"MSE: {error}")
    grad = mse_d(p, y)
    print(f"Gradient shape: {grad.shape}")

    # MSE EXAMPLE (2 DIM)
    y = y.reshape(-1, 1)
    error = mse(p, y)
    print(f"MSE: {error}")
    grad = mse_d(p, y)
    print(f"Gradient shape: {grad.shape}")

    # BINARY CROSS ENTROPY EXAMPLE (1 DIM)
    y = np.squeeze(y, axis=-1)
    error = binary_crossentropy(p, y)
    print(f"BCE: {error}")
    grad = binary_crossentropy_d(p, y)
    print(f"Gradient shape: {grad.shape}")

    # BINARY CROSS ENTROPY EXAMPLE (2 DIM)
    y = y.reshape(-1, 1)
    error = binary_crossentropy(p, y)
    print(f"BCE: {error}")
    grad = binary_crossentropy_d(p, y)
    print(f"Gradient shape: {grad.shape}")

    # CROSS ENTROPY LOSS EXAMPLE (1 DIM)
    p = np.random.rand(16, 3)
    y = np.random.randint(0, 3, (16, ))
    y_onehot = onehot(y)
    error = cross_entropy(p, y_onehot)
    print(f"CCE: {error}")
    grad = cross_entropy_d(p, y_onehot)
    print(f"Gradient shape: {grad.shape}")

    # CROSS ENTROPY LOSS EXAMPLE (2 DIM)
    y = unhot(y_onehot).reshape(-1, 1)
    y_onehot = onehot(y)
    error = cross_entropy(p, y_onehot)
    print(f"CCE: {error}")
    grad = cross_entropy_d(p, y_onehot)
    print(f"Gradient shape: {grad.shape}")

    # SPARSE CROSS ENTROPY LOSS EXAMPLE (1 DIM)
    y = np.squeeze(y, axis=-1)
    error = sparse_cross_entropy(p, y)
    print(f"SCCE: {error}")
    grad = sparse_cross_entropy_d(p, y)
    print(f"Gradient shape: {grad.shape}")

    # SPARSE CROSS ENTROPY LOSS EXAMPLE (2 DIM)
    y = y.reshape(-1, 1)
    error = sparse_cross_entropy(p, y)
    print(f"SCCE: {error}")
    grad = sparse_cross_entropy_d(p, y)
    print(f"Gradient shape: {grad.shape}")

    # NEGATIVE LOG LIKELIHOOD LOSS EXAMPLE (1 DIM)
    y = np.squeeze(y, axis=-1)
    error = nll(p, y)
    print(f"NLL: {error}")
    grad = nll_d(p, y)
    print(f"Gradient shape: {grad.shape}")



