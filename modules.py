import numpy as np
from abc import ABC, abstractclassmethod


class Module(ABC):

    """
    Abstract class for all Modules
    """

    def __init__(self, *args, **kwargs) -> None:
        pass

    @abstractclassmethod
    def forward(self, *args):
        pass
    
    @abstractclassmethod
    def backward(self, *args):
        pass

    @abstractclassmethod
    def learnable(self):
        pass

    def __call__(self, *args):
        return self.forward(*args)


class Linear(Module):

    """
    Module for Linear Projections
    """

    def __init__(self, inshape: int, outshape: int):
        super().__init__()
        # init weights & biases
        self.params = [np.random.randn(inshape, outshape) * np.sqrt(2 / (inshape + outshape)), 
                        np.zeros((1, outshape))]

    def forward(self, x):
        # find dot product -> return dot prod
        self.x = x
        z = np.dot(self.x, self.params[0]) + self.params[1]
        return z

    def backward(self, grads, lr):
        # update params with their grads & lr applied
        for param, grad in zip(self.params, grads):
            param -= grad * lr
    
    def gradients(self, grad):
        # calc grads wrt to weights & biases -> calc grad wrt inputs
        grads = [np.dot(self.x.T, grad), 
                np.sum(grad, axis=0, keepdims=True)]
        grad = np.dot(grad, self.params[0].T)
        return grad, grads

    def learnable(self):
        return 2


class LayerNorm(Module):

    """
    Module for Layer Normalization
    """

    def __init__(self, features: int, eps=1e-9) -> None:
        # init learnable params
        super().__init__()
        self.eps = eps
        self.params = [np.ones((1, features)), 
                        np.zeros((1, features))]

    def forward(self, x):
        # calc norm metrics -> scale & shift with params
        self.x = x
        self.mean = np.mean(x, axis=-1, keepdims=True)
        self.meandev = x - self.mean
        self.var = np.power(self.meandev, 2).mean(axis=-1, keepdims=True)
        self.std = np.sqrt(self.var + self.eps)
        self.norm = (self.meandev) / self.std
        y = self.norm * self.params[0] + self.params[1]
        return y

    def backward(self, grads, lr):
        # update params with their grad & lr applied
        for param, grad in zip(self.params, grads):
            param -= grad * lr

    def gradients(self, grad):
        # calc grad wrt gamma & beta -> calc grad wrt to inputs

        # grad wrt to params
        grads = [np.sum(grad * self.norm, axis=0, keepdims=True), 
                np.sum(grad, axis=0, keepdims=True)]
        
        # grad wrt to inputs
        ones, n = np.ones(grad.shape), grad.shape[0]
        grad = self.params[0] * grad
        d_istd = np.sum(grad * self.meandev, axis=0, keepdims=True)
        d_meandev_1 = 1 / self.std * grad
        d_std = -1 / np.power(self.std, 2) * d_istd
        d_var = 1 / self.std * 0.5 * d_std
        d_meandevsq = 1 / n * ones * d_var
        d_meandev_2 = 2 * self.meandev * d_meandevsq
        d_mean = -1 * np.sum(d_meandev_1 + d_meandev_2, axis=0, keepdims=True)
        d_x1 = d_meandev_1 + d_meandev_2
        d_x2 = 1 / n * ones * d_mean
        grad = d_x1 + d_x2
        return grad, grads

    def learnable(self):
        return 2


class BatchNorm(Module):

    """
    Module for Batch Normalization
    """

    def __init__(self, features: int, eps=1e-9) -> None:
        # init learnable params
        super().__init__()
        self.eps = eps
        self.params = [np.ones((1, features)), 
                        np.zeros((1, features))]

    def forward(self, x):
        # calc norm metrics -> scale & shift with params
        self.x = x
        self.mean = np.mean(x, axis=0, keepdims=True)
        self.meandev = x - self.mean
        self.var = np.power(self.meandev, 2).mean(axis=0, keepdims=True)
        self.std = np.sqrt(self.var + self.eps)
        self.norm = (self.meandev) / self.std
        y = self.params[0] * self.norm + self.params[1]
        return y

    def backward(self, grads, lr):
        # update params with their grad & lr applied
        for param, grad in zip(self.params, grads):
            param -= grad * lr

    def gradients(self, grad):
        # calc grad wrt gamma & beta -> calc grad wrt to inputs

        # grad wrt params
        grads = [np.sum(grad * self.norm, axis=0, keepdims=True), 
                np.sum(grad, axis=0, keepdims=True)]
        
        # grad wrt to inputs
        ones, n = np.ones(grad.shape), grad.shape[0]
        grad = self.params[0] * grad
        d_istd = np.sum(grad * self.meandev, axis=0, keepdims=True)
        d_meandev_1 = 1 / self.std * grad
        d_std = -1 / np.power(self.std, 2) * d_istd
        d_var = 1 / self.std * 0.5 * d_std
        d_meandevsq = 1 / n * ones * d_var
        d_meandev_2 = 2 * self.meandev * d_meandevsq
        d_mean = -1 * np.sum(d_meandev_1 + d_meandev_2, axis=0, keepdims=True)
        d_x1 = d_meandev_1 + d_meandev_2
        d_x2 = 1 / n * ones * d_mean
        grad = d_x1 + d_x2
        return grad, grads

    def learnable(self):
        return 2


class Dropout(Module):

    """
    Module for Dropping Neurons
    """
    
    def __init__(self, p=0.5):
        # init prob of drop
        super().__init__()
        self.p = p

    def forward(self, x):
        # select random neurons to drop -> drop neurons
        neurons = x.shape[1]
        n = int(neurons * self.p)
        drops = np.random.choice(neurons, n, replace=False)
        x[:, drops] = 0
        return x
        
    def backward(self, grad):
        # no grad just return the gradient
        return grad

    def learnable(self):
        return False


class Flatten(Module):

    """
    Module for Flattening Inputs
    """
    
    def forward(self, x):
        super().__init__()
        # reshape to (batch_size, neurons)
        batch_size = len(x)
        return x.reshape(batch_size, -1)

    def backward(self, grad):
        # no grad just return the gradient
        return grad

    def learnable(self):
        return False


if __name__ == "__main__":
    pass