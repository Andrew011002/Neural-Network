import numpy as np
from collections import deque
from abc import ABC, abstractclassmethod
from modules import Module
from typing import Iterable


class Optimizer(ABC):

    """
    Base class for Optimizer Algorithms
    """

    @abstractclassmethod
    def __init__(self, *args, **kwargs) -> None:
        pass

    @abstractclassmethod
    def update(self, *args, **kwargs):
        pass


class SGD(Optimizer):

    """
    Stochastic Gradient Descent Optimization Algorithm
    """

    def __init__(self, parameters: Iterable[Module], lr=0.01):
        # init params & lr
        super().__init__()
        self.lr = lr
        self.params = parameters

    def update(self, grad):
        # back propagate grad through params
        for param in reversed(self.params):
            # updating learnables params
            if param.learnable():
                grad, grads = param.gradients(grad)
                param.backward(grads, self.lr)
            # passing grad non-learnable params
            else:
                grad = param.backward(grad)

class SGDM(Optimizer):

    """
    Stochastic Gradient Descent with Momentum
    Optimization Algorithm
    """

    def __init__(self, parameters: Iterable[Module], lr=0.01, momentum=0.9):
        # init params, lr, & momentum -> create initial moments
        self.params = parameters
        self.lr = lr
        self.momentum = momentum
        self.grads = self.store_grads(parameters)

    def update(self, grad):
        # back propagate grad through params
        for param in reversed(self.params):
            # updating learnables params
            if param.learnable():
                # find moment for learnable params of current params
                grad, grads = param.gradients(grad)
                for i, wrt in enumerate(grads):
                    grads[i] = self.momentum * self.grads[0] + (1 - self.momentum) * wrt
                    self.grads.append(grads[i])
                param.backward(grads, self.lr)
            # passing grad non-learnable params
            else:
                param.backward(grad)

    def store_grads(self, params):
        # create queue that stores moments for params (init w/ 0s)
        n_grads = sum([param.learnable() for param in params]) * 2
        return deque(np.zeros(n_grads), n_grads)


class Adam(Optimizer):

    """
    Adam Optimization Algorithm
    """

    def __init__(self, parameters: Iterable[Module], lr=0.01, betas=(0.9, 0.999), eps=1e-8):
        # init params, lr, betas, & episolon -> create initial moments
        self.params = parameters
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.grads = self.store_grads(parameters)
    
    def update(self, grad):
        # back propagate grad through params
        for param in reversed(self.params):
            # updating learnables params
            if param.learnable():
                # find first & second moments for learnable params of current param
                grad, grads = param.gradients(grad)
                for i, wrt in enumerate(grads):
                    # calculate first & second moments
                    beta_1, beta_2 = self.betas
                    m, v, t = self.grads[0]
                    m = beta_1 * m + (1 - beta_1) * wrt
                    v = beta_2 * v + (1 - beta_2) * np.power(wrt, 2)
                    self.grads.append((m, v, t + 1)) # store moments & time
                    # correct bias for moments
                    m = m / (1 - beta_1 ** int(t))
                    v = v / (1 - beta_2 ** int(t))
                    grads[i] = self.lr * m / (np.sqrt(v) + self.eps) # store delta w for param
                param.backward(grads, 1) # ignore lr (used above)
            # passing grad non-learnable params
            else:
                grad = param.backward(grad)
    
    def store_grads(self, params):
        # create that stores first & second moments of learnable params (init w/ 0s)
        n_grads = sum([param.learnable() for param in params]) * 2
        values = np.zeros((n_grads, 3))
        values[:, 2] = 1
        grads = deque(values, n_grads)
        return grads


if __name__ == "__main__":
    pass
    
    

    

    
    
    

    

        



        


