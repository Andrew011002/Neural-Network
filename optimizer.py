import numpy as np
from collections import deque
from abc import ABC, abstractclassmethod
import sys


class Optimizer(ABC):

    @abstractclassmethod
    def __init__(self, *args, **kwargs) -> None:
        pass

    @abstractclassmethod
    def update(self, *args, **kwargs):
        pass


class SGD(Optimizer):

    def __init__(self, params, lr=0.01):
        super().__init__()
        self.lr = lr
        self.params = params

    def update(self, grad):

        for param in reversed(self.params):
            
            if param.learnable():
                grad, grads = param.gradients(grad)
                param.backward(grads, self.lr)
            else:
                grad = param.backward(grad)

class SGDM(Optimizer):

    def __init__(self, params, lr=0.01, momentum=0.9):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.grads = self.store_grads(params)

    def update(self, grad):
        
        for param in reversed(self.params):

            if param.learnable():

                grad, grads = param.gradients(grad)
                for i, wrt in enumerate(grads):
                    grads[i] = self.momentum * self.grads[0] + (1 - self.momentum) * wrt
                    self.grads.append(grads[i])
                param.backward(grads, self.lr)

            else:
                param.backward(grad)

    def store_grads(self, params):
        n_grads = sum([param.learnable() for param in params]) * 2
        return deque(np.zeros(n_grads), n_grads)


class Adam(Optimizer):

    def __init__(self, params, lr=0.01, betas=(0.9, 0.999), eps=1e-9):
        self.params = params
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.grads = self.store_grads(params)
    
    def update(self, grad):

        for param in reversed(self.params):

            if param.learnable():

                grad, grads = param.gradients(grad)
                for i, wrt in enumerate(grads):
                    # calculate first & second moments
                    m, v, t = self.grads[0]
                    m = self.betas[0] * m + (1 - self.betas[0]) * wrt
                    v = self.betas[1] * v + (1 - self.betas[1]) * np.power(wrt, 2)
                    self.grads.append((m, v, t + 1)) # store moments & time

                    # correct bias for moments
                    m = m / (1 - np.power(self.betas[0], int(t)))
                    v = v / (1 - np.power(self.betas[1], int(t)))
                    grads[i] = self.lr * m / (np.sqrt(v) + self.eps) # store delta w for param
                param.backward(grads, 1) # ignore lr (used above)
                    
            else:
                grad = param.backward(grad)
    
    def store_grads(self, params):
        n_grads = sum([param.learnable() for param in params]) * 2
        values = np.zeros((n_grads, 3))
        values[:, 2] = 1
        grads = deque(values, n_grads)
        return grads


if __name__ == "__main__":
    b = np.zeros((1, 5))
    print(b + 1)

    
    
    

    

        



        


