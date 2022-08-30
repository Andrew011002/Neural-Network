import numpy as np
from collections import deque
from abc import ABC, abstractclassmethod


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

    def __init__(self, lr):
        pass


if __name__ == "__main__":
    pass
    
    

    

        



        


