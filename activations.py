import numpy as np
from numpy import ndarray
from modules import Module


class ReLU(Module):

    """
    Rectified Linear Unit Activation Function
    """

    def __init__(self):
        pass

    def forward(self, z):
        self.z = z
        return np.maximum(0, z)

    def backward(self, grad):
        return np.where(self.z > 0, 1, 0) * grad

    def learnable(self):
        return False


class Sigmoid(Module):

    """
    Sigmoid Activation Function
    """

    def __init__(self):
        pass
    
    def forward(self, z):
        self.z = z
        return sigmoid(z)

    def backward(self, grad):
        return sigmoid(self.z) * (1 - sigmoid(self.z)) * grad

    def learnable(self):
        return False

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


class Softmax(Module):

    """
    Softmax Activation Function
    """

    def __init__(self, axis=-1):
        self.axis = axis

    def forward(self, z):
        return softmax(z, self.axis)

    def backward(self, grad):
        return grad
    
    def learnable(self):
        return False

def softmax(z, axis):
    exp = np.exp(z - np.max(z, axis=axis, keepdims=True))
    norm = np.sum(exp, axis=axis, keepdims=True)
    return exp / norm


class Tanh(Module):

    """
    Hyperbolic Tangent Activation Function
    """

    def __init__(self):
        pass

    def forward(self, z):
        self.z = z
        return tanh(z)

    def backward(self, grad):
        return 1 - np.power(tanh(self.z), 2) * grad

    def learnable(self):
        return False

def tanh(z):
    exp1, exp2 = np.exp(z), np.exp(-z)
    return (exp1 - exp2) / (exp1 + exp2)

if __name__ == '__main__':
    pass
    
    


    