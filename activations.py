import numpy as np


class _Activation:

    """
    Base class for all Activation Functions
    """

    def __init__(self, activation, derivative):
        # init activation function & derivative
        self.activation = activation
        self.derivative = derivative

    def __call__(self, z):
        # activate input
        return self.activation(z)
        


class ReLU(_Activation):

    """
    Rectified Linear Unit Activation Function
    """

    def __init__(self):
        super().__init__(relu, relu_d)

def relu(z):
    return np.maximum(0, z)

def relu_d(z):
    return np.where(z > 0, 1, 0)



class Sigmoid(_Activation):

    """
    Sigmoid Activation Function
    """

    def __init__(self):
        super().__init__(sigmoid, sigmoid_d)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_d(z):
    return sigmoid(z) * (1 - sigmoid(z))



class Softmax(_Activation):

    """
    Softmax Activation Function (NLL Loss only)
    """

    def __init__(self):
        super().__init__(softmax, softmax_d)

def softmax(z):
    exp = np.exp(z - np.max(z, axis=1, keepdims=True))
    norm = np.sum(exp, axis=1, keepdims=True)
    return exp / norm

def softmax_d(z):
    grad = 1
    return grad


class Tanh(_Activation):

    """
    Hyperbolic Tangent Activation Function
    """

    def __init__(self):
        super().__init__(tanh, tanh_d)

def tanh(z):
    exp1, exp2 = np.exp(z), np.exp(-z)
    return (exp1 - exp2) / (exp1 + exp2)

def tanh_d(z):
    grad = 1 - np.power(tanh(z), 2)
    return grad

if __name__ == '__main__':
    
    # sigmoid test
    inputs = np.random.randint(-10, 11, (64, 1))
    out = sigmoid(inputs)
    print(out.shape)
    grad = sigmoid_d(inputs)
    print(grad.shape)

    # softmax test
    inputs = np.random.rand(3, 3)
    out = softmax(inputs)
    print(out.shape)
    grad = softmax_d(inputs)
    print(grad)

    # relu test
    inputs = np.random.randint(-10, 11, (64, 8))
    out = relu(inputs)
    print(out.shape)
    grad = relu_d(inputs)
    print(grad.shape)
    
    


    